"""
Inference Script — Joint NER + RE  (UPDATED)
=============================================
Loads the trained model and label map from the model/ subfolder.
Reads test samples from splits/test.jsonl.
Saves all predictions + evaluation to results/inference_results.json
and a human-readable results/inference_report.txt.

THREE evaluation tasks are computed:

  Task 1 — Binary classification (paper level)
           Did the model correctly detect whether ANY AI tool was used?
           Evaluated with accuracy, precision, recall, F1.
           No exact/partial split — it is a binary yes/no per paper.

  Task 2 — Multi-label entity extraction (entity level, micro F1)
           2a. Tool Entity   : which tool names are mentioned?
           2b. Usage Entity  : which usage descriptions are mentioned?
           Both evaluated under EXACT and PARTIAL match.

  Task 3 — Relation extraction: used_for (tool-paper pair level)
           For each tool in a paper, did the model assign the correct
           predefined contribution role(s)?
           Evaluated under EXACT and PARTIAL match.
           NOTE: Task 3 performance is directly bounded by Task 2 —
           better entity extraction leads to better relation scores.

Match criteria:
  Exact   — predicted span text must match ground truth text exactly
             (case-insensitive, whitespace stripped)
  Partial — predicted character span overlaps ground truth span by
             at least one character
"""

import json
import os
import sys
from collections import defaultdict

import torch
import torch.nn as nn
from transformers import BertTokenizerFast, BertModel

# ─────────────────────────────────────────────
# 1.  PATHS
# ─────────────────────────────────────────────
BERT_DIR = r"path/to/bert_output"   # same OUTPUT_DIR used during training

MODEL_DIR    = os.path.join(BERT_DIR, "model")
SPLITS_DIR   = os.path.join(BERT_DIR, "splits")
RESULTS_DIR  = os.path.join(BERT_DIR, "results")

TOKENIZER_PATH    = os.path.join(MODEL_DIR, "tokenizer")
MODEL_WEIGHTS     = os.path.join(MODEL_DIR, "best_model.pt")
LABEL_MAP_PATH    = os.path.join(MODEL_DIR, "label_map.json")
MODEL_CONFIG_PATH = os.path.join(MODEL_DIR, "model_config.json")
TEST_DATA_PATH    = os.path.join(SPLITS_DIR, "test.jsonl")

RESULTS_JSON = os.path.join(RESULTS_DIR, "inference_results.json")
RESULTS_TXT  = os.path.join(RESULTS_DIR, "inference_report.txt")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─────────────────────────────────────────────
# 2.  LOAD LABEL MAPS
# ─────────────────────────────────────────────

def load_label_maps(label_map_path: str):
    with open(label_map_path, "r", encoding="utf-8") as f:
        lm = json.load(f)

    head1_labels = lm["head1_labels"]
    head2_labels = lm["head2_labels"]
    rel_labels   = lm["rel_labels"]

    head1_to_id = {l: i for i, l in enumerate(head1_labels)}
    id_to_head1 = {i: l for l, i in head1_to_id.items()}

    head2_to_id = {l: i for i, l in enumerate(head2_labels)}
    id_to_head2 = {i: l for l, i in head2_to_id.items()}

    rel_to_id = {l: i for i, l in enumerate(rel_labels)}
    id_to_rel = {i: l for l, i in rel_to_id.items()}

    return (
        head1_labels, head1_to_id, id_to_head1,
        head2_labels, head2_to_id, id_to_head2,
        rel_labels,   rel_to_id,   id_to_rel,
    )

# ─────────────────────────────────────────────
# 3.  MODEL
# ─────────────────────────────────────────────

class JointNER_RE(nn.Module):
    def __init__(self, model_name, n_head1, n_head2, n_rel, dropout=0.1):
        super().__init__()
        self.bert  = BertModel.from_pretrained(model_name)
        hidden     = self.bert.config.hidden_size

        self.head1 = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, n_head1))
        self.head2 = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, n_head2))
        self.head3 = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_rel),
        )

    def forward(self, input_ids, attention_mask, token_type_ids):
        out     = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids)
        seq_out = out.last_hidden_state
        return seq_out, self.head1(seq_out), self.head2(seq_out)

    def classify_relation(self, seq_out, tool_range, pred_range):
        tool_emb = seq_out[tool_range[0]:tool_range[1]].mean(0)
        pred_emb = seq_out[pred_range[0]:pred_range[1]].mean(0)
        return self.head3(torch.cat([tool_emb, pred_emb]).unsqueeze(0))

# ─────────────────────────────────────────────
# 4.  DATA LOADING
# ─────────────────────────────────────────────

def load_test_jsonl(path: str):
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples

# ─────────────────────────────────────────────
# 5.  SPAN EXTRACTION
# ─────────────────────────────────────────────

def extract_spans(label_ids, id_to_label):
    spans, cur_lbl, cur_start = [], None, None
    for i, lid in enumerate(label_ids):
        tag = id_to_label[lid]
        if tag.startswith("B-"):
            if cur_lbl is not None:
                spans.append((cur_start, i, cur_lbl))
            cur_lbl, cur_start = tag[2:], i
        elif tag.startswith("I-") and cur_lbl == tag[2:]:
            pass
        else:
            if cur_lbl is not None:
                spans.append((cur_start, i, cur_lbl))
            cur_lbl, cur_start = None, None
    if cur_lbl is not None:
        spans.append((cur_start, len(label_ids), cur_lbl))
    return spans

# ─────────────────────────────────────────────
# 6.  PREDICTION
#     predict() now returns BOTH tool spans and usage spans from Head 1,
#     plus category spans from Head 2 and relations from Head 3.
# ─────────────────────────────────────────────

def predict(text, model, tokenizer, id_to_head1, id_to_head2, id_to_rel, max_len=256):
    model.eval()
    enc = tokenizer(
        text,
        max_length=max_len,
        truncation=True,
        return_offsets_mapping=True,
        padding="max_length",
        return_tensors="pt",
    )
    offsets = enc["offset_mapping"][0].tolist()

    with torch.no_grad():
        seq_out, logits_h1, logits_h2 = model(
            enc["input_ids"].to(DEVICE),
            enc["attention_mask"].to(DEVICE),
            enc["token_type_ids"].to(DEVICE),
        )

    pred_h1 = logits_h1[0].argmax(-1).cpu().tolist()
    pred_h2 = logits_h2[0].argmax(-1).cpu().tolist()
    seq_out  = seq_out[0]

    valid         = [i for i, (s, e) in enumerate(offsets) if s != e]
    pred_h1_v     = [pred_h1[i] for i in valid]
    pred_h2_v     = [pred_h2[i] for i in valid]
    valid_offsets = [offsets[i] for i in valid]

    # Extract Tool spans and Usage spans separately from Head 1
    all_h1_spans  = extract_spans(pred_h1_v, id_to_head1)
    tool_spans    = [(s, e, lbl) for s, e, lbl in all_h1_spans if lbl == "Tool"]
    usage_spans   = [(s, e, lbl) for s, e, lbl in all_h1_spans if lbl == "Usage"]

    # Extract predefined category spans from Head 2
    pred_spans = extract_spans(pred_h2_v, id_to_head2)

    def tok_to_char(tok_start, tok_end):
        cs = valid_offsets[tok_start][0]
        ce = valid_offsets[min(tok_end - 1, len(valid_offsets) - 1)][1]
        return cs, ce

    result = {"text": text, "tools": [], "usages": [], "categories": [], "relations": []}

    # Store Tool entity predictions
    for ts, te, _ in tool_spans:
        cs, ce = tok_to_char(ts, te)
        result["tools"].append({
            "text":      text[cs:ce],
            "char_span": [cs, ce],
        })

    # Store Usage entity predictions (NEW — was missing before)
    for us, ue, _ in usage_spans:
        cs, ce = tok_to_char(us, ue)
        result["usages"].append({
            "text":      text[cs:ce],
            "char_span": [cs, ce],
        })

    # Store predefined category predictions from Head 2
    for ps, pe, cat in pred_spans:
        cs, ce = tok_to_char(ps, pe)
        result["categories"].append({
            "text":      text[cs:ce],
            "category":  cat,
            "char_span": [cs, ce],
        })

    # Relations: Tool span → Predefined Category span via Head 3
    # Deduplicate to avoid inflated FP counts
    seen_relations = set()

    for tool in tool_spans:
        for pred in pred_spans:
            ti_s = valid[tool[0]]
            ti_e = valid[min(tool[1] - 1, len(valid) - 1)] + 1
            pi_s = valid[pred[0]]
            pi_e = valid[min(pred[1] - 1, len(valid) - 1)] + 1

            logit = model.classify_relation(seq_out, (ti_s, ti_e), (pi_s, pi_e))
            rel   = id_to_rel[logit.argmax(-1).item()]

            if rel != "No_relation":
                tc, te_c = tok_to_char(tool[0], tool[1])
                pc, pe_c = tok_to_char(pred[0], pred[1])
                dedup_key = (tc, te_c, pred[2])
                if dedup_key in seen_relations:
                    continue
                seen_relations.add(dedup_key)
                result["relations"].append({
                    "tool":           text[tc:te_c],
                    "tool_char_span": [tc, te_c],
                    "category":       pred[2],
                    "category_char_span": [pc, pe_c],
                    "relation":       rel,
                })

    return result

# ─────────────────────────────────────────────
# 7.  MATCHING HELPERS
# ─────────────────────────────────────────────

def spans_overlap(pred_span, gt_span):
    """Character-level overlap: at least one char in common."""
    return max(pred_span[0], gt_span[0]) < min(pred_span[1], gt_span[1])


def prf(tp, fp, fn):
    """Return (precision, recall, f1) rounded to 4 decimal places."""
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return round(p, 4), round(r, 4), round(f, 4)


def eval_entity_exact(pred_texts: set, gt_texts: set):
    """Exact match: compare normalised text strings."""
    tp = len(pred_texts & gt_texts)
    fp = len(pred_texts) - tp
    fn = len(gt_texts)   - tp
    return tp, fp, fn


def eval_entity_partial(pred_spans: list, gt_spans: list):
    """Partial match: character span overlap, greedy one-to-one matching."""
    gt_used = [False] * len(gt_spans)
    tp = 0
    for ps in pred_spans:
        for j, gs in enumerate(gt_spans):
            if not gt_used[j] and spans_overlap(ps, gs):
                gt_used[j] = True
                tp += 1
                break
    fp = len(pred_spans) - tp
    fn = len(gt_spans)   - tp
    return tp, fp, fn

# ─────────────────────────────────────────────
# 8.  COMPUTE METRICS
# ─────────────────────────────────────────────

def compute_metrics(predictions: list, ground_truth: list):

    # ══════════════════════════════════════════════════════════════════
    # TASK 1 — Binary classification (paper level)
    # Does the paper mention any AI tool? Yes (1) or No (0).
    # This is derived from tool entity detection — if the model finds
    # at least one tool entity the paper is classified as positive.
    # No exact/partial split needed — it is a binary decision.
    # ══════════════════════════════════════════════════════════════════
    bin_tp = bin_fp = bin_tn = bin_fn = 0

    for pred, gt in zip(predictions, ground_truth):
        gt_has_ai   = len(gt["tool_spans"]) > 0
        pred_has_ai = len(pred["tools"])    > 0

        if     gt_has_ai and     pred_has_ai: bin_tp += 1
        elif   gt_has_ai and not pred_has_ai: bin_fn += 1
        elif not gt_has_ai and   pred_has_ai: bin_fp += 1
        else:                                 bin_tn += 1

    bin_p, bin_r, bin_f1 = prf(bin_tp, bin_fp, bin_fn)
    bin_acc = (bin_tp + bin_tn) / len(predictions) if predictions else 0.0

    task1 = {
        "accuracy":  round(bin_acc, 4),
        "precision": bin_p,
        "recall":    bin_r,
        "f1":        bin_f1,
        "tp": bin_tp, "fp": bin_fp, "tn": bin_tn, "fn": bin_fn,
    }

    # ══════════════════════════════════════════════════════════════════
    # TASK 2a — Tool Entity extraction (micro F1, exact + partial)
    # Compare predicted tool text/spans against ground truth tool spans.
    # Exact:   normalised text strings must match (case-insensitive)
    # Partial: character spans must overlap by at least one character
    # ══════════════════════════════════════════════════════════════════
    ex_tool = {"tp": 0, "fp": 0, "fn": 0}
    pa_tool = {"tp": 0, "fp": 0, "fn": 0}
    per_tool_ex = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})

    for pred, gt in zip(predictions, ground_truth):
        text = gt["text"]

        gt_tool_texts = {text[s:e].lower().strip() for s, e in gt["tool_spans"]}
        gt_tool_spans = [[s, e] for s, e in gt["tool_spans"]]

        pred_tool_texts = {t["text"].lower().strip() for t in pred["tools"]}
        pred_tool_spans = [t["char_span"] for t in pred["tools"]]

        # Exact
        tp, fp, fn = eval_entity_exact(pred_tool_texts, gt_tool_texts)
        ex_tool["tp"] += tp; ex_tool["fp"] += fp; ex_tool["fn"] += fn

        matched = pred_tool_texts & gt_tool_texts
        for name in matched:                          per_tool_ex[name]["tp"] += 1
        for name in pred_tool_texts - gt_tool_texts:  per_tool_ex[name]["fp"] += 1
        for name in gt_tool_texts   - pred_tool_texts: per_tool_ex[name]["fn"] += 1

        # Partial
        tp, fp, fn = eval_entity_partial(pred_tool_spans, gt_tool_spans)
        pa_tool["tp"] += tp; pa_tool["fp"] += fp; pa_tool["fn"] += fn

    ex_tool_p, ex_tool_r, ex_tool_f = prf(ex_tool["tp"], ex_tool["fp"], ex_tool["fn"])
    pa_tool_p, pa_tool_r, pa_tool_f = prf(pa_tool["tp"], pa_tool["fp"], pa_tool["fn"])

    per_tool_metrics = {}
    for tool_name, c in sorted(per_tool_ex.items()):
        p, r, f = prf(c["tp"], c["fp"], c["fn"])
        per_tool_metrics[tool_name] = {
            "precision": p, "recall": r, "f1": f,
            "tp": c["tp"], "fp": c["fp"], "fn": c["fn"],
        }

    # ══════════════════════════════════════════════════════════════════
    # TASK 2b — Usage Entity extraction (micro F1, exact + partial)
    # Compare predicted usage text/spans against ground truth usage_spans.
    # These are the free-text usage descriptions from Head 1 of BERT.
    # Exact:   normalised text strings must match (case-insensitive)
    # Partial: character spans must overlap by at least one character
    # ══════════════════════════════════════════════════════════════════
    ex_usage = {"tp": 0, "fp": 0, "fn": 0}
    pa_usage = {"tp": 0, "fp": 0, "fn": 0}

    for pred, gt in zip(predictions, ground_truth):
        text = gt["text"]

        gt_usage_texts = {text[s:e].lower().strip() for s, e in gt["usage_spans"]}
        gt_usage_spans = [[s, e] for s, e in gt["usage_spans"]]

        pred_usage_texts = {u["text"].lower().strip() for u in pred["usages"]}
        pred_usage_spans = [u["char_span"] for u in pred["usages"]]

        # Exact
        tp, fp, fn = eval_entity_exact(pred_usage_texts, gt_usage_texts)
        ex_usage["tp"] += tp; ex_usage["fp"] += fp; ex_usage["fn"] += fn

        # Partial
        tp, fp, fn = eval_entity_partial(pred_usage_spans, gt_usage_spans)
        pa_usage["tp"] += tp; pa_usage["fp"] += fp; pa_usage["fn"] += fn

    ex_usage_p, ex_usage_r, ex_usage_f = prf(ex_usage["tp"], ex_usage["fp"], ex_usage["fn"])
    pa_usage_p, pa_usage_r, pa_usage_f = prf(pa_usage["tp"], pa_usage["fp"], pa_usage["fn"])

    task2 = {
        "tool_entity": {
            "exact":   {"precision": ex_tool_p, "recall": ex_tool_r, "f1": ex_tool_f,
                        "tp": ex_tool["tp"], "fp": ex_tool["fp"], "fn": ex_tool["fn"]},
            "partial": {"precision": pa_tool_p, "recall": pa_tool_r, "f1": pa_tool_f,
                        "tp": pa_tool["tp"], "fp": pa_tool["fp"], "fn": pa_tool["fn"]},
            "per_tool": per_tool_metrics,
        },
        "usage_entity": {
            "exact":   {"precision": ex_usage_p, "recall": ex_usage_r, "f1": ex_usage_f,
                        "tp": ex_usage["tp"], "fp": ex_usage["fp"], "fn": ex_usage["fn"]},
            "partial": {"precision": pa_usage_p, "recall": pa_usage_r, "f1": pa_usage_f,
                        "tp": pa_usage["tp"], "fp": pa_usage["fp"], "fn": pa_usage["fn"]},
        },
    }

    # ══════════════════════════════════════════════════════════════════
    # TASK 3 — Relation extraction: used_for (tool-paper pair level)
    # For each tool in a paper, did the model assign the correct
    # predefined contribution role(s)?
    #
    # Two-step evaluation:
    #   Step 1: Match predicted tool to ground truth tool
    #           Exact:   normalised tool text must match exactly
    #           Partial: tool character spans must overlap
    #   Step 2: Once tool is matched, compare predicted role(s) to GT roles
    #           using set-level intersection (micro F1 over role labels)
    #
    # IMPORTANT: Task 3 performance is bounded by Task 2 tool entity
    # performance. Better tool extraction → more correct role assignments.
    # The improvement from exact to partial match in Task 3 is entirely
    # driven by Task 2 recovering more entity pairs under relaxed matching.
    # ══════════════════════════════════════════════════════════════════
    ex_role = {"tp": 0, "fp": 0, "fn": 0}
    pa_role = {"tp": 0, "fp": 0, "fn": 0}
    ex_role_exact_match = 0
    total_gt_pairs      = 0
    per_role_ex = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    per_role_pa = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})

    for pred, gt in zip(predictions, ground_truth):
        text = gt["text"]

        # Build GT: tool_text → set of predefined category labels
        gt_tool_roles_ex = defaultdict(set)
        for r in gt["relations"]:
            ts, te   = r["tool_span"][0], r["tool_span"][1]
            tool_txt = text[ts:te].lower().strip()
            gt_tool_roles_ex[tool_txt].add(r["pred_cat"])

        # Build GT span-keyed for partial matching
        seen_spans = {}
        for r in gt["relations"]:
            ts, te = r["tool_span"][0], r["tool_span"][1]
            key    = (ts, te)
            if key not in seen_spans:
                seen_spans[key] = set()
            seen_spans[key].add(r["pred_cat"])
        gt_tool_roles_pa = [([ts, te], roles) for (ts, te), roles in seen_spans.items()]

        # Build Pred: tool_text → set of predicted category labels
        pred_tool_roles_ex = defaultdict(set)
        pred_tool_roles_pa = {}
        for r in pred["relations"]:
            tool_txt = r["tool"].lower().strip()
            pred_tool_roles_ex[tool_txt].add(r["category"])
            span_key = tuple(r.get("tool_char_span", [0, 0]))
            if span_key not in pred_tool_roles_pa:
                pred_tool_roles_pa[span_key] = set()
            pred_tool_roles_pa[span_key].add(r["category"])

        # ── Exact role matching ──────────────────────────────────────
        for tool_txt, gt_roles in gt_tool_roles_ex.items():
            total_gt_pairs += 1
            pred_roles = pred_tool_roles_ex.get(tool_txt, set())
            tp = len(pred_roles & gt_roles)
            fp = len(pred_roles - gt_roles)
            fn = len(gt_roles   - pred_roles)
            ex_role["tp"] += tp; ex_role["fp"] += fp; ex_role["fn"] += fn
            if pred_roles == gt_roles:
                ex_role_exact_match += 1
            for role in (pred_roles & gt_roles): per_role_ex[role]["tp"] += 1
            for role in (pred_roles - gt_roles): per_role_ex[role]["fp"] += 1
            for role in (gt_roles - pred_roles): per_role_ex[role]["fn"] += 1

        # FP: predicted tools not in GT
        for tool_txt, pred_roles in pred_tool_roles_ex.items():
            if tool_txt not in gt_tool_roles_ex:
                ex_role["fp"] += len(pred_roles)
                for role in pred_roles: per_role_ex[role]["fp"] += 1

        # ── Partial role matching ────────────────────────────────────
        gt_pa_used = [False] * len(gt_tool_roles_pa)

        for pred_span_key, pred_roles in pred_tool_roles_pa.items():
            pred_span = list(pred_span_key)
            pred_txt  = ""
            for r in pred["relations"]:
                if tuple(r.get("tool_char_span", [])) == pred_span_key:
                    pred_txt = r["tool"].lower().strip()
                    break

            best_j = -1
            for j, (g_span, g_roles) in enumerate(gt_tool_roles_pa):
                if gt_pa_used[j]:
                    continue
                span_ok = spans_overlap(pred_span, g_span) if pred_span != [0, 0] else False
                text_ok = (pred_txt != "" and
                           pred_txt == text[g_span[0]:g_span[1]].lower().strip())
                if span_ok or text_ok:
                    best_j = j
                    break

            if best_j >= 0:
                gt_pa_used[best_j] = True
                g_roles = gt_tool_roles_pa[best_j][1]
                tp = len(pred_roles & g_roles)
                fp = len(pred_roles - g_roles)
                fn = len(g_roles    - pred_roles)
                pa_role["tp"] += tp; pa_role["fp"] += fp; pa_role["fn"] += fn
                for role in (pred_roles & g_roles): per_role_pa[role]["tp"] += 1
                for role in (pred_roles - g_roles): per_role_pa[role]["fp"] += 1
                for role in (g_roles - pred_roles): per_role_pa[role]["fn"] += 1
            else:
                pa_role["fp"] += len(pred_roles)
                for role in pred_roles: per_role_pa[role]["fp"] += 1

        # Unmatched GT pairs → all FN
        for j, (g_span, g_roles) in enumerate(gt_tool_roles_pa):
            if not gt_pa_used[j]:
                pa_role["fn"] += len(g_roles)
                for role in g_roles: per_role_pa[role]["fn"] += 1

    ex_rp, ex_rr, ex_rf = prf(ex_role["tp"], ex_role["fp"], ex_role["fn"])
    pa_rp, pa_rr, pa_rf = prf(pa_role["tp"], pa_role["fp"], pa_role["fn"])
    exact_match_acc = (round(ex_role_exact_match / total_gt_pairs, 4)
                       if total_gt_pairs > 0 else 0.0)

    all_roles = sorted(set(list(per_role_ex.keys()) + list(per_role_pa.keys())))
    per_role_metrics = {}
    for role in all_roles:
        ec = per_role_ex[role]; pc = per_role_pa[role]
        ep, er, ef = prf(ec["tp"], ec["fp"], ec["fn"])
        pp, pr_r, pf = prf(pc["tp"], pc["fp"], pc["fn"])
        per_role_metrics[role] = {
            "exact":   {"precision": ep, "recall": er, "f1": ef,
                        "tp": ec["tp"], "fp": ec["fp"], "fn": ec["fn"]},
            "partial": {"precision": pp, "recall": pr_r, "f1": pf,
                        "tp": pc["tp"], "fp": pc["fp"], "fn": pc["fn"]},
        }

    task3 = {
        "exact": {
            "micro_precision": ex_rp, "micro_recall": ex_rr, "micro_f1": ex_rf,
            "exact_match_accuracy": exact_match_acc,
            "tp": ex_role["tp"], "fp": ex_role["fp"], "fn": ex_role["fn"],
        },
        "partial": {
            "micro_precision": pa_rp, "micro_recall": pa_rr, "micro_f1": pa_rf,
            "tp": pa_role["tp"], "fp": pa_role["fp"], "fn": pa_role["fn"],
        },
        "per_role": per_role_metrics,
    }

    return {
        "task1_binary":          task1,
        "task2_entity_extraction": task2,
        "task3_used_for":        task3,
    }

# ─────────────────────────────────────────────
# 9.  REPORT WRITING
# ─────────────────────────────────────────────

def write_report(predictions, ground_truth, metrics, report_path: str):
    lines = []
    SEP  = "=" * 78
    THIN = "-" * 78

    lines.append(SEP)
    lines.append("  BERT Joint NER + RE — Evaluation Report")
    lines.append(f"  Test samples : {len(predictions)}")
    lines.append(SEP)

    # ── Task 1 ────────────────────────────────────────────────────────
    t1 = metrics["task1_binary"]
    lines.append("\n── TASK 1: Binary Classification — AI Tool Mentioned? (paper level) ──")
    lines.append("   (No exact/partial split — binary yes/no decision per paper)")
    lines.append(f"  Accuracy  : {t1['accuracy']:.4f}")
    lines.append(f"  Precision : {t1['precision']:.4f}")
    lines.append(f"  Recall    : {t1['recall']:.4f}")
    lines.append(f"  F1        : {t1['f1']:.4f}")
    lines.append(f"  TP={t1['tp']}  FP={t1['fp']}  TN={t1['tn']}  FN={t1['fn']}")

    # ── Task 2 ────────────────────────────────────────────────────────
    t2 = metrics["task2_entity_extraction"]
    lines.append("\n── TASK 2: Multi-label Entity Extraction ──")
    lines.append(f"  {'Entity':<16} {'Setting':<10} {'P':>8} {'R':>8} {'F1':>8} "
                 f"{'TP':>6} {'FP':>6} {'FN':>6}")
    lines.append(f"  {THIN}")

    for entity_key, entity_label in [("tool_entity", "Tool Entity"),
                                      ("usage_entity", "Usage Entity")]:
        for s_key, s_label in [("exact", "Exact"), ("partial", "Partial")]:
            m = t2[entity_key][s_key]
            lines.append(
                f"  {entity_label:<16} {s_label:<10} "
                f"{m['precision']:>8.4f} {m['recall']:>8.4f} {m['f1']:>8.4f} "
                f"{m['tp']:>6} {m['fp']:>6} {m['fn']:>6}"
            )
        lines.append("")

    lines.append("  Per-Tool Breakdown (Tool Entity, Exact Match):")
    lines.append(f"  {'Tool':<30} {'P':>7} {'R':>7} {'F1':>7} {'TP':>5} {'FP':>5} {'FN':>5}")
    lines.append(f"  {'-'*65}")
    for tool_name, m in t2["tool_entity"]["per_tool"].items():
        lines.append(
            f"  {tool_name:<30} {m['precision']:>7.4f} {m['recall']:>7.4f} "
            f"{m['f1']:>7.4f} {m['tp']:>5} {m['fp']:>5} {m['fn']:>5}"
        )

    # ── Task 3 ────────────────────────────────────────────────────────
    t3 = metrics["task3_used_for"]
    lines.append("\n── TASK 3: used_for Relation Extraction (tool-paper pair level) ──")
    lines.append("   (Performance is bounded by Task 2 — better entity extraction")
    lines.append("    leads to higher relation scores under both match settings)")
    lines.append(f"  {'Setting':<12} {'Micro-P':>9} {'Micro-R':>9} {'Micro-F1':>10}  Notes")
    lines.append(f"  {'-'*70}")
    ex = t3["exact"]
    pa = t3["partial"]
    lines.append(
        f"  {'Exact':<12} {ex['micro_precision']:>9.4f} {ex['micro_recall']:>9.4f} "
        f"{ex['micro_f1']:>10.4f}  exact-match-acc={ex['exact_match_accuracy']:.4f}"
    )
    lines.append(
        f"  {'Partial':<12} {pa['micro_precision']:>9.4f} {pa['micro_recall']:>9.4f} "
        f"{pa['micro_f1']:>10.4f}"
    )

    lines.append("\n  Per-Role Breakdown:")
    lines.append(
        f"  {'Role':<45} {'Setting':<10} {'P':>7} {'R':>7} {'F1':>7} "
        f"{'TP':>5} {'FP':>5} {'FN':>5}"
    )
    lines.append(f"  {'-'*98}")
    for role, m in sorted(t3["per_role"].items()):
        for s_key, s_label in [("exact", "Exact"), ("partial", "Partial")]:
            sm = m[s_key]
            lines.append(
                f"  {role:<45} {s_label:<10} "
                f"{sm['precision']:>7.4f} {sm['recall']:>7.4f} {sm['f1']:>7.4f} "
                f"{sm['tp']:>5} {sm['fp']:>5} {sm['fn']:>5}"
            )
        lines.append("")

    # ── Summary table (for easy copy into paper) ──────────────────────
    lines.append(f"\n{SEP}")
    lines.append("  SUMMARY TABLE  (matches paper table structure)")
    lines.append(f"  {'Setting':<10} {'Task':<35} {'P':>8} {'R':>8} {'F1':>8}")
    lines.append(f"  {THIN}")

    t1 = metrics["task1_binary"]
    lines.append(f"  {'—':<10} {'Task 1: AI Tool Detection':<35} "
                 f"{t1['precision']:>8.4f} {t1['recall']:>8.4f} {t1['f1']:>8.4f}")
    lines.append("")

    for s_key, s_label in [("exact", "Exact"), ("partial", "Partial")]:
        lines.append(f"  {s_label:<10} {'Task 2: Tool Entity':<35} "
                     f"{t2['tool_entity'][s_key]['precision']:>8.4f} "
                     f"{t2['tool_entity'][s_key]['recall']:>8.4f} "
                     f"{t2['tool_entity'][s_key]['f1']:>8.4f}")
        lines.append(f"  {'':<10} {'Task 2: Usage Entity':<35} "
                     f"{t2['usage_entity'][s_key]['precision']:>8.4f} "
                     f"{t2['usage_entity'][s_key]['recall']:>8.4f} "
                     f"{t2['usage_entity'][s_key]['f1']:>8.4f}")
        lines.append(f"  {'':<10} {'Task 3: used_for':<35} "
                     f"{t3[s_key]['micro_precision']:>8.4f} "
                     f"{t3[s_key]['micro_recall']:>8.4f} "
                     f"{t3[s_key]['micro_f1']:>8.4f}")
        lines.append("")

    # ── Per-sample detail ─────────────────────────────────────────────
    lines.append(f"\n{SEP}")
    lines.append("  Per-Sample Predictions vs Ground Truth")
    lines.append(SEP)

    for i, (pred, gt) in enumerate(zip(predictions, ground_truth)):
        text     = gt["text"]
        gt_tools  = [text[s:e] for s, e in gt["tool_spans"]]
        gt_usages = [text[s:e] for s, e in gt["usage_spans"]]
        gt_rels   = [(text[r["tool_span"][0]:r["tool_span"][1]], r["pred_cat"])
                     for r in gt["relations"]]

        lines.append(f"\n[Sample {i+1}]")
        lines.append(f"  Text       : {text[:120]}{'...' if len(text) > 120 else ''}")
        lines.append(f"  GT  Tools  : {gt_tools}")
        lines.append(f"  PRD Tools  : {[t['text'] for t in pred['tools']]}")
        lines.append(f"  GT  Usages : {gt_usages}")
        lines.append(f"  PRD Usages : {[u['text'] for u in pred['usages']]}")
        lines.append(f"  GT  Rels   : {gt_rels}")
        lines.append(f"  PRD Rels   : {[(r['tool'], r['category']) for r in pred['relations']]}")

    report_text = "\n".join(lines)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    return report_text

# ─────────────────────────────────────────────
# 10. MAIN
# ─────────────────────────────────────────────

def main():
    # ── Validate paths ────────────────────────────────────────────────
    missing = []
    for path, label, kind in [
        (MODEL_WEIGHTS,     "best_model.pt",     "file"),
        (LABEL_MAP_PATH,    "label_map.json",    "file"),
        (MODEL_CONFIG_PATH, "model_config.json", "file"),
        (TEST_DATA_PATH,    "splits/test.jsonl", "file"),
        (TOKENIZER_PATH,    "tokenizer/",        "dir"),
    ]:
        ok = os.path.isdir(path) if kind == "dir" else os.path.isfile(path)
        if not ok:
            missing.append((label, path))

    if missing:
        print("\nERROR: Required files/folders not found:")
        for label, path in missing:
            print(f"  x  {label}  →  {path}")
        sys.exit(1)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ── Load label maps ───────────────────────────────────────────────
    print("Loading label maps ...")
    (
        head1_labels, head1_to_id, id_to_head1,
        head2_labels, head2_to_id, id_to_head2,
        rel_labels,   rel_to_id,   id_to_rel,
    ) = load_label_maps(LABEL_MAP_PATH)
    print(f"  Head1 labels : {head1_labels}")
    print(f"  Head2 labels : {len(head2_labels)}")
    print(f"  Rel labels   : {rel_labels}")

    # ── Load model config ─────────────────────────────────────────────
    with open(MODEL_CONFIG_PATH, "r") as f:
        model_config = json.load(f)
    model_name = model_config["model_name"]
    max_len    = model_config["max_len"]
    print(f"\nModel : {model_name}  |  Max len: {max_len}  |  Device: {DEVICE}")

    # ── Load tokenizer & model ────────────────────────────────────────
    print("\nLoading tokenizer and model weights ...")
    tokenizer = BertTokenizerFast.from_pretrained(TOKENIZER_PATH)
    model = JointNER_RE(
        model_name=model_name,
        n_head1=len(head1_labels),
        n_head2=len(head2_labels),
        n_rel=len(rel_labels),
    ).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=DEVICE))
    model.eval()
    print("  Model loaded successfully.")

    # ── Load test data ────────────────────────────────────────────────
    print(f"\nLoading test data from:\n  {TEST_DATA_PATH}")
    test_samples = load_test_jsonl(TEST_DATA_PATH)
    print(f"  {len(test_samples)} test samples found.\n")

    # ── Run predictions ───────────────────────────────────────────────
    print("Running inference ...")
    predictions = []
    for i, sample in enumerate(test_samples):
        pred = predict(
            text=sample["text"],
            model=model,
            tokenizer=tokenizer,
            id_to_head1=id_to_head1,
            id_to_head2=id_to_head2,
            id_to_rel=id_to_rel,
            max_len=max_len,
        )
        predictions.append(pred)
        if (i + 1) % 10 == 0 or (i + 1) == len(test_samples):
            print(f"  {i+1}/{len(test_samples)} done")

    # ── Evaluate ──────────────────────────────────────────────────────
    print("\nEvaluating ...")
    metrics = compute_metrics(predictions, test_samples)

    # ── Print summary ─────────────────────────────────────────────────
    SEP = "=" * 78
    print(f"\n{SEP}")

    t1 = metrics["task1_binary"]
    print(f"\n  TASK 1 — Binary AI Tool Detection (paper level, no exact/partial split)")
    print(f"  Accuracy={t1['accuracy']:.4f}  P={t1['precision']:.4f}  "
          f"R={t1['recall']:.4f}  F1={t1['f1']:.4f}")
    print(f"  TP={t1['tp']}  FP={t1['fp']}  TN={t1['tn']}  FN={t1['fn']}")

    t2 = metrics["task2_entity_extraction"]
    print(f"\n  TASK 2 — Multi-label Entity Extraction")
    print(f"  {'Entity':<16} {'Setting':<10} {'P':>8} {'R':>8} {'F1':>8}")
    print(f"  {'-'*55}")
    for entity_key, entity_label in [("tool_entity",  "Tool Entity"),
                                      ("usage_entity", "Usage Entity")]:
        for s_key, s_label in [("exact", "Exact"), ("partial", "Partial")]:
            m = t2[entity_key][s_key]
            print(f"  {entity_label:<16} {s_label:<10} "
                  f"{m['precision']:>8.4f} {m['recall']:>8.4f} {m['f1']:>8.4f}")

    t3 = metrics["task3_used_for"]
    print(f"\n  TASK 3 — used_for Relation Extraction")
    print(f"  {'Setting':<12} {'Micro-P':>9} {'Micro-R':>9} {'Micro-F1':>10}  Notes")
    print(f"  {'-'*70}")
    ex = t3["exact"];  pa = t3["partial"]
    print(f"  {'Exact':<12} {ex['micro_precision']:>9.4f} {ex['micro_recall']:>9.4f} "
          f"{ex['micro_f1']:>10.4f}  exact-match-acc={ex['exact_match_accuracy']:.4f}")
    print(f"  {'Partial':<12} {pa['micro_precision']:>9.4f} {pa['micro_recall']:>9.4f} "
          f"{pa['micro_f1']:>10.4f}")
    print(SEP)

    # ── Save JSON results ─────────────────────────────────────────────
    output_json = {
        "metrics": metrics,
        "predictions": [
            {
                "sample_id": i + 1,
                "text":      pred["text"],
                "predicted": {
                    "tools":      pred["tools"],
                    "usages":     pred["usages"],
                    "categories": pred["categories"],
                    "relations":  pred["relations"],
                },
                "ground_truth": {
                    "tools":  [sample["text"][s:e] for s, e in sample["tool_spans"]],
                    "usages": [sample["text"][s:e] for s, e in sample["usage_spans"]],
                    "categories": [
                        {"category": sp[2], "text": sample["text"][sp[0]:sp[1]]}
                        for sp in sample["pred_spans"]
                    ],
                    "relations": [
                        {
                            "tool":     sample["text"][r["tool_span"][0]:r["tool_span"][1]],
                            "category": r["pred_cat"],
                        }
                        for r in sample["relations"]
                    ],
                },
            }
            for i, (pred, sample) in enumerate(zip(predictions, test_samples))
        ],
    }
    with open(RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(output_json, f, indent=2, ensure_ascii=False)
    print(f"\n  Full results (JSON) → {RESULTS_JSON}")

    write_report(predictions, test_samples, metrics, RESULTS_TXT)
    print(f"  Human report (TXT)  → {RESULTS_TXT}")
    print("\nDone!")


if __name__ == "__main__":
    main()