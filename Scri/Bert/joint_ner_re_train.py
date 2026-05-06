"""
Joint NER + Relation Extraction with BERT
==========================================
Architecture:
  - Head 1: BIO tagger  → [O, B-Tool, I-Tool, B-Usage, I-Usage]
  - Head 2: BIO tagger  → [O, B-<PreDefinedCategory>, I-<PreDefinedCategory>, ...]
  - Head 3: Span-pair classifier → [Used_for, No_relation]

Data: JSONL format from Annotation_aftab.jsonl
Split: 70% train / 10% val / 20% test (stratified)

Key design:
  - LABEL_MAP is built automatically from the actual data at runtime
  - No hardcoded assumptions about label text variations
"""

# ─────────────────────────────────────────────
# 0.  IMPORTS
# ─────────────────────────────────────────────
import json, os, re, random, warnings
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

from transformers import (
    BertTokenizerFast,
    BertModel,
    get_linear_schedule_with_warmup,
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
from seqeval.metrics import classification_report as seq_report
from seqeval.metrics import f1_score as seq_f1

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# 1.  CONFIGURATION
# ─────────────────────────────────────────────
DATA_PATH   = r"path/to/Annotation.jsonl"   # set this to your annotated JSONL file
OUTPUT_DIR  = r"path/to/bert_output"        # directory for splits/, model/, results/
MODEL_NAME  = "bert-base-uncased"   # or "allenai/scibert_scivocab_uncased"

MAX_LEN     = 256
BATCH_SIZE  = 8
EPOCHS      = 30
LR          = 2e-5
LAMBDA_NER1 = 1.0
LAMBDA_NER2 = 1.0
LAMBDA_RE   = 1.5
SEED        = 42
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─────────────────────────────────────────────
# 2.  BUILD LABEL MAP DIRECTLY FROM DATA
# ─────────────────────────────────────────────

def to_canonical(raw: str) -> str:
    """
    Convert any raw label string into a BIO-safe canonical name.
    e.g. 'Grammar and spelling check' → 'Grammar_and_spelling_check'
         'Transcription (speech-to-text conversion)' → 'Transcription_speech_to_text_conversion'
    """
    s = raw.strip()
    s = re.sub(r"[^a-zA-Z0-9]+", "_", s)   # replace non-alphanumeric runs with _
    s = s.strip("_")
    return s


def build_label_map_from_data(data_path: str):
    """
    Read the JSONL file once and collect every unique pre-defined label.
    Returns:
      canonical_categories : sorted list of BIO-safe category names
      raw_to_canonical     : dict {raw_label_string → canonical_name}

    Nothing is hardcoded — all variation in the data is handled automatically.
    """
    raw_labels = set()
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            for ent in rec.get("entities", []):
                lbl = ent["label"].strip()
                if lbl not in ("Tool", "Usage"):
                    raw_labels.add(lbl)

    raw_to_canonical     = {raw: to_canonical(raw) for raw in raw_labels}
    canonical_categories = sorted(set(raw_to_canonical.values()))

    print("\n=== LABEL MAP BUILT FROM DATA ===")
    for raw, canon in sorted(raw_to_canonical.items()):
        marker = "  ✓" if raw == canon else " →"
        print(f"  '{raw}'{marker}  '{canon}'")
    print(f"\nTotal pre-defined categories: {len(canonical_categories)}")

    return canonical_categories, raw_to_canonical

# ─────────────────────────────────────────────
# 3.  LABEL DEFINITIONS  (populated in main)
# ─────────────────────────────────────────────

# Head 1 — always the same
HEAD1_LABELS = ["O", "B-Tool", "I-Tool", "B-Usage", "I-Usage"]
HEAD1_TO_ID  = {l: i for i, l in enumerate(HEAD1_LABELS)}
ID_TO_HEAD1  = {i: l for l, i in HEAD1_TO_ID.items()}

# Head 2 & Relation — set after reading data
HEAD2_LABELS = None
HEAD2_TO_ID  = None
ID_TO_HEAD2  = None

REL_LABELS   = ["No_relation", "Used_for"]
REL_TO_ID    = {l: i for i, l in enumerate(REL_LABELS)}
ID_TO_REL    = {i: l for l, i in REL_TO_ID.items()}

CANONICAL_CATEGORIES = None
RAW_TO_CANONICAL     = None


def init_head2_labels(canonical_categories):
    labels = (
        ["O"]
        + [f"B-{c}" for c in canonical_categories]
        + [f"I-{c}" for c in canonical_categories]
    )
    to_id = {l: i for i, l in enumerate(labels)}
    id_to = {i: l for l, i in to_id.items()}
    return labels, to_id, id_to


def normalize_label(raw: str) -> str:
    return RAW_TO_CANONICAL.get(raw.strip(), "Unknown")

# ─────────────────────────────────────────────
# 4.  DATA LOADING & PARSING
# ─────────────────────────────────────────────

def load_jsonl(path: str):
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def parse_sample(sample: dict):
    """
    Parse one JSONL record.
    Resolves: Tool --[Used_for]--> Usage --[Maps_to]--> PreDefined
    into direct: Tool → PreDefined  (the meaningful relation)
    """
    text      = sample["text"]
    entities  = sample.get("entities", [])
    relations = sample.get("relations", [])
    ent_by_id = {e["id"]: e for e in entities}

    tool_spans, usage_spans, pred_spans = [], [], []

    for e in entities:
        lbl = e["label"].strip()
        if lbl == "Tool":
            tool_spans.append((e["start_offset"], e["end_offset"]))
        elif lbl == "Usage":
            usage_spans.append((e["start_offset"], e["end_offset"]))
        else:
            cat = normalize_label(lbl)
            if cat != "Unknown":
                pred_spans.append((e["start_offset"], e["end_offset"], cat, e["id"]))

    # Resolve Tool → Usage → PreDefined chain
    rel_list = []
    for r in relations:
        if r["type"] != "Used_for":
            continue
        from_ent = ent_by_id.get(r["from_id"])
        to_ent   = ent_by_id.get(r["to_id"])
        if not (from_ent and to_ent):
            continue
        if from_ent["label"].strip() != "Tool":
            continue
        usage_id = r["to_id"]
        for r2 in relations:
            if r2["type"] == "Maps_to" and r2["from_id"] == usage_id:
                pred_ent = ent_by_id.get(r2["to_id"])
                if pred_ent:
                    cat = normalize_label(pred_ent["label"].strip())
                    if cat != "Unknown":
                        rel_list.append({
                            "tool_span": (from_ent["start_offset"], from_ent["end_offset"]),
                            "pred_span": (pred_ent["start_offset"],  pred_ent["end_offset"]),
                            "pred_cat":  cat,
                        })

    return {
        "text":        text,
        "tool_spans":  tool_spans,
        "usage_spans": usage_spans,
        "pred_spans":  pred_spans,
        "relations":   rel_list,
    }

# ─────────────────────────────────────────────
# 5.  TOKENIZATION & LABEL ALIGNMENT
# ─────────────────────────────────────────────

def align_labels_to_tokens(text, spans_h1, spans_h2, tokenizer):
    encoding = tokenizer(
        text,
        max_length=MAX_LEN,
        truncation=True,
        return_offsets_mapping=True,
        padding="max_length",
    )
    offsets = encoding["offset_mapping"]
    L       = len(offsets)
    h1_ids  = [HEAD1_TO_ID["O"]] * L
    h2_ids  = [HEAD2_TO_ID["O"]] * L

    def find_label(tok_s, tok_e, spans):
        for span in spans:
            s, e, lbl = span[0], span[1], span[2]
            if tok_s >= s and tok_e <= e:
                return lbl, (tok_s == s)
        return None, False

    for i, (tok_s, tok_e) in enumerate(offsets):
        if tok_s == tok_e:          # special / padding token → ignore in loss
            h1_ids[i] = -100
            h2_ids[i] = -100
            continue

        lbl, is_begin = find_label(tok_s, tok_e, spans_h1)
        if lbl:
            tag = f"{'B' if is_begin else 'I'}-{lbl}"
            h1_ids[i] = HEAD1_TO_ID.get(tag, HEAD1_TO_ID["O"])

        lbl2, is_begin2 = find_label(tok_s, tok_e, spans_h2)
        if lbl2:
            tag2 = f"{'B' if is_begin2 else 'I'}-{lbl2}"
            h2_ids[i] = HEAD2_TO_ID.get(tag2, HEAD2_TO_ID["O"])

    return encoding, h1_ids, h2_ids


def encode_relations(encoding, relations):
    offsets  = encoding["offset_mapping"]
    encoded  = []
    for rel in relations:
        ts, te = rel["tool_span"]
        ps, pe = rel["pred_span"]
        tool_toks = [i for i, (s, e) in enumerate(offsets) if s >= ts and e <= te and s != e]
        pred_toks = [i for i, (s, e) in enumerate(offsets) if s >= ps and e <= pe and s != e]
        if tool_toks and pred_toks:
            encoded.append({
                "tool_tok_range": (min(tool_toks), max(tool_toks) + 1),
                "pred_tok_range": (min(pred_toks), max(pred_toks) + 1),
                "label_id":       REL_TO_ID["Used_for"],
            })
    return encoded

# ─────────────────────────────────────────────
# 6.  DATASET
# ─────────────────────────────────────────────

class AIDeclarationDataset(Dataset):
    def __init__(self, parsed_samples, tokenizer):
        self.items = []
        for ps in parsed_samples:
            text     = ps["text"]
            spans_h1 = (
                [(s, e, "Tool")  for s, e in ps["tool_spans"]] +
                [(s, e, "Usage") for s, e in ps["usage_spans"]]
            )
            spans_h2 = [(s, e, cat) for s, e, cat, _ in ps["pred_spans"]]

            encoding, h1_ids, h2_ids = align_labels_to_tokens(
                text, spans_h1, spans_h2, tokenizer
            )
            rels = encode_relations(encoding, ps["relations"])

            self.items.append({
                "input_ids":      torch.tensor(encoding["input_ids"],      dtype=torch.long),
                "attention_mask": torch.tensor(encoding["attention_mask"], dtype=torch.long),
                "token_type_ids": torch.tensor(encoding["token_type_ids"], dtype=torch.long),
                "h1_labels":      torch.tensor(h1_ids,                    dtype=torch.long),
                "h2_labels":      torch.tensor(h2_ids,                    dtype=torch.long),
                "relations":      rels,
                "text":           text,
            })

    def __len__(self):           return len(self.items)
    def __getitem__(self, idx):  return self.items[idx]


def collate_fn(batch):
    keys = ["input_ids", "attention_mask", "token_type_ids", "h1_labels", "h2_labels"]
    out  = {k: torch.stack([b[k] for b in batch]) for k in keys}
    out["relations"] = [b["relations"] for b in batch]
    out["texts"]     = [b["text"]      for b in batch]
    return out

# ─────────────────────────────────────────────
# 7.  MODEL
# ─────────────────────────────────────────────

class JointNER_RE(nn.Module):
    def __init__(self, model_name, n_head1, n_head2, n_rel, dropout=0.1):
        super().__init__()
        self.bert  = BertModel.from_pretrained(model_name)
        hidden     = self.bert.config.hidden_size   # 768

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
# 8.  LOSS
# ─────────────────────────────────────────────

ce_loss = nn.CrossEntropyLoss(ignore_index=-100)

def compute_loss(model, batch, device):
    input_ids       = batch["input_ids"].to(device)
    attention_mask  = batch["attention_mask"].to(device)
    token_type_ids  = batch["token_type_ids"].to(device)
    h1_labels       = batch["h1_labels"].to(device)
    h2_labels       = batch["h2_labels"].to(device)
    relations_batch = batch["relations"]

    seq_out, logits_h1, logits_h2 = model(input_ids, attention_mask, token_type_ids)

    B, L, _ = logits_h1.shape
    loss_h1  = ce_loss(logits_h1.view(B * L, -1), h1_labels.view(-1))
    loss_h2  = ce_loss(logits_h2.view(B * L, -1), h2_labels.view(-1))

    loss_re  = torch.tensor(0.0, device=device)
    re_count = 0
    for b_idx, rels in enumerate(relations_batch):
        s_out = seq_out[b_idx]
        for rel in rels:
            logit = model.classify_relation(s_out, rel["tool_tok_range"], rel["pred_tok_range"])
            lbl   = torch.tensor([rel["label_id"]], dtype=torch.long, device=device)
            loss_re  += ce_loss(logit, lbl)
            re_count += 1

    if re_count > 0:
        loss_re = loss_re / re_count

    total = LAMBDA_NER1 * loss_h1 + LAMBDA_NER2 * loss_h2 + LAMBDA_RE * loss_re
    return total, loss_h1.item(), loss_h2.item(), loss_re.item()

# ─────────────────────────────────────────────
# 9.  TRAINING
# ─────────────────────────────────────────────

def train_epoch(model, loader, optimizer, scheduler, device):
    model.train()
    totals = [0.0, 0.0, 0.0, 0.0]
    for batch in loader:
        optimizer.zero_grad()
        loss, l1, l2, lr_ = compute_loss(model, batch, device)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        for i, v in enumerate([loss.item(), l1, l2, lr_]):
            totals[i] += v
    n = len(loader)
    return [t / n for t in totals]

# ─────────────────────────────────────────────
# 10. EVALUATION
# ─────────────────────────────────────────────

def evaluate(model, loader, device, split_name="Val"):
    model.eval()
    all_h1_true, all_h1_pred = [], []
    all_h2_true, all_h2_pred = [], []
    all_re_true,  all_re_pred = [], []

    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            h1_labels      = batch["h1_labels"]
            h2_labels      = batch["h2_labels"]
            relations_batch= batch["relations"]

            seq_out, logits_h1, logits_h2 = model(input_ids, attention_mask, token_type_ids)
            pred_h1 = logits_h1.argmax(-1).cpu()
            pred_h2 = logits_h2.argmax(-1).cpu()

            for b in range(input_ids.size(0)):
                t1, p1 = [], []
                for t in range(h1_labels.size(1)):
                    if h1_labels[b, t].item() == -100:
                        continue
                    t1.append(ID_TO_HEAD1[h1_labels[b, t].item()])
                    p1.append(ID_TO_HEAD1[pred_h1[b, t].item()])
                all_h1_true.append(t1); all_h1_pred.append(p1)

                t2, p2 = [], []
                for t in range(h2_labels.size(1)):
                    if h2_labels[b, t].item() == -100:
                        continue
                    t2.append(ID_TO_HEAD2[h2_labels[b, t].item()])
                    p2.append(ID_TO_HEAD2[pred_h2[b, t].item()])
                all_h2_true.append(t2); all_h2_pred.append(p2)

                s_out = seq_out[b]
                for rel in relations_batch[b]:
                    logit = model.classify_relation(
                        s_out, rel["tool_tok_range"], rel["pred_tok_range"]
                    )
                    all_re_true.append(rel["label_id"])
                    all_re_pred.append(logit.argmax(-1).item())

    print(f"\n{'='*60}\n  {split_name} Results\n{'='*60}")

    print("\n── Head 1 NER (Tool / Usage) ──")
    print(seq_report(all_h1_true, all_h1_pred, zero_division=0))
    h1_f1 = seq_f1(all_h1_true, all_h1_pred, zero_division=0)

    print("\n── Head 2 NER (Pre-defined Categories) ──")
    print(seq_report(all_h2_true, all_h2_pred, zero_division=0))
    h2_f1 = seq_f1(all_h2_true, all_h2_pred, zero_division=0)

    re_f1 = 0.0
    if all_re_true:
        print("\n── Head 3 Relation (Used_for) ──")
        # Pass labels= explicitly so the report always shows both classes
        # even if only one appears in this split (common in early epochs / small val set)
        unique_true = sorted(set(all_re_true))
        unique_pred = sorted(set(all_re_pred))
        present_ids = sorted(set(unique_true) | set(unique_pred))
        present_names = [REL_LABELS[i] for i in present_ids]
        print(classification_report(
            all_re_true, all_re_pred,
            labels=present_ids,
            target_names=present_names,
            zero_division=0,
        ))
        re_f1 = f1_score(all_re_true, all_re_pred, average="macro", zero_division=0)

    print(f"\n  Summary | NER-H1 F1: {h1_f1:.4f} | NER-H2 F1: {h2_f1:.4f} | RE F1: {re_f1:.4f}")
    return h1_f1, h2_f1, re_f1

# ─────────────────────────────────────────────
# 11. MAIN
# ─────────────────────────────────────────────

def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def get_stratify_key(ps):
    # NOTE: replaced by safe_stratify_key() inside main() which handles rare classes
    cats = [cat for _, _, cat, _ in ps["pred_spans"]]
    return Counter(cats).most_common(1)[0][0] if cats else "No_AI"


def main():
    global HEAD2_LABELS, HEAD2_TO_ID, ID_TO_HEAD2
    global CANONICAL_CATEGORIES, RAW_TO_CANONICAL

    set_seed(SEED)

    # ── Output folder structure ───────────────────────────────────────
    #   BERT/
    #   ├── splits/          raw JSONL for train / val / test
    #   ├── model/           best_model.pt + tokenizer + label_map + config
    #   └── results/         training_history.json + per-epoch eval logs
    SPLITS_DIR  = os.path.join(OUTPUT_DIR, "splits")
    MODEL_DIR   = os.path.join(OUTPUT_DIR, "model")
    RESULTS_DIR = os.path.join(OUTPUT_DIR, "results")
    for d in [OUTPUT_DIR, SPLITS_DIR, MODEL_DIR, RESULTS_DIR]:
        os.makedirs(d, exist_ok=True)

    print(f"Device : {DEVICE}")
    print(f"Data   : {DATA_PATH}")
    print(f"Output : {OUTPUT_DIR}")

    # ── Step 1: Build label map from ACTUAL data ──────────────────────
    CANONICAL_CATEGORIES, RAW_TO_CANONICAL = build_label_map_from_data(DATA_PATH)
    HEAD2_LABELS, HEAD2_TO_ID, ID_TO_HEAD2 = init_head2_labels(CANONICAL_CATEGORIES)

    # Save so inference.py can reload without re-reading training data
    with open(os.path.join(MODEL_DIR, "label_map.json"), "w", encoding="utf-8") as f:
        json.dump({
            "raw_to_canonical":     RAW_TO_CANONICAL,
            "canonical_categories": CANONICAL_CATEGORIES,
            "head1_labels":         HEAD1_LABELS,
            "head2_labels":         HEAD2_LABELS,
            "rel_labels":           REL_LABELS,
        }, f, indent=2, ensure_ascii=False)
    print(f"Label map saved → {MODEL_DIR}/label_map.json")

    # ── Step 2: Load & parse ──────────────────────────────────────────
    raw    = load_jsonl(DATA_PATH)
    parsed = [parse_sample(s) for s in raw]
    print(f"\nTotal samples: {len(parsed)}")

    # ── Step 3: Distribution report ───────────────────────────────────
    cat_counts = Counter()
    for ps in parsed:
        for _, _, cat, _ in ps["pred_spans"]:
            cat_counts[cat] += 1

    print("\n=== CATEGORY DISTRIBUTION ===")
    for cat, cnt in cat_counts.most_common():
        warn = "  ⚠️  <10 samples" if cnt < 10 else ""
        print(f"  {cnt:4d}  {cat}{warn}")

    # ── Step 4: Build safe stratify key ──────────────────────────────
    # We do TWO splits: (all→trainval+test) then (trainval→train+val).
    # For a class to survive both splits it must appear in enough samples.
    #
    # Split 1: 300 → 240 trainval + 60 test   (80/20)
    # Split 2: 240 → 210 train   + 30 val     (87.5/12.5)
    #
    # sklearn needs ≥ n_splits samples per class.  With 2 splits the
    # safe minimum is ceil(300 / 60) = 5, but to be conservative we
    # use 10 (anything below 10 is too small to trust anyway).
    MIN_STRAT_COUNT = 10
    rare_cats = {cat for cat, cnt in cat_counts.items() if cnt < MIN_STRAT_COUNT}
    if rare_cats:
        print(f"\n  ℹ️  Categories with <{MIN_STRAT_COUNT} samples collapsed to "
              f"'Other_rare' for stratification only: {sorted(rare_cats)}")

    def safe_stratify_key(ps):
        """
        Dominant category of a sample, collapsed to 'Other_rare' when
        the category is too rare to survive stratified splitting.
        'No_AI' for samples with no AI tool at all.
        """
        cats = [cat for _, _, cat, _ in ps["pred_spans"]]
        if not cats:
            return "No_AI"
        dominant = Counter(cats).most_common(1)[0][0]
        return "Other_rare" if dominant in rare_cats else dominant

    strat_keys = [safe_stratify_key(p) for p in parsed]

    # Final safety check — if still broken just use random split
    key_counts = Counter(strat_keys)
    still_rare = {k for k, v in key_counts.items() if v < 2}
    if still_rare:
        print(f"  ⚠️  Still-rare after grouping: {still_rare}. "
              f"Falling back to random split.")
        strat_keys = None

    # ── Step 5: Stratified 70/10/20 split ────────────────────────────
    trainval, test_data = train_test_split(
        parsed, test_size=0.20, random_state=SEED,
        stratify=strat_keys,
    )

    # Recompute keys for the trainval portion only
    tv_strat = None
    if strat_keys is not None:
        tv_keys_raw   = [safe_stratify_key(p) for p in trainval]
        tv_key_counts = Counter(tv_keys_raw)
        tv_still_rare = {k for k, v in tv_key_counts.items() if v < 2}
        if tv_still_rare:
            print(f"  ⚠️  Trainval still-rare keys: {tv_still_rare}. "
                  f"Using random split for train/val.")
            tv_strat = None
        else:
            tv_strat = tv_keys_raw

    train_data, val_data = train_test_split(
        trainval, test_size=0.125, random_state=SEED,
        stratify=tv_strat,
    )
    print(f"\nSplit → Train: {len(train_data)} | Val: {len(val_data)} | Test: {len(test_data)}")

    # ── Save splits for reproducibility & reuse by other models ──────
    # Each file is valid JSONL — one parsed sample per line.
    # Other models (e.g. RoBERTa, SpanBERT) can load the same splits
    # directly to ensure fair comparison.
    def save_split(data, filename):
        path = os.path.join(SPLITS_DIR, filename)
        with open(path, "w", encoding="utf-8") as f:
            for sample in data:
                # Convert sets/tuples to lists so json can serialise them
                record = {
                    "text":        sample["text"],
                    "tool_spans":  [list(s) for s in sample["tool_spans"]],
                    "usage_spans": [list(s) for s in sample["usage_spans"]],
                    "pred_spans":  [list(s[:3]) for s in sample["pred_spans"]],  # drop entity id
                    "relations":   [
                        {
                            "tool_span": list(r["tool_span"]),
                            "pred_span": list(r["pred_span"]),
                            "pred_cat":  r["pred_cat"],
                        }
                        for r in sample["relations"]
                    ],
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"  Saved {len(data):>3d} samples → {path}")

    print("\nSaving splits...")
    save_split(train_data, "train.jsonl")
    save_split(val_data,   "val.jsonl")
    save_split(test_data,  "test.jsonl")

    # Also save a split summary so it's easy to inspect later
    split_summary = {
        "total":      len(parsed),
        "train":      len(train_data),
        "val":        len(val_data),
        "test":       len(test_data),
        "seed":       SEED,
        "stratified": strat_keys is not None,
        "min_strat_count": MIN_STRAT_COUNT,
        "rare_cats_grouped": sorted(rare_cats),
    }
    with open(os.path.join(SPLITS_DIR, "split_summary.json"), "w") as f:
        json.dump(split_summary, f, indent=2)
    print(f"  Split summary → {SPLITS_DIR}/split_summary.json")

    # ── Step 5: Tokenizer & DataLoaders ──────────────────────────────
    tokenizer    = BertTokenizerFast.from_pretrained(MODEL_NAME)
    train_loader = DataLoader(AIDeclarationDataset(train_data, tokenizer),
                              batch_size=BATCH_SIZE, shuffle=True,  collate_fn=collate_fn)
    val_loader   = DataLoader(AIDeclarationDataset(val_data,   tokenizer),
                              batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    test_loader  = DataLoader(AIDeclarationDataset(test_data,  tokenizer),
                              batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    # ── Step 6: Model ────────────────────────────────────────────────
    model = JointNER_RE(MODEL_NAME, len(HEAD1_LABELS), len(HEAD2_LABELS), len(REL_LABELS)).to(DEVICE)
    print(f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # ── Step 7: Optimizer & scheduler ────────────────────────────────
    optimizer   = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    total_steps = len(train_loader) * EPOCHS
    scheduler   = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )

    # ── Step 8: Training loop ─────────────────────────────────────────
    best_val_f1 = 0.0
    best_path   = os.path.join(MODEL_DIR, "best_model.pt")
    history     = []
    epoch_log_path = os.path.join(RESULTS_DIR, "epoch_log.txt")

    print(f"\nTraining for {EPOCHS} epochs...")
    with open(epoch_log_path, "w") as elog:
        elog.write("epoch\ttrain_loss\th1_loss\th2_loss\tre_loss\t"
                   "val_h1_f1\tval_h2_f1\tval_re_f1\tavg_f1\n")

    for epoch in range(1, EPOCHS + 1):
        t = train_epoch(model, train_loader, optimizer, scheduler, DEVICE)
        print(f"\nEpoch {epoch}/{EPOCHS}  Loss: {t[0]:.4f} "
              f"(H1={t[1]:.4f}, H2={t[2]:.4f}, RE={t[3]:.4f})")

        h1_f1, h2_f1, re_f1 = evaluate(model, val_loader, DEVICE, "Validation")
        avg_f1 = (h1_f1 + h2_f1 + re_f1) / 3
        history.append({"epoch": epoch, "train_loss": t[0],
                        "val_h1_f1": h1_f1, "val_h2_f1": h2_f1, "val_re_f1": re_f1,
                        "avg_f1": avg_f1})

        # Append to TSV log so you can monitor live with Excel / tail
        with open(epoch_log_path, "a") as elog:
            elog.write(f"{epoch}\t{t[0]:.4f}\t{t[1]:.4f}\t{t[2]:.4f}\t{t[3]:.4f}\t"
                       f"{h1_f1:.4f}\t{h2_f1:.4f}\t{re_f1:.4f}\t{avg_f1:.4f}\n")

        if avg_f1 > best_val_f1:
            best_val_f1 = avg_f1
            torch.save(model.state_dict(), best_path)
            print(f"  ✓ Best model saved (avg F1={avg_f1:.4f})")

    # ── Step 9: Final test evaluation ────────────────────────────────
    print("\n" + "="*60 + "\n  FINAL TEST EVALUATION\n" + "="*60)
    model.load_state_dict(torch.load(best_path, map_location=DEVICE))

    # Redirect test results to a file as well as stdout
    import sys
    test_log_path = os.path.join(RESULTS_DIR, "test_results.txt")
    class Tee:
        """Write to both stdout and a file simultaneously."""
        def __init__(self, *files): self.files = files
        def write(self, s):
            for f in self.files: f.write(s)
        def flush(self):
            for f in self.files: f.flush()

    with open(test_log_path, "w", encoding="utf-8") as tlog:
        old_stdout = sys.stdout
        sys.stdout = Tee(old_stdout, tlog)
        evaluate(model, test_loader, DEVICE, "Test")
        sys.stdout = old_stdout

    print(f"\n  Test results saved → {test_log_path}")

    # ── Step 10: Save artefacts ───────────────────────────────────────
    # training_history.json  (full per-epoch metrics)
    with open(os.path.join(RESULTS_DIR, "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    # model checkpoint + tokenizer + config
    tokenizer.save_pretrained(os.path.join(MODEL_DIR, "tokenizer"))
    model_config = {
        "model_name":    MODEL_NAME,
        "max_len":       MAX_LEN,
        "batch_size":    BATCH_SIZE,
        "epochs":        EPOCHS,
        "lr":            LR,
        "lambda_ner1":   LAMBDA_NER1,
        "lambda_ner2":   LAMBDA_NER2,
        "lambda_re":     LAMBDA_RE,
        "seed":          SEED,
        "best_val_avg_f1": round(best_val_f1, 4),
    }
    with open(os.path.join(MODEL_DIR, "model_config.json"), "w") as f:
        json.dump(model_config, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  All outputs saved under: {OUTPUT_DIR}")
    print(f"  ├── splits/")
    print(f"  │   ├── train.jsonl          ({len(train_data)} samples)")
    print(f"  │   ├── val.jsonl            ({len(val_data)} samples)")
    print(f"  │   ├── test.jsonl           ({len(test_data)} samples)")
    print(f"  │   └── split_summary.json")
    print(f"  ├── model/")
    print(f"  │   ├── best_model.pt")
    print(f"  │   ├── label_map.json")
    print(f"  │   ├── model_config.json")
    print(f"  │   └── tokenizer/")
    print(f"  └── results/")
    print(f"      ├── epoch_log.txt")
    print(f"      ├── training_history.json")
    print(f"      └── test_results.txt")
    print(f"{'='*60}")
    print("Done!")


if __name__ == "__main__":
    main()