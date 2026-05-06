# AI Declaration Detection in Academic Papers

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Official code for the paper:

> **Detecting and Analysing AI Usage Declarations in Academic Publications**  
> *TPDL 2025 — International Conference on Theory and Practice of Digital Libraries*

---

## Overview

This repository provides the full pipeline for detecting, extracting, and analysing AI tool usage declarations in academic papers. We study two corpora:

- **CEUR-WS proceedings** — open-access workshop papers from CEUR-WS.org
- **Non-CEUR proceedings** — broader academic publications outside the CEUR ecosystem

We combine three complementary detection methods:

| Method | Description |
|--------|-------------|
| **Joint BERT NER + RE** | Multi-head BERT model for named-entity recognition (tool & usage spans) and relation extraction (tool → contribution role) |
| **LLM-based extraction** | Few-shot GPT prompting to extract structured AI declaration information |
| **Regex + Semantic similarity** | Rule-based pattern matching with position-aware scoring and sentence-transformer embeddings |

**Key findings across 15,409 non-CEUR papers:**
- 203 papers (1.3 %) explicitly declare AI tool usage
- 58 unique AI tools identified; ChatGPT (76 papers) and Grammarly (56 papers) are most common
- Language-enhancement tasks dominate (72.1 % of all role mentions)

---

## Repository Structure

```
.
├── Bert/
│   ├── joint_ner_re_train.py   # Joint NER + RE training (BERT multi-head)
│   └── inference.py            # Inference & evaluation on test split
│
├── CEUR-publication/
│   ├── AI_Usage_Complete_EDA-Meta_data.ipynb      # EDA: author metadata
│   ├── AI_Usage_Complete_EDA_AI_declarions.ipynb  # EDA: AI declarations
│   └── LLM_meta_AI.ipynb                          # LLM-based extraction
│
├── Non-CEUR-publication/
│   ├── EDA_non_CEUR_AI_declarations.ipynb         # EDA: non-CEUR corpus
│   ├── LLM_AI_dec_non-CEUR.ipynb                  # LLM extraction pipeline
│   ├── Regex.ipynb                                 # Regex-based detection
│   └── sementic.ipynb                              # Semantic similarity detection
│
├── website/
│   └── non_ceur_website.py     # Streamlit interactive dashboard
│
├── config_example.py           # Template for user-specific paths & API keys
├── requirements.txt
└── LICENSE
```

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate          # Linux / macOS
.venv\Scripts\activate             # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

> **GPU note:** PyTorch is listed without a CUDA build tag. For GPU training replace the `torch` line in `requirements.txt` with the appropriate wheel from [pytorch.org](https://pytorch.org/get-started/locally/).

---

## Configuration

Copy the template and fill in your paths and API keys:

```bash
cp config_example.py config.py
```

Edit `config.py`:

```python
# Paths
DATA_PATH  = "/path/to/Annotation.jsonl"   # annotated training data
OUTPUT_DIR = "/path/to/bert_output"        # model checkpoints & results

# API keys (only needed for LLM notebooks)
OPENAI_API_KEY = "sk-..."
```

---

## Usage

### 1. Train the Joint BERT Model

```bash
# Set DATA_PATH and OUTPUT_DIR in Bert/joint_ner_re_train.py (or config.py)
python Bert/joint_ner_re_train.py
```

Output layout after training:

```
<OUTPUT_DIR>/
├── splits/
│   ├── train.jsonl          # 70 % of data
│   ├── val.jsonl            # 10 % of data
│   ├── test.jsonl           # 20 % of data
│   └── split_summary.json
├── model/
│   ├── best_model.pt
│   ├── label_map.json
│   ├── model_config.json
│   └── tokenizer/
└── results/
    ├── epoch_log.txt
    ├── training_history.json
    └── test_results.txt
```

### 2. Run Inference & Evaluation

```bash
# Set BERT_DIR in Bert/inference.py to point to <OUTPUT_DIR>
python Bert/inference.py
```

Three evaluation tasks are reported:

| Task | Description | Metric |
|------|-------------|--------|
| Task 1 | Binary AI-tool detection (paper level) | Accuracy, P, R, F1 |
| Task 2a | Tool entity extraction | Exact + Partial micro-F1 |
| Task 2b | Usage entity extraction | Exact + Partial micro-F1 |
| Task 3 | `used_for` relation extraction (tool → role) | Exact + Partial micro-F1 |

Results are saved to `results/inference_results.json` and `results/inference_report.txt`.

### 3. Regex & Semantic Detection Notebooks

Open the notebooks in `Non-CEUR-publication/` with JupyterLab:

```bash
jupyter lab Non-CEUR-publication/
```

| Notebook | Method |
|----------|--------|
| `Regex.ipynb` | Position-aware regex pipeline (FRONT/BODY/BACK/FINAL multipliers) |
| `sementic.ipynb` | Sentence-transformer semantic similarity with section-aware scoring |
| `LLM_AI_dec_non-CEUR.ipynb` | Few-shot GPT extraction with JSON recovery |
| `EDA_non_CEUR_AI_declarations.ipynb` | Exploratory data analysis & figures |

### 4. Interactive Dashboard

```bash
streamlit run website/non_ceur_website.py
```

Place pre-generated figures and CSVs in `website/output/` (produced by the EDA notebooks).

---

## Data Format

### Annotation file (`Annotation.jsonl`)

Each line is a JSON object:

```json
{
  "text": "We used ChatGPT to improve the writing style of this paper.",
  "entities": [
    {"id": 1, "label": "Tool",  "start_offset": 8,  "end_offset": 15},
    {"id": 2, "label": "Usage", "start_offset": 19, "end_offset": 52},
    {"id": 3, "label": "Improve writing style", "start_offset": 19, "end_offset": 52}
  ],
  "relations": [
    {"type": "Used_for",  "from_id": 1, "to_id": 2},
    {"type": "Maps_to",   "from_id": 2, "to_id": 3}
  ]
}
```

**Entity labels:**
- `Tool` — name of the AI tool (e.g., *ChatGPT*, *Grammarly*)
- `Usage` — free-text description of how the tool was used
- Pre-defined category labels (e.g., *Improve writing style*, *Grammar and spelling check*, *Translation*, …)

**Relation types:**
- `Used_for` — links a Tool to a Usage span
- `Maps_to` — links a Usage span to a pre-defined category

---


## Reproducibility

All random seeds are fixed (`SEED = 42`). Data splits are saved as JSONL files after the first training run so that other models (e.g., RoBERTa, SpanBERT) can use the identical train/val/test partition.

```bash
# Re-run training from saved splits (edit joint_ner_re_train.py to load splits directly)
python Bert/joint_ner_re_train.py
```

---

## Requirements

See [requirements.txt](requirements.txt). Key dependencies:

| Package | Purpose |
|---------|---------|
| `torch` | Deep learning framework |
| `transformers` | BERT tokeniser & model |
| `seqeval` | NER sequence-level F1 |
| `sentence-transformers` | Semantic similarity |
| `streamlit` | Interactive dashboard |
| `pandas`, `matplotlib`, `seaborn` | Data analysis & visualisation |
| `PyMuPDF`, `pdfplumber` | PDF parsing |



## License

This project is released under the [MIT License](LICENSE).
