# Named Entity Recognition on the Archelec Corpus

This project was carried out as part of the **Machine Learning for NLP** course. It applies Named Entity Recognition (NER) and discourse analysis techniques to the **Archelec corpus** — a collection of French electoral tracts spanning from 1973 to 1993 (legislative elections), and some recent files from 2015, 2017, 2019 and 2020

The full report is available in [`projet_archelec/report/Wiam_report.pdf`](projet_archelec/report/Wiam_report.pdf).

---

## Corpus

The **Archelec corpus** consists of scanned and OCR-processed electoral tracts from French elections. Text files are organized by year and election type under `projet_archelec/data/raw/arkindex_archelec/text_files/`. Metadata (party, department, candidate) is provided in (https://archelec.sciencespo.fr/explorer).

The text extraction pipeline is adapted from the repository:
> https://gitlab.teklia.com/ckermorvant/arkindex_archelec

---

## Project Structure

```
projet_archelec/
│
├── data/
│   ├── raw/                        # Raw corpus (text files + archelec.csv metadata)
│   ├── annotated/                  # Manually annotated JSON files (NER gold labels)
│   ├── processed/                  # Cleaned CSVs, statistics, spaCy baseline outputs
│   ├── splits/                     # Train/test splits (json)
│   └── results/
│       ├── output_best_model/
│       │   ├── leaders_mentions.xlsx           # GLiNER detections of political leaders
│       │   ├── leaders_mentions_classified.xlsx # + sentiment classification (XGBoost)
│       ├── Gliner/                 # GLiNER evaluation outputs
│       ├── CamemBERT/              # CamemBERT evaluation outputs
│       └── Spacy/                  # spaCy baseline evaluation outputs
│
├── models/
│   ├── Gliner/
│   │   ├── checkpoint-700/         # GLiNER fine-tuned checkpoint (epoch 700)
│   │   └── checkpoint-800/         # GLiNER fine-tuned checkpoint (epoch 800) ← used
│   ├── model-best/                 # Best spaCy NER model
│   └── model-last/                 # Last spaCy NER model checkpoint
│
├── notebooks/
│   ├── gliner.ipynb                # GLiNER fine-tuning and evaluation
│   ├── baseline_spacy.ipynb        # spaCy NER baseline
│   ├── leaders_detection.ipynb     # Detecting political leader mentions with GLiNER
│   ├── stage1_distantsup_sspcloud.ipynb   # Distant supervision annotation (stage 1) - Camembert
│   ├── stage2_manual_sspcloud.ipynb       # Manual annotation refinement (stage 2) - Camembert
│   └── analyses/
│       ├── modélisation_analyse_president.ipynb
│       │     Pipeline 1 (TF-IDF + XGBoost) and Pipeline 2 (CamemBERT + CatBoost)
│       │     for sentiment classification of president mentions; full discourse analysis
│       └──    by party, year, political bloc, and presidential era.        
│
├── scripts/
│   ├── 01_explore_data.py          # Corpus statistics and exploration
│   ├── 02_load_texts.py            # Text loading utilities
│   ├── 03_annotate.py              # Annotation helpers
│   ├── make_distantsup_bio.py      # Build BIO corpus via distant supervision
│   ├── split_train_test.py         # Train/val/test split generation
│   ├── evaluation.py               # Model evaluation metrics
│   └── text_extraction/
│       ├── extract_text.py         # Text extraction from Arkindex (pre-2000 files)
│       └── text_recent_years.py    # Text extraction for recent years
│
├── report/
│   └── Wiam_report.pdf             # Full project report
│
├── requirements.txt                # Python dependencies
├── pyproject.toml                  # Project configuration (uv)
└── .venv/                          # Virtual environment (managed by uv)
```

---

## Main Tasks

| Task | Method | Notebook |
|---|---|---|
| NER (persons, orgs, locations) | GLiNER fine-tuned on Archelec | `gliner.ipynb` |
| NER baseline | spaCy `fr_core_news_lg` | `baseline_spacy.ipynb` |
| Leader mention detection | GLiNER (checkpoint-800) | `leaders_detection.ipynb` |
| Sentiment toward president | TF-IDF + XGBoost / CamemBERT + CatBoost | `analyses/modélisation_analyse_president.ipynb` |

---

## Setup

```bash
# Install uv (if not already installed)
pip install uv

# Create virtual environment and install dependencies
cd projet_archelec
uv venv
uv pip install -r requirements.txt
```

Requires Python 3.11+. Apple Silicon (MPS) is supported for GLiNER and CamemBERT inference.
