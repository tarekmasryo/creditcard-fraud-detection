# 💳 Credit Card Fraud Detection — Cost-Aware Pipeline

A practical fraud detection workflow with time-aware evaluation, probability calibration, and explicit threshold policies
for controlling false positives vs false negatives.

Case study: `CASE_STUDY.md`

---

## What this repository includes
- Notebook: `Credit-Card-Fraud-Detection-A-Pipeline-Journey.ipynb`
- Exported artifacts under `./artifacts/` (models + thresholds + run metadata)
- Scoring script: `scripts/score_csv.py` (optional)

---

## Dataset
Source: Kaggle “Credit Card Fraud Detection” dataset (`creditcard.csv`).

Expected columns:
- `Time`, `V1`…`V28`, `Amount`, `Class` (0 = normal, 1 = fraud)

### Local (recommended)
1) Download the dataset CSV.
2) Place it at:
`data/raw/creditcard.csv`

### Kaggle
The notebook also supports the common Kaggle input path:
`/kaggle/input/creditcardfraud/creditcard.csv`

---

## Getting started

### 1) Install
```bash
python -m venv .venv
# Windows: .\.venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate

pip install -r requirements.txt
```

### 2) Run the notebook
Open and run:
- `Credit-Card-Fraud-Detection-A-Pipeline-Journey.ipynb`

The notebook will:
- train baseline + stronger models
- calibrate probabilities
- select threshold policies
- export artifacts to `./artifacts/`

---

## Artifacts
Artifacts are written to `./artifacts/` (models, thresholds, and run metadata).
See `artifacts/README.md` for the expected files produced by the notebook.

---

## Score a CSV (optional)
After exporting artifacts, you can score any CSV with the same feature columns:

```bash
python scripts/score_csv.py --csv data/raw/creditcard.csv --out artifacts/scored.csv --model xgb --policy min_cost
```

The output adds:
- `fraud_proba`
- `fraud_pred` (0/1)

---

## Methodology notes
- Leak-safe evaluation via time-based train/test windows.
- Calibration produces probabilities suitable for threshold policies.
- Threshold policies define operating points (e.g., min expected cost).

---

## License
MIT (code). Dataset licensing depends on the dataset source where you download it.
