# Dataset schema

Expected columns (Kaggle creditcard fraud dataset):
- `Time`
- `V1` ... `V28`
- `Amount`
- `Class` (0 = normal, 1 = fraud)

This repo expects the raw CSV at `data/raw/creditcard.csv` for local runs.
On Kaggle, the notebook can read from the input dataset path.
