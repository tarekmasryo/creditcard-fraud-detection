# Dataset Schema

This repository expects the Kaggle Credit Card Fraud Detection dataset.

## Expected raw columns

| Column | Type | Description |
|---|---:|---|
| `Time` | numeric | Relative seconds elapsed from the first transaction in the dataset. This is not a real calendar timestamp. |
| `V1` to `V28` | numeric | Anonymized PCA-transformed features provided by the dataset source. |
| `Amount` | numeric | Transaction amount. |
| `Class` | integer | Target label: `0` = normal transaction, `1` = fraudulent transaction. |

## Derived notebook features

The notebook may create additional features for analysis and modeling, including:

| Feature | Description |
|---|---|
| `Amount_log1p` | Log-transformed transaction amount using `log1p`. |
| `Hour_from_start_mod24` | Hour-like proxy derived from relative `Time`. |
| `is_night_proxy` | Proxy flag derived from `Hour_from_start_mod24`. |
| `is_business_hours_proxy` | Proxy flag derived from `Hour_from_start_mod24`. |

## Important notes

- `Time` is relative seconds-from-start, not a confirmed real-world clock timestamp.
- Hour and day-part features are proxy features, not verified calendar-time features.
- `V1` to `V28` are PCA components and should not be interpreted as original business features.
- Local runs expect the raw CSV at `data/raw/creditcard.csv`.
- On Kaggle, the notebook can read from `/kaggle/input/creditcardfraud/creditcard.csv`.
