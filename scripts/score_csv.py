#!/usr/bin/env python3
"""Score a CSV using exported fraud-detection artifacts.

Expected workflow:
1. Run the notebook and export artifacts to ./artifacts.
2. Score a CSV with the raw dataset columns:
   Time, V1..V28, Amount, and optionally Class.

The script recreates the same lightweight feature engineering used in the
notebook before calling the exported calibrated model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


MODEL_FILES = {
    "xgb": "xgb_calibrated.joblib",
    "rf": "rf_calibrated.joblib",
}

POLICY_KEYS = {
    "min_cost": {"xgb": "XGB_Thr_MinCost", "rf": "RF_Thr_MinCost"},
    "p90": {"xgb": "XGB_Thr_P90", "rf": "RF_Thr_P90"},
}

FEATURE_COLUMNS = (
    [f"V{i}" for i in range(1, 29)]
    + [
        "Amount",
        "_log_amount",
        "Hour_from_start_mod24",
        "is_night_proxy",
        "is_business_hours_proxy",
    ]
)

REQUIRED_RAW_COLUMNS = ["Time", "Amount", *[f"V{i}" for i in range(1, 29)]]


def load_threshold(art_dir: Path, model: str, policy: str) -> float:
    thresholds_path = art_dir / "thresholds.json"
    if not thresholds_path.exists():
        raise FileNotFoundError(f"Missing thresholds file: {thresholds_path}")

    data = json.loads(thresholds_path.read_text(encoding="utf-8"))
    key = POLICY_KEYS[policy][model]
    if key not in data:
        raise KeyError(f"Threshold key '{key}' not found in {thresholds_path.name}")
    return float(data[key])


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    missing = [col for col in REQUIRED_RAW_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required input columns: {missing}")

    features_df = df.copy()
    features_df["Hour_from_start_mod24"] = ((features_df["Time"] // 3600) % 24).astype(int)
    features_df["is_night_proxy"] = (
        features_df["Hour_from_start_mod24"].isin([22, 23, 0, 1, 2, 3, 4, 5]).astype(int)
    )
    features_df["is_business_hours_proxy"] = features_df["Hour_from_start_mod24"].between(9, 17).astype(int)
    features_df["_log_amount"] = np.log1p(features_df["Amount"])

    return features_df[FEATURE_COLUMNS]


def main() -> int:
    parser = argparse.ArgumentParser(description="Score a CSV using exported fraud artifacts.")
    parser.add_argument("--csv", required=True, help="Path to input CSV with raw credit-card fraud columns")
    parser.add_argument("--out", required=True, help="Path to write scored CSV")
    parser.add_argument("--artifacts", default="artifacts", help="Artifacts directory (default: ./artifacts)")
    parser.add_argument("--model", choices=["xgb", "rf"], default="xgb", help="Which calibrated model to use")
    parser.add_argument("--policy", choices=["min_cost", "p90"], default="min_cost", help="Threshold policy")
    args = parser.parse_args()

    art_dir = Path(args.artifacts)
    model_path = art_dir / MODEL_FILES[args.model]
    if not model_path.exists():
        raise FileNotFoundError(
            f"Missing model artifact: {model_path}. Run the notebook and export artifacts first."
        )

    model = joblib.load(model_path)
    threshold = load_threshold(art_dir, args.model, args.policy)

    df = pd.read_csv(args.csv)
    X = build_features(df)

    if not hasattr(model, "predict_proba"):
        raise TypeError("Loaded artifact does not expose predict_proba().")

    proba = model.predict_proba(X)[:, 1]

    out_df = df.copy()
    out_df["fraud_proba"] = proba
    out_df["fraud_pred"] = (proba >= threshold).astype(int)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)

    print(f"Wrote: {out_path} (model={args.model}, policy={args.policy}, threshold={threshold:.6f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
