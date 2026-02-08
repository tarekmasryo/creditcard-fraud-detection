#!/usr/bin/env python3
"""Score a CSV using exported artifacts.

Expected workflow:
1) Run the notebook and export artifacts to ./artifacts
2) Score a CSV that has the same feature columns (Time, V1..V28, Amount)

This script does not train models. It only loads artifacts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd


MODEL_FILES = {
    "xgb": "xgb_calibrated.joblib",
    "rf": "rf_calibrated.joblib",
}

POLICY_KEYS = {
    # These keys match the notebook export.
    "min_cost": {"xgb": "XGB_Thr_MinCost", "rf": "RF_Thr_MinCost"},
    "p90": {"xgb": "XGB_Thr_P90", "rf": "RF_Thr_P90"},
}


def load_threshold(art_dir: Path, model: str, policy: str) -> float:
    thresholds_path = art_dir / "thresholds.json"
    if not thresholds_path.exists():
        raise FileNotFoundError(f"Missing thresholds file: {thresholds_path}")

    data = json.loads(thresholds_path.read_text(encoding="utf-8"))
    key = POLICY_KEYS[policy][model]
    if key not in data:
        raise KeyError(f"Threshold key '{key}' not found in {thresholds_path.name}")
    return float(data[key])


def main() -> int:
    p = argparse.ArgumentParser(description="Score a CSV using exported fraud artifacts.")
    p.add_argument("--csv", required=True, help="Path to input CSV (must contain Time, V1..V28, Amount)")
    p.add_argument("--out", required=True, help="Path to write scored CSV")
    p.add_argument("--artifacts", default="artifacts", help="Artifacts directory (default: ./artifacts)")
    p.add_argument("--model", choices=["xgb", "rf"], default="xgb", help="Which calibrated model to use")
    p.add_argument("--policy", choices=["min_cost", "p90"], default="min_cost", help="Threshold policy")
    args = p.parse_args()

    art_dir = Path(args.artifacts)
    model_path = art_dir / MODEL_FILES[args.model]
    if not model_path.exists():
        raise FileNotFoundError(
            f"Missing model artifact: {model_path}. Run the notebook and export artifacts first."
        )

    model = joblib.load(model_path)
    thr = load_threshold(art_dir, args.model, args.policy)

    df = pd.read_csv(args.csv)
    X = df.drop(columns=["Class"], errors="ignore")

    if not hasattr(model, "predict_proba"):
        raise TypeError("Loaded artifact does not expose predict_proba().")

    proba = model.predict_proba(X)[:, 1]
    out_df = df.copy()
    out_df["fraud_proba"] = proba
    out_df["fraud_pred"] = (proba >= thr).astype(int)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"Wrote: {out_path} (threshold={thr:.6f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
