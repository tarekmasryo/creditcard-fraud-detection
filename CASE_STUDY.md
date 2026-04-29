# Case Study — Practical Fraud Modeling and Threshold Tuning

## Overview

This repository implements a practical fraud detection workflow focused on decision-making, not just model accuracy.

The notebook trains baseline and tree-based models, calibrates probability scores, tunes thresholds on validation data, and reports final operating points on a held-out test window.

The goal is to make the fraud decision trade-off explicit: catching more fraud while keeping false-positive alert volume manageable.

## Problem

Credit card fraud detection is a highly imbalanced classification problem.

False positives can create customer friction and manual review cost. False negatives can create direct financial loss. A useful model therefore needs more than a high global metric; it needs calibrated scores and threshold policies that can be inspected and adjusted.

## Goals

- Build a leakage-aware fraud modeling workflow.
- Use time-ordered train, validation, and test windows.
- Compare Logistic Regression, Random Forest, and XGBoost.
- Use AUPRC as the primary metric for imbalanced fraud detection.
- Calibrate probability scores for threshold-based decisions.
- Select operating thresholds on validation data only.
- Report final performance on a held-out test window.
- Export optional artifacts for lightweight reuse.

## Approach

### 1. Data health and focused EDA

The notebook checks the dataset shape, duplicate rows, target distribution, amount behavior, and time-derived proxy patterns.

The `Time` feature is treated as a relative seconds-from-start field, not a real calendar timestamp. Any hour or day-part features are proxy features derived from that relative field.

### 2. Time-aware evaluation

The workflow uses time-ordered train, validation, and test windows to reduce look-ahead leakage.

Thresholds are selected using validation data and then evaluated on the held-out test window. This keeps the final test report separate from model and threshold selection.

### 3. Models

The notebook compares:

- Logistic Regression
- Random Forest
- XGBoost

It also includes a lightweight resampling diagnostic using Logistic Regression. That section is used as a baseline diagnostic only. The final recommendation is based on calibrated RF/XGB models, validation-selected thresholds, and operational trade-offs.

### 4. Calibration

Raw model scores are not automatically reliable probabilities. The workflow uses probability calibration to make threshold-based decisions more interpretable.

Calibration is evaluated with Brier Score and ECE alongside ranking metrics such as AUPRC.

### 5. Threshold decisions

The notebook evaluates two operating styles:

- Precision-first thresholding for low alert burden.
- Simple cost-aware thresholding using illustrative FP/FN costs.

The cost values are examples only. In a real setting, they should be replaced with business-specific investigation and fraud-loss assumptions.

## Outputs

When artifact saving is enabled in the notebook, generated files may include:

- `xgb_pipe.joblib`
- `xgb_calibrated.joblib`
- `rf_pipe.joblib`
- `rf_calibrated.joblib`
- `thresholds.json`

Artifacts are written to the configured artifacts directory and are not committed by default.

## Key limitations

- `Time` is relative seconds-from-start, not a real timestamp.
- `V1` to `V28` are anonymized PCA-transformed features and are not directly interpretable.
- Cost assumptions are illustrative.
- Thresholds and calibration should be revalidated under data drift.
- This repository is an analytical modeling workflow, not a complete production fraud monitoring system.

## Next steps

- Add sensitivity analysis for different FP/FN cost assumptions.
- Track threshold stability over time.
- Add model drift and calibration drift checks.
- Package the scoring path as a small API or batch job if needed.
