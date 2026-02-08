# Case Study — Credit Card Fraud Detection (Cost-Aware Pipeline)

## Overview
This repository implements a fraud detection workflow that is designed for **decision-making**, not just model accuracy.
It produces **calibrated probabilities** and exports **threshold policies** so teams can control the trade-off between
blocking legitimate transactions and missing fraud.

## The real problem
Fraud detection is asymmetric-risk classification:
- **False positives** (legitimate transactions flagged) create customer friction and operational review cost.
- **False negatives** (fraud missed) create direct financial loss and abuse risk.

A model is useful only if its scores can be turned into a consistent policy.

## Goals (definition of done)
**Functional goals**
- Train strong baselines for highly imbalanced data.
- Produce reliable probability estimates (calibration).
- Select explicit operating thresholds for deployment decisions.

**Engineering goals**
- Leak-safe evaluation using time-aware splits.
- Reproducible runs (set seed (random_state), deterministic preprocessing).
- Export artifacts (models + thresholds) for reuse.

## Approach
### 1) Leak-safe evaluation (time-aware)
Random splits can leak temporal patterns in transaction data.
This workflow evaluates using **time-based windows** to better reflect real deployment conditions.

### 2) Models suited for sparse fraud signals
The notebook compares baselines and stronger models (e.g., RF/XGB) with imbalance-aware training choices.

### 3) Calibrated probabilities
Raw model scores are not automatically usable as probabilities.
Calibration makes the score interpretable for threshold selection and policy design.

### 4) Threshold policies (decision layer)
Instead of defaulting to 0.5, the workflow exports thresholds such as:
- a **minimum expected cost** threshold (based on FP vs FN costs)
- a conservative high-precision threshold (when review capacity is limited)

This makes operating trade-offs explicit and repeatable.

## Outputs
The notebook can export:
- trained pipelines (base + calibrated)
- `thresholds.json` (thresholds + cost assumptions)

Artifacts are saved under `./artifacts/`.

## Usage
Setup and execution steps are documented in `README.md`.

Minimal flow:
- Run the notebook to train and export artifacts.
- Score a CSV with `scripts/score_csv.py` using a chosen model and policy.

## Limitations
- The dataset uses anonymized PCA features; real production features and drift patterns differ.
- Cost values (FP/FN) are illustrative; real policies should be derived from business constraints.
- Production deployments should include monitoring, access controls, and incident response playbooks.

## Next steps
- Add a release gate: minimum recall on critical slices + maximum review load.
- Track stability over time (weekly baselines, drift alerts).
- Add a small monitoring notebook for threshold performance (precision/recall vs volume).
