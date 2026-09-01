# Validation Report

Model selection and threshold tuning. **The held-out test set is not read by this script** — `train.py::load()` drops those rows before returning.

- train: 22,772 transactions, 457 fraud (2.01%)
- validation: 5,692 transactions, 123 fraud (2.16%)
- features: 33

Class imbalance handled with `scale_pos_weight = 48.8` (negatives per positive in train).

## Baselines

| model | precision | recall | F1 | PR-AUC | ROC-AUC |
|---|---|---|---|---|---|
| always-legitimate | 0.0% | 0.0% | 0.000 | 0.022 | 0.500 |
| rule: burst AND new device | 77.3% | 27.6% | 0.407 | — | — |
| logistic regression | 82.4% | 72.4% | 0.771 | 0.815 | 0.987 |

## XGBoost selection

Grid scored by **validation PR-AUC** — the honest ranking metric at 2% prevalence, where ROC-AUC flatters everything.

| max_depth | n_estimators | learning_rate | PR-AUC | ROC-AUC |
|---|---|---|---|---|
| 6 | 400 | 0.05 | 0.9444 | 0.9964 |
| 6 | 400 | 0.1 | 0.9410 | 0.9958 |
| 6 | 200 | 0.1 | 0.9395 | 0.9960 |
| 4 | 400 | 0.1 | 0.9393 | 0.9965 |
| 6 | 200 | 0.05 | 0.9383 | 0.9973 |
| 3 | 400 | 0.1 | 0.9371 | 0.9966 |
| 3 | 200 | 0.05 | 0.9369 | 0.9971 |
| 3 | 400 | 0.05 | 0.9367 | 0.9970 |
| 4 | 400 | 0.05 | 0.9356 | 0.9968 |
| 4 | 200 | 0.1 | 0.9326 | 0.9966 |
| 3 | 200 | 0.1 | 0.9318 | 0.9969 |
| 4 | 200 | 0.05 | 0.9266 | 0.9965 |

Winner: `max_depth=6, n_estimators=400, learning_rate=0.05` at validation PR-AUC **0.9444**.

## SMOTE ablation

| approach | validation PR-AUC |
|---|---|
| class weighting (`scale_pos_weight`) | **0.9444** |
| SMOTE oversampling | 0.9381 |

SMOTE scores worse here. We use class weighting regardless of the margin, for a reason that matters more than the number: SMOTE interpolates between fraud rows from *different accounts*, synthesising transactions whose velocity counts come from one account and whose device age comes from another. Those rows cannot occur in reality, and training on them undermines the causal guarantee Phase 2 was built to provide.

## Operating threshold

Chosen on validation by maximising F1: **0.1349** (validation precision 86.7%, recall 90.2%, F1 0.884).

This is the headline operating point. Phase 4's cost analysis may recommend a different one; if so, the change will be made on validation and stated explicitly, not chosen by looking at test.
