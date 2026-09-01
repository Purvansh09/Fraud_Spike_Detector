# Held-Out Test Report

Scored **once**, with the model and threshold fixed by `train.py` on validation. Nothing below was tuned against these numbers.

- test set SHA-256: `622d42b4fde8ed01...` — **MATCHES** the pinned split manifest
- transactions: 5,865, fraud: 121 (2.06%), episodes: 20
- operating threshold: **0.1349** (chosen on validation, unchanged)
- model: XGBoost {'max_depth': 6, 'n_estimators': 400, 'learning_rate': 0.05}, scale_pos_weight=48.8

## Headline

| metric | value |
|---|---|
| Precision | **82.3%** |
| Recall | **84.3%** |
| F1 | **0.833** |
| PR-AUC | **0.911** |
| ROC-AUC | 0.996 |

PR-AUC is the metric to read at 2% prevalence. ROC-AUC is shown because it is conventional, but it flatters every model on imbalanced data and should not be the number anyone quotes.

## Confusion matrix (raw counts)

| | predicted legit | predicted fraud |
|---|---|---|
| **actually legit** | 5,722 | 22 |
| **actually fraud** | 19 | 102 |

22 false positives out of 5,744 legitimate transactions = a **0.38% false-positive rate**. Phase 4 converts that into rupees.

## Versus baselines

| model | precision | recall | F1 |
|---|---|---|---|
| always-legitimate | 0.0% | 0.0% | 0.000 |
| rule: burst AND new device | 67.9% | 31.4% | 0.429 |
| **XGBoost** | **82.3%** | **84.3%** | **0.833** |

## Episode-level recall

**20 of 20 fraud episodes** had at least one transaction flagged (100.0%). This is the number a risk team actually cares about: catching an episode on its third transaction still stops the remaining ones.

## Recall by attack archetype

| archetype | fraud txns | txn recall | episodes | episodes caught |
|---|---|---|---|---|
| `blend_in` | 37 | 75.7% | 5 | 5/5 |
| `classic_burst` | 35 | 100.0% | 6 | 6/6 |
| `session_hijack` | 33 | 69.7% | 6 | 6/6 |
| `slow_drain` | 16 | 100.0% | 3 | 3/3 |

Reported because an average hides which attacks get through. A detector that is excellent on classic bursts and blind to session hijacks is not an 90%-recall detector, it is two different detectors.

## Threshold sensitivity

Provided in this single pass so the cost analysis in Phase 4 can pick an operating point without opening the test set a second time.

| threshold | precision | recall | F1 | flagged | false positives |
|---|---|---|---|---|---|
| 0.0500 | 77.0% | 88.4% | 0.823 | 139 | 32 |
| 0.1000 | 81.7% | 85.1% | 0.834 | 126 | 23 |
| 0.1349 (chosen) | 82.3% | 84.3% | 0.833 | 124 | 22 |
| 0.2000 | 86.3% | 83.5% | 0.849 | 117 | 16 |
| 0.3000 | 85.8% | 80.2% | 0.829 | 113 | 16 |
| 0.5000 | 87.4% | 68.6% | 0.769 | 95 | 12 |
| 0.7000 | 88.0% | 66.9% | 0.761 | 92 | 11 |
| 0.9000 | 93.6% | 60.3% | 0.734 | 78 | 5 |
