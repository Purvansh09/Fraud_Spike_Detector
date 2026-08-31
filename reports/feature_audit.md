# Feature Audit

Train split only: **22,775** transactions, **460** fraud (2.02%). Validation and test are untouched.

## Single-feature discriminative power

AUC is direction-corrected (max of AUC, 1-AUC). Anything above **0.98** is flagged for investigation as a probable leak.

| feature | AUC alone | missing | flag |
|---|---|---|---|
| `device_age_secs` | 0.930 | 0.0% |  |
| `is_high_risk_category` | 0.915 | 0.0% |  |
| `txn_count_15m` | 0.862 | 0.0% |  |
| `amt_ratio_to_window_max` | 0.858 | 84.7% |  |
| `txn_count_5m` | 0.857 | 0.0% |  |
| `txn_count_60m` | 0.849 | 0.0% |  |
| `secs_since_prev` | 0.847 | 2.6% |  |
| `device_txns_prior` | 0.811 | 0.0% |  |
| `cat_share_prior` | 0.792 | 2.6% |  |
| `cat_txns_prior` | 0.771 | 0.0% |  |
| `hour_dist_from_acct_norm` | 0.764 | 2.6% |  |
| `hour` | 0.749 | 0.0% |  |
| `is_night` | 0.738 | 0.0% |  |
| `city_txns_prior` | 0.731 | 0.0% |  |
| `distinct_devices_prior` | 0.705 | 0.0% |  |
| `amt_ratio_to_prior_median` | 0.705 | 2.6% |  |
| `amt_pct_rank_prior` | 0.698 | 2.6% |  |
| `amt_log_zscore` | 0.683 | 7.9% |  |
| `mean_gap_prev5` | 0.671 | 5.3% |  |
| `amount_sum_5m` | 0.670 | 0.0% |  |
| `amount_sum_15m` | 0.668 | 0.0% |  |
| `amount_sum_60m` | 0.668 | 0.0% |  |
| `log_amount` | 0.662 | 0.0% |  |
| `is_new_category` | 0.643 | 0.0% |  |
| `acct_age_days` | 0.558 | 2.6% |  |
| `is_new_device` | 0.550 | 0.0% |  |
| `city_changed_from_prev` | 0.539 | 2.6% |  |
| `acct_txns_prior` | 0.533 | 0.0% |  |
| `is_new_city` | 0.532 | 0.0% |  |
| `channel_card` | 0.520 | 0.0% |  |
| `channel_netbanking` | 0.510 | 0.0% |  |
| `channel_upi` | 0.508 | 0.0% |  |
| `channel_wallet` | 0.502 | 0.0% |  |

> No feature exceeds the 0.98 threshold. The strongest alone is `device_age_secs` at AUC 0.930 — informative, not decisive, which is what we want.

## Highly correlated pairs (|r| > 0.95)

| feature A | feature B | r |
|---|---|---|
| `amount_sum_5m` | `amount_sum_15m` | 0.991 |
| `amount_sum_15m` | `amount_sum_60m` | 0.969 |
| `amount_sum_5m` | `amount_sum_60m` | 0.960 |
| `txn_count_5m` | `txn_count_15m` | 0.960 |

Kept for now — XGBoost tolerates collinearity — but SHAP attribution will split between twins, so this table is the reference when explanations look diluted.
