# Feature Audit

Train split only: **22,772** transactions, **457** fraud (2.01%). Validation and test are untouched.

## Single-feature discriminative power

AUC is direction-corrected (max of AUC, 1-AUC). Anything above **0.98** is flagged for investigation as a probable leak.

| feature | AUC alone | missing | flag |
|---|---|---|---|
| `device_age_secs` | 0.882 | 0.0% |  |
| `txn_count_60m` | 0.846 | 0.0% |  |
| `txn_count_15m` | 0.832 | 0.0% |  |
| `secs_since_prev` | 0.825 | 2.6% |  |
| `is_high_risk_category` | 0.806 | 0.0% |  |
| `txn_count_5m` | 0.804 | 0.0% |  |
| `device_txns_prior` | 0.783 | 0.0% |  |
| `amt_ratio_to_window_max` | 0.771 | 84.8% |  |
| `amount_sum_15m` | 0.769 | 0.0% |  |
| `amount_sum_60m` | 0.767 | 0.0% |  |
| `amount_sum_5m` | 0.759 | 0.0% |  |
| `amt_ratio_to_prior_median` | 0.754 | 2.6% |  |
| `amt_log_zscore` | 0.739 | 7.9% |  |
| `amt_pct_rank_prior` | 0.739 | 2.6% |  |
| `hour` | 0.736 | 0.0% |  |
| `log_amount` | 0.726 | 0.0% |  |
| `hour_dist_from_acct_norm` | 0.713 | 2.6% |  |
| `cat_share_prior` | 0.713 | 2.6% |  |
| `cat_txns_prior` | 0.703 | 0.0% |  |
| `is_night` | 0.687 | 0.0% |  |
| `city_txns_prior` | 0.667 | 0.0% |  |
| `distinct_devices_prior` | 0.646 | 0.0% |  |
| `mean_gap_prev5` | 0.630 | 5.3% |  |
| `is_new_category` | 0.614 | 0.0% |  |
| `is_new_device` | 0.541 | 0.0% |  |
| `acct_txns_prior` | 0.535 | 0.0% |  |
| `city_changed_from_prev` | 0.523 | 2.6% |  |
| `acct_age_days` | 0.521 | 2.6% |  |
| `is_new_city` | 0.515 | 0.0% |  |
| `channel_netbanking` | 0.509 | 0.0% |  |
| `channel_wallet` | 0.507 | 0.0% |  |
| `channel_card` | 0.506 | 0.0% |  |
| `channel_upi` | 0.504 | 0.0% |  |

> No feature exceeds the 0.98 threshold. The strongest alone is `device_age_secs` at AUC 0.882 — informative, not decisive, which is what we want.

## Highly correlated pairs (|r| > 0.95)

| feature A | feature B | r |
|---|---|---|
| `amount_sum_5m` | `amount_sum_15m` | 0.982 |
| `amount_sum_15m` | `amount_sum_60m` | 0.969 |
| `txn_count_5m` | `txn_count_15m` | 0.950 |

Kept for now — XGBoost tolerates collinearity — but SHAP attribution will split between twins, so this table is the reference when explanations look diluted.
