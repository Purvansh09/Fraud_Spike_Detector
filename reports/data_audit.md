# Data Audit

- transactions: **34,336**, accounts: **600**, days: **61**
- fraud transactions: **708** (2.06%), episodes: **120**

## Temporal split occupancy

| split | days | transactions | fraud txns | fraud rate | episodes |
|---|---|---|---|---|---|
| train | [0, 40) | 22,775 | 460 | 2.02% | 79 |
| val | [40, 50) | 5,704 | 135 | 2.37% | 23 |
| test | [50, 61) | 5,857 | 113 | 1.93% | 18 |

## Single-marker separability

Each row is a rule a risk team could write without any model. High precision here would mean the dataset is trivially separable and the model adds nothing.

| marker | flagged | precision | recall | F1 |
|---|---|---|---|---|
| velocity: 4+ txns in 10 min | 2,025 | 17.1% | 49.0% | 0.254 |
| velocity: 5+ txns in 5 min | 790 | 22.4% | 25.0% | 0.236 |
| device new to account (<1h) | 2,248 | 31.3% | 99.4% | 0.476 |
| hour in 00:00-05:00 | 4,176 | 9.7% | 57.2% | 0.166 |
| high-risk category | 2,153 | 28.9% | 87.9% | 0.435 |
| city != account's usual | 2,916 | 16.8% | 69.1% | 0.270 |
| amount > 3x account median | 3,132 | 12.7% | 56.4% | 0.208 |

**Burst AND new-device combined:** flagged 426, precision 81.2%, recall 48.9%, F1 0.610

## Confounder health

- transactions inside a 4-in-10min burst: **1,678 legitimate** vs **347 fraudulent** (83% of bursts are legitimate)
- first-time-seen devices: **1,544 legitimate** vs **704 fraudulent**
- transactions in 00:00-05:00: **3,771 legitimate** vs **405 fraudulent**
- amount overlap: legitimate p50 Rs 1,201 / p99 Rs 13,212; fraud p50 Rs 4,921 / p99 Rs 42,505
