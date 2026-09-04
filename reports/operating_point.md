# Operating Point Reconciliation

Two thresholds appear in this project. Both were chosen on the **validation** split; neither was chosen by looking at test. The model's test scores were computed once, and the table below is that single measurement read at two cut-offs.

| | F1-optimal (Phase 3) | **cost-optimal (deployed)** |
|---|---|---|
| threshold | 0.1349 | **0.1282** |
| chosen by | best F1 on validation | best net rupee saving on validation |
| precision | 82.3% | **82.4%** |
| recall | 84.3% | **85.1%** |
| F1 | 0.833 | **0.837** |
| true positives | 102 | 103 |
| false positives | 22 | 22 |
| fraud missed | 19 | 18 |
| episodes caught | 20/20 | 20/20 |

Threshold-independent, so identical for both: **PR-AUC 0.911**, ROC-AUC 0.996. PR-AUC is the one to read at 2% prevalence.

**Which to quote.** The deployed column. The service, the dashboard and the cost analysis all run at 0.1282, so that is the number the demo shows. The Phase 3 column is kept because it is what the original held-out report recorded, and deleting it to make the story tidier would be exactly the kind of edit this project is built to avoid.
