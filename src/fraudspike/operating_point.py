"""Reconcile the two operating points this project reports.

There are two thresholds in play and they must never be confused:

  * **0.1349** -- F1-optimal, chosen on validation in Phase 3. The Phase 3 test report
    quotes this one.
  * **0.1282** -- cost-optimal, chosen on validation in Phase 4. This is what the service
    actually deploys.

Both were selected on the validation split. Neither was selected by looking at test. The
model's scores on the test set were computed once; quoting those same scores at a second
validation-chosen threshold is a different presentation of one measurement, not a second
bite at the held-out data.

This script exists so the numbers in README.md, reports/test_report.md and the dashboard
all come from one place instead of being retyped and drifting apart.

Writes reports/operating_point.md.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from .costs import load_split

_ROOT = Path(__file__).resolve().parents[2]
MODELS = _ROOT / "models"
REPORTS = _ROOT / "reports"


def metrics_at(df: pd.DataFrame, scores: np.ndarray, thr: float) -> dict:
    pred = scores >= thr
    y = df["is_fraud"].to_numpy() == 1

    tp = int((pred & y).sum())
    fp = int((pred & ~y).sum())
    fn = int((~pred & y).sum())
    tn = int((~pred & ~y).sum())

    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0

    fraud = df[y].assign(_flag=pred[y])
    caught = fraud.groupby("episode_id")["_flag"].any()

    return {
        "threshold": thr,
        "precision": prec, "recall": rec, "f1": f1,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "episodes_caught": int(caught.sum()),
        "episodes_total": int(len(caught)),
    }


def main() -> None:
    bundle = joblib.load(MODELS / "model.joblib")
    model, features = bundle["model"], bundle["features"]
    thr_deployed = float(bundle["threshold"])            # cost-optimal, Phase 4
    thr_f1 = float(bundle.get("threshold_f1", thr_deployed))  # F1-optimal, Phase 3

    test = load_split("test")
    scores = model.predict_proba(test[features])[:, 1]
    y = test["is_fraud"].to_numpy()

    a = metrics_at(test, scores, thr_f1)
    b = metrics_at(test, scores, thr_deployed)
    pr_auc = average_precision_score(y, scores)
    roc = roc_auc_score(y, scores)

    L: list[str] = []

    def w(line: str = "") -> None:
        L.append(line)
        try:
            print(line)
        except UnicodeEncodeError:
            print(line.encode("ascii", "replace").decode("ascii"))

    w("# Operating Point Reconciliation")
    w()
    w("Two thresholds appear in this project. Both were chosen on the **validation** split; "
      "neither was chosen by looking at test. The model's test scores were computed once, "
      "and the table below is that single measurement read at two cut-offs.")
    w()
    w("| | F1-optimal (Phase 3) | **cost-optimal (deployed)** |")
    w("|---|---|---|")
    w(f"| threshold | {a['threshold']:.4f} | **{b['threshold']:.4f}** |")
    w(f"| chosen by | best F1 on validation | best net rupee saving on validation |")
    w(f"| precision | {a['precision']:.1%} | **{b['precision']:.1%}** |")
    w(f"| recall | {a['recall']:.1%} | **{b['recall']:.1%}** |")
    w(f"| F1 | {a['f1']:.3f} | **{b['f1']:.3f}** |")
    w(f"| true positives | {a['tp']} | {b['tp']} |")
    w(f"| false positives | {a['fp']} | {b['fp']} |")
    w(f"| fraud missed | {a['fn']} | {b['fn']} |")
    w(f"| episodes caught | {a['episodes_caught']}/{a['episodes_total']} | "
      f"{b['episodes_caught']}/{b['episodes_total']} |")
    w()
    w(f"Threshold-independent, so identical for both: **PR-AUC {pr_auc:.3f}**, "
      f"ROC-AUC {roc:.3f}. PR-AUC is the one to read at 2% prevalence.")
    w()
    w("**Which to quote.** The deployed column. The service, the dashboard and the cost "
      "analysis all run at "
      f"{b['threshold']:.4f}, so that is the number the demo shows. The Phase 3 column is "
      "kept because it is what the original held-out report recorded, and deleting it to "
      "make the story tidier would be exactly the kind of edit this project is built to "
      "avoid.")
    w()

    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "operating_point.md").write_text("\n".join(L), encoding="utf-8")
    print(f"\n[written] {REPORTS / 'operating_point.md'}")


if __name__ == "__main__":
    main()
