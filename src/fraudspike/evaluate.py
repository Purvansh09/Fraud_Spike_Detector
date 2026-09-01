"""Held-out test evaluation. RUNS ONCE.

This is the only script in the project that reads the test split. It consumes the model
and threshold that train.py already committed to, and changes nothing based on what it
sees -- if a number here disappoints, the honest response is to report it, not to go back
and retune. Retuning against these results would silently turn the test set into a second
validation set and void every metric below.

To keep that promise cheap to keep, this script reports everything Phase 4 could plausibly
need in a single pass: the operating point, a threshold sweep, per-archetype recall, and
episode-level recall. There is no reason to open the test set again.

Writes reports/test_report.md.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

RAW = Path("data/raw")
SPLITS = Path("data/splits")
MODELS = Path("models")
REPORTS = Path("reports")


def prf_at(scores: np.ndarray, y: np.ndarray, thr: float) -> tuple[float, float, float, dict]:
    pred = scores >= thr
    tp = int((pred & (y == 1)).sum())
    fp = int((pred & (y == 0)).sum())
    fn = int((~pred & (y == 1)).sum())
    tn = int((~pred & (y == 0)).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f, {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def main() -> None:
    bundle = joblib.load(MODELS / "model.joblib")
    model, features, thr = bundle["model"], bundle["features"], bundle["threshold"]

    feats = pd.read_parquet(SPLITS / "features.parquet")
    labels = pd.read_parquet(RAW / "labels.parquet")
    assign = pd.read_parquet(SPLITS / "assignment.parquet")
    df = feats.merge(assign, on="txn_id").merge(labels, on="txn_id")
    test = df[df["split"] == "test"].reset_index(drop=True)

    # Prove we scored the same test set the split manifest pinned.
    digest = hashlib.sha256(
        "\n".join(sorted(test["txn_id"].tolist())).encode()
    ).hexdigest()
    manifest = json.loads((REPORTS / "split_manifest.json").read_text())
    pinned = manifest["test_set_sha256"]

    X = test[features]
    y = test["is_fraud"].to_numpy()
    scores = model.predict_proba(X)[:, 1]

    L: list[str] = []

    def w(line: str = "") -> None:
        L.append(line)
        # The report file is UTF-8; the Windows console is cp1252 and chokes on dashes
        # and arrows. Degrade the console echo rather than lose the report.
        try:
            print(line)
        except UnicodeEncodeError:
            print(line.encode("ascii", "replace").decode("ascii"))

    w("# Held-Out Test Report")
    w()
    w("Scored **once**, with the model and threshold fixed by `train.py` on validation. "
      "Nothing below was tuned against these numbers.")
    w()
    match = "MATCHES" if digest == pinned else "DOES NOT MATCH"
    w(f"- test set SHA-256: `{digest[:16]}...` — **{match}** the pinned split manifest")
    w(f"- transactions: {len(test):,}, fraud: {int(y.sum())} ({y.mean():.2%}), "
      f"episodes: {test.loc[test['is_fraud'] == 1, 'episode_id'].nunique()}")
    w(f"- operating threshold: **{thr:.4f}** (chosen on validation, unchanged)")
    w(f"- model: XGBoost {bundle['params']}, scale_pos_weight={bundle['scale_pos_weight']:.1f}")
    w()

    # ---------------- headline ----------------
    p, r, f, cm = prf_at(scores, y, thr)
    pr_auc = average_precision_score(y, scores)
    roc_auc = roc_auc_score(y, scores)

    w("## Headline")
    w()
    w("| metric | value |")
    w("|---|---|")
    w(f"| Precision | **{p:.1%}** |")
    w(f"| Recall | **{r:.1%}** |")
    w(f"| F1 | **{f:.3f}** |")
    w(f"| PR-AUC | **{pr_auc:.3f}** |")
    w(f"| ROC-AUC | {roc_auc:.3f} |")
    w()
    w("PR-AUC is the metric to read at 2% prevalence. ROC-AUC is shown because it is "
      "conventional, but it flatters every model on imbalanced data and should not be "
      "the number anyone quotes.")
    w()

    w("## Confusion matrix (raw counts)")
    w()
    w("| | predicted legit | predicted fraud |")
    w("|---|---|---|")
    w(f"| **actually legit** | {cm['tn']:,} | {cm['fp']:,} |")
    w(f"| **actually fraud** | {cm['fn']:,} | {cm['tp']:,} |")
    w()
    w(f"{cm['fp']:,} false positives out of {cm['tn'] + cm['fp']:,} legitimate transactions "
      f"= a **{cm['fp'] / (cm['tn'] + cm['fp']):.2%} false-positive rate**. Phase 4 converts "
      "that into rupees.")
    w()

    # ---------------- baselines, same test set ----------------
    w("## Versus baselines")
    w()
    rule = ((test["txn_count_15m"] >= 4) & (test["device_age_secs"] < 3600)).to_numpy()
    rp, rr, rf, _ = prf_at(rule.astype(float), y, 0.5)
    w("| model | precision | recall | F1 |")
    w("|---|---|---|---|")
    w("| always-legitimate | 0.0% | 0.0% | 0.000 |")
    w(f"| rule: burst AND new device | {rp:.1%} | {rr:.1%} | {rf:.3f} |")
    w(f"| **XGBoost** | **{p:.1%}** | **{r:.1%}** | **{f:.3f}** |")
    w()

    # ---------------- episode level ----------------
    fraud = test[test["is_fraud"] == 1].copy()
    fraud["flagged"] = scores[test["is_fraud"] == 1] >= thr
    caught = fraud.groupby("episode_id")["flagged"].any()
    w("## Episode-level recall")
    w()
    w(f"**{int(caught.sum())} of {len(caught)} fraud episodes** had at least one transaction "
      f"flagged ({caught.mean():.1%}). This is the number a risk team actually cares about: "
      "catching an episode on its third transaction still stops the remaining ones.")
    w()

    # ---------------- per archetype: where does it fail? ----------------
    w("## Recall by attack archetype")
    w()
    w("| archetype | fraud txns | txn recall | episodes | episodes caught |")
    w("|---|---|---|---|---|")
    for style, g in fraud.groupby("archetype"):
        eps = g.groupby("episode_id")["flagged"].any()
        w(f"| `{style}` | {len(g)} | {g['flagged'].mean():.1%} | {len(eps)} | "
          f"{int(eps.sum())}/{len(eps)} |")
    w()
    w("Reported because an average hides which attacks get through. A detector that is "
      "excellent on classic bursts and blind to session hijacks is not an 90%-recall "
      "detector, it is two different detectors.")
    w()

    # ---------------- threshold sweep, so Phase 4 never reopens this set ----------------
    w("## Threshold sensitivity")
    w()
    w("Provided in this single pass so the cost analysis in Phase 4 can pick an operating "
      "point without opening the test set a second time.")
    w()
    w("| threshold | precision | recall | F1 | flagged | false positives |")
    w("|---|---|---|---|---|---|")
    for t in [0.05, 0.10, thr, 0.20, 0.30, 0.50, 0.70, 0.90]:
        tp_, tr_, tf_, tcm = prf_at(scores, y, t)
        star = " (chosen)" if abs(t - thr) < 1e-9 else ""
        w(f"| {t:.4f}{star} | {tp_:.1%} | {tr_:.1%} | {tf_:.3f} | "
          f"{tcm['tp'] + tcm['fp']:,} | {tcm['fp']:,} |")
    w()

    (REPORTS / "test_report.md").write_text("\n".join(L), encoding="utf-8")
    np.save(MODELS / "test_scores.npy", scores)
    test[["txn_id"]].assign(score=scores).to_parquet(MODELS / "test_scores.parquet", index=False)
    print(f"\n[written] {REPORTS / 'test_report.md'}")


if __name__ == "__main__":
    main()
