"""Model selection and threshold tuning. READS TRAIN AND VALIDATION ONLY.

This script cannot open the held-out test set: `load()` drops those rows before returning,
so there is no code path here that could peek. Test scoring lives in evaluate.py, runs
once, and consumes only the artifacts this script writes.

Everything decided here -- hyperparameters, the operating threshold, whether to use SMOTE
-- is decided on validation, per SPEC.md section 3.

Writes:
    models/model.joblib          the fitted model, feature order, and chosen threshold
    reports/validation_report.md what was tried and why the winner won
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

RAW = Path("data/raw")
SPLITS = Path("data/splits")
MODELS = Path("models")
REPORTS = Path("reports")

SEED = 7


def load() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Train and validation only. Test rows are dropped here and never returned."""
    feats = pd.read_parquet(SPLITS / "features.parquet")
    labels = pd.read_parquet(RAW / "labels.parquet")
    assign = pd.read_parquet(SPLITS / "assignment.parquet")

    df = feats.merge(assign, on="txn_id").merge(labels[["txn_id", "is_fraud"]], on="txn_id")
    df = df[df["split"] != "test"]                      # <- the test set leaves the building

    drop = ["txn_id", "split", "is_fraud"]
    tr = df[df["split"] == "train"]
    va = df[df["split"] == "val"]
    return (tr.drop(columns=drop), tr["is_fraud"],
            va.drop(columns=drop), va["is_fraud"])


def prf_at(scores: np.ndarray, y: np.ndarray, thr: float) -> tuple[float, float, float]:
    pred = scores >= thr
    tp = int((pred & (y == 1)).sum())
    fp = int((pred & (y == 0)).sum())
    fn = int((~pred & (y == 1)).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def best_f1_threshold(scores: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Sweep candidate thresholds and take the F1 maximiser. Validation only."""
    candidates = np.unique(np.quantile(scores, np.linspace(0.90, 0.9999, 400)))
    best_thr, best_f1 = 0.5, -1.0
    for thr in candidates:
        _, _, f = prf_at(scores, y, thr)
        if f > best_f1:
            best_thr, best_f1 = float(thr), f
    return best_thr, best_f1


def rule_burst_and_new_device(X: pd.DataFrame) -> np.ndarray:
    """The hand-written baseline the model has to beat: a burst on a device this account
    has only just started using. Expressed in the same features the model sees."""
    return ((X["txn_count_15m"] >= 4) & (X["device_age_secs"] < 3600)).to_numpy()


def main() -> None:
    Xtr, ytr, Xva, yva = load()
    MODELS.mkdir(exist_ok=True)
    REPORTS.mkdir(exist_ok=True)
    features = list(Xtr.columns)
    L: list[str] = []

    def w(line: str = "") -> None:
        L.append(line)
        print(line)

    w("# Validation Report")
    w()
    w("Model selection and threshold tuning. **The held-out test set is not read by this "
      "script** — `train.py::load()` drops those rows before returning.")
    w()
    w(f"- train: {len(Xtr):,} transactions, {int(ytr.sum())} fraud ({ytr.mean():.2%})")
    w(f"- validation: {len(Xva):,} transactions, {int(yva.sum())} fraud ({yva.mean():.2%})")
    w(f"- features: {len(features)}")
    w()

    pos_weight = float((ytr == 0).sum() / (ytr == 1).sum())
    w(f"Class imbalance handled with `scale_pos_weight = {pos_weight:.1f}` "
      f"(negatives per positive in train).")
    w()

    # ---------------- baselines ----------------
    w("## Baselines")
    w()
    w("| model | precision | recall | F1 | PR-AUC | ROC-AUC |")
    w("|---|---|---|---|---|---|")

    w(f"| always-legitimate | 0.0% | 0.0% | 0.000 | {yva.mean():.3f} | 0.500 |")

    rule = rule_burst_and_new_device(Xva)
    p, r, f = prf_at(rule.astype(float), yva.to_numpy(), 0.5)
    w(f"| rule: burst AND new device | {p:.1%} | {r:.1%} | {f:.3f} | — | — |")

    logreg = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED)),
    ])
    logreg.fit(Xtr, ytr)
    lr_scores = logreg.predict_proba(Xva)[:, 1]
    lr_thr, _ = best_f1_threshold(lr_scores, yva.to_numpy())
    p, r, f = prf_at(lr_scores, yva.to_numpy(), lr_thr)
    w(f"| logistic regression | {p:.1%} | {r:.1%} | {f:.3f} | "
      f"{average_precision_score(yva, lr_scores):.3f} | {roc_auc_score(yva, lr_scores):.3f} |")
    w()

    # ---------------- XGBoost grid, scored on validation PR-AUC ----------------
    w("## XGBoost selection")
    w()
    w("Grid scored by **validation PR-AUC** — the honest ranking metric at 2% prevalence, "
      "where ROC-AUC flatters everything.")
    w()
    grid = list(itertools.product([3, 4, 6], [200, 400], [0.05, 0.10]))
    results = []
    for depth, n_est, lr_rate in grid:
        m = XGBClassifier(
            max_depth=depth, n_estimators=n_est, learning_rate=lr_rate,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=pos_weight, eval_metric="aucpr",
            random_state=SEED, n_jobs=4, tree_method="hist",
        )
        m.fit(Xtr, ytr)
        s = m.predict_proba(Xva)[:, 1]
        results.append({
            "max_depth": depth, "n_estimators": n_est, "learning_rate": lr_rate,
            "pr_auc": average_precision_score(yva, s), "roc_auc": roc_auc_score(yva, s),
            "model": m, "scores": s,
        })

    results.sort(key=lambda d: -d["pr_auc"])
    w("| max_depth | n_estimators | learning_rate | PR-AUC | ROC-AUC |")
    w("|---|---|---|---|---|")
    for d in results:
        w(f"| {d['max_depth']} | {d['n_estimators']} | {d['learning_rate']} | "
          f"{d['pr_auc']:.4f} | {d['roc_auc']:.4f} |")
    w()

    best = results[0]
    w(f"Winner: `max_depth={best['max_depth']}, n_estimators={best['n_estimators']}, "
      f"learning_rate={best['learning_rate']}` at validation PR-AUC **{best['pr_auc']:.4f}**.")
    w()

    # ---------------- SMOTE ablation: tried, and reported either way ----------------
    w("## SMOTE ablation")
    w()
    try:
        from imblearn.over_sampling import SMOTE

        imp = SimpleImputer(strategy="median")
        Xtr_i = imp.fit_transform(Xtr)
        Xsm, ysm = SMOTE(random_state=SEED, k_neighbors=5).fit_resample(Xtr_i, ytr)
        m_sm = XGBClassifier(
            max_depth=best["max_depth"], n_estimators=best["n_estimators"],
            learning_rate=best["learning_rate"], subsample=0.8, colsample_bytree=0.8,
            eval_metric="aucpr", random_state=SEED, n_jobs=4, tree_method="hist",
        )
        m_sm.fit(Xsm, ysm)
        s_sm = m_sm.predict_proba(imp.transform(Xva))[:, 1]
        pr_sm = average_precision_score(yva, s_sm)
        w(f"| approach | validation PR-AUC |")
        w("|---|---|")
        w(f"| class weighting (`scale_pos_weight`) | **{best['pr_auc']:.4f}** |")
        w(f"| SMOTE oversampling | {pr_sm:.4f} |")
        w()
        verdict = "worse" if pr_sm < best["pr_auc"] else "better"
        w(f"SMOTE scores {verdict} here. We use class weighting regardless of the margin, "
          "for a reason that matters more than the number: SMOTE interpolates between fraud "
          "rows from *different accounts*, synthesising transactions whose velocity counts "
          "come from one account and whose device age comes from another. Those rows cannot "
          "occur in reality, and training on them undermines the causal guarantee Phase 2 "
          "was built to provide.")
    except Exception as exc:  # pragma: no cover
        w(f"SMOTE ablation could not be run: `{exc}`")
    w()

    # ---------------- operating threshold, chosen on validation ----------------
    thr, f1 = best_f1_threshold(best["scores"], yva.to_numpy())
    p, r, _ = prf_at(best["scores"], yva.to_numpy(), thr)
    w("## Operating threshold")
    w()
    w(f"Chosen on validation by maximising F1: **{thr:.4f}** "
      f"(validation precision {p:.1%}, recall {r:.1%}, F1 {f1:.3f}).")
    w()
    w("This is the headline operating point. Phase 4's cost analysis may recommend a "
      "different one; if so, the change will be made on validation and stated explicitly, "
      "not chosen by looking at test.")
    w()

    joblib.dump(
        {"model": best["model"], "features": features, "threshold": thr,
         "params": {k: best[k] for k in ("max_depth", "n_estimators", "learning_rate")},
         "scale_pos_weight": pos_weight, "seed": SEED},
        MODELS / "model.joblib",
    )
    (REPORTS / "validation_report.md").write_text("\n".join(L), encoding="utf-8")
    print(f"\n[written] {MODELS / 'model.joblib'}, {REPORTS / 'validation_report.md'}")


if __name__ == "__main__":
    main()
