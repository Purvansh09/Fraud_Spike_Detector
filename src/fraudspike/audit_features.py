"""Adversarial audit of the engineered features. TRAIN SPLIT ONLY.

Two questions, both asked before any model exists:

  1. Is any single feature suspiciously good? A lone feature at AUC > 0.98 almost always
     means a leak rather than a discovery. This is the check that would have caught the
     city leak in Phase 1 had it been a feature rather than a raw column.
  2. Is any feature degenerate -- constant, or missing so often it cannot carry signal?

The validation and test splits are deliberately not read here. Exploration happens on
train; validation is for tuning; test is opened once, in Phase 3.

Writes reports/feature_audit.md.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

RAW = Path("data/raw")
SPLITS = Path("data/splits")
REPORTS = Path("reports")

SUSPICIOUS_AUC = 0.98


def load_train() -> tuple[pd.DataFrame, pd.Series]:
    feats = pd.read_parquet(SPLITS / "features.parquet")
    labels = pd.read_parquet(RAW / "labels.parquet")
    assign = pd.read_parquet(SPLITS / "assignment.parquet")

    df = feats.merge(assign, on="txn_id").merge(labels[["txn_id", "is_fraud"]], on="txn_id")
    train = df[df["split"] == "train"]
    y = train["is_fraud"]
    X = train.drop(columns=["txn_id", "split", "is_fraud"])
    return X, y


def main() -> None:
    X, y = load_train()
    REPORTS.mkdir(exist_ok=True)
    L: list[str] = []

    def w(line: str = "") -> None:
        L.append(line)
        print(line)

    w("# Feature Audit")
    w()
    w(f"Train split only: **{len(X):,}** transactions, **{int(y.sum())}** fraud "
      f"({y.mean():.2%}). Validation and test are untouched.")
    w()

    rows = []
    for col in X.columns:
        v = X[col]
        miss = float(v.isna().mean())
        if v.nunique(dropna=True) <= 1:
            rows.append((col, np.nan, miss, "CONSTANT"))
            continue
        # Score with the median where missing, so AUC reflects the feature rather than
        # the missingness pattern; missingness is reported separately alongside.
        filled = v.fillna(v.median())
        auc = roc_auc_score(y, filled)
        # A feature that ranks fraud *low* is just as informative as one that ranks it high.
        directed = max(auc, 1 - auc)
        note = "SUSPICIOUS" if directed > SUSPICIOUS_AUC else ""
        rows.append((col, directed, miss, note))

    rows.sort(key=lambda r: (-(r[1] if not np.isnan(r[1]) else -1)))

    w("## Single-feature discriminative power")
    w()
    w(f"AUC is direction-corrected (max of AUC, 1-AUC). Anything above "
      f"**{SUSPICIOUS_AUC}** is flagged for investigation as a probable leak.")
    w()
    w("| feature | AUC alone | missing | flag |")
    w("|---|---|---|---|")
    for col, auc, miss, note in rows:
        a = "n/a" if np.isnan(auc) else f"{auc:.3f}"
        w(f"| `{col}` | {a} | {miss:.1%} | {note} |")
    w()

    flagged = [r for r in rows if r[3]]
    if flagged:
        w(f"> **{len(flagged)} feature(s) flagged.** Investigate before training.")
    else:
        top = rows[0]
        w(f"> No feature exceeds the {SUSPICIOUS_AUC} threshold. The strongest alone is "
          f"`{top[0]}` at AUC {top[1]:.3f} — informative, not decisive, which is what we want.")
    w()

    # Redundancy: near-duplicate features inflate apparent importance and split SHAP
    # attribution between twins, which muddies the explanations in Phase 5.
    w("## Highly correlated pairs (|r| > 0.95)")
    w()
    corr = X.corr(numeric_only=True).abs()
    pairs = [
        (a, b, corr.loc[a, b])
        for i, a in enumerate(corr.columns)
        for b in corr.columns[i + 1:]
        if corr.loc[a, b] > 0.95
    ]
    if pairs:
        w("| feature A | feature B | r |")
        w("|---|---|---|")
        for a, b, r in sorted(pairs, key=lambda p: -p[2]):
            w(f"| `{a}` | `{b}` | {r:.3f} |")
        w()
        w("Kept for now — XGBoost tolerates collinearity — but SHAP attribution will split "
          "between twins, so this table is the reference when explanations look diluted.")
    else:
        w("None.")
    w()

    (REPORTS / "feature_audit.md").write_text("\n".join(L), encoding="utf-8")
    print(f"\n[written] {REPORTS / 'feature_audit.md'}")


if __name__ == "__main__":
    main()
