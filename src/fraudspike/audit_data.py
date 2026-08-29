"""Sanity audit of the generated data. Run after every regeneration.

The point of this script is adversarial: it tries to break our own dataset by checking
whether any single marker separates fraud from legitimate traffic. If a one-line rule
already scores high precision, the confounders in SPEC.md section 2 are too weak and
the modelling that follows would be meaningless.

Writes reports/data_audit.md.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import HIGH_RISK_CATEGORIES, GenConfig

RAW = Path("data/raw")
REPORTS = Path("reports")


def load() -> pd.DataFrame:
    txns = pd.read_parquet(RAW / "transactions.parquet")
    labels = pd.read_parquet(RAW / "labels.parquet")
    return txns.merge(labels, on="txn_id", validate="one_to_one")


def burst_flag(df: pd.DataFrame, n: int = 4, minutes: int = 10) -> pd.Series:
    """The hand-written rule any risk team would write first: n+ transactions on one
    account within the trailing window. Strictly backward-looking, like a real rule."""
    out = np.zeros(len(df), dtype=bool)
    window = pd.Timedelta(minutes=minutes)
    for _, idx in df.groupby("account_id", sort=False).indices.items():
        ts = df["timestamp"].values[idx]
        # For each transaction, how many of this account's transactions fall in (t-w, t]?
        left = np.searchsorted(ts, ts - np.timedelta64(window), side="left")
        counts = np.arange(len(ts)) - left + 1
        out[idx] = counts >= n
    return pd.Series(out, index=df.index)


def prf(pred: pd.Series, truth: pd.Series) -> tuple[float, float, float]:
    tp = int((pred & (truth == 1)).sum())
    fp = int((pred & (truth == 0)).sum())
    fn = int((~pred & (truth == 1)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return prec, rec, f1


def main() -> None:
    cfg = GenConfig()
    df = load().sort_values("timestamp").reset_index(drop=True)
    REPORTS.mkdir(exist_ok=True)
    L = []

    def w(line: str = "") -> None:
        L.append(line)
        print(line)

    n, nf = len(df), int(df["is_fraud"].sum())
    w("# Data Audit")
    w()
    w(f"- transactions: **{n:,}**, accounts: **{df['account_id'].nunique():,}**, "
      f"days: **{int(df['day'].max()) + 1}**")
    w(f"- fraud transactions: **{nf:,}** ({nf / n:.2%}), episodes: **{df['episode_id'].nunique()}**")
    w()

    # --- split occupancy: does the held-out test set have enough positives to report on? ---
    w("## Temporal split occupancy")
    w()
    w("| split | days | transactions | fraud txns | fraud rate | episodes |")
    w("|---|---|---|---|---|---|")
    bounds = {
        "train": (0, cfg.train_end_day),
        "val": (cfg.train_end_day, cfg.val_end_day),
        "test": (cfg.val_end_day, int(df["day"].max()) + 1),
    }
    for name, (lo, hi) in bounds.items():
        s = df[(df["day"] >= lo) & (df["day"] < hi)]
        eps = s.loc[s["is_fraud"] == 1, "episode_id"].nunique()
        w(f"| {name} | [{lo}, {hi}) | {len(s):,} | {int(s['is_fraud'].sum()):,} | "
          f"{s['is_fraud'].mean():.2%} | {eps} |")
    w()

    # --- the adversarial part: can one marker do the job on its own? ---
    w("## Single-marker separability")
    w()
    w("Each row is a rule a risk team could write without any model. High precision here "
      "would mean the dataset is trivially separable and the model adds nothing.")
    w()
    w("| marker | flagged | precision | recall | F1 |")
    w("|---|---|---|---|---|")

    truth = df["is_fraud"]
    markers = {
        "velocity: 4+ txns in 10 min": burst_flag(df, 4, 10),
        "velocity: 5+ txns in 5 min": burst_flag(df, 5, 5),
        "device new to account (<1h)": _first_seen_device(df),
        "hour in 00:00-05:00": df["timestamp"].dt.hour < 5,
        "high-risk category": df["merchant_category"].isin(HIGH_RISK_CATEGORIES),
        "city != account's usual": _foreign_city(df),
        "amount > 3x account median": _amount_spike(df),
    }
    for name, pred in markers.items():
        p, r, f = prf(pred.astype(bool), truth)
        w(f"| {name} | {int(pred.sum()):,} | {p:.1%} | {r:.1%} | {f:.3f} |")
    w()

    # --- the conjunction, which is what the model is meant to learn ---
    conj = (markers["velocity: 4+ txns in 10 min"].astype(bool)
            & markers["device new to account (<1h)"].astype(bool))
    p, r, f = prf(conj, truth)
    w(f"**Burst AND new-device combined:** flagged {int(conj.sum()):,}, "
      f"precision {p:.1%}, recall {r:.1%}, F1 {f:.3f}")
    w()

    # --- confounder health: fraud must not own the burst signal ---
    w("## Confounder health")
    w()
    burst = markers["velocity: 4+ txns in 10 min"].astype(bool)
    legit_burst = int((burst & (truth == 0)).sum())
    fraud_burst = int((burst & (truth == 1)).sum())
    w(f"- transactions inside a 4-in-10min burst: **{legit_burst:,} legitimate** vs "
      f"**{fraud_burst:,} fraudulent** "
      f"({legit_burst / max(1, legit_burst + fraud_burst):.0%} of bursts are legitimate)")

    newdev = markers["device new to account (<1h)"].astype(bool)
    w(f"- first-time-seen devices: **{int((newdev & (truth == 0)).sum()):,} legitimate** vs "
      f"**{int((newdev & (truth == 1)).sum()):,} fraudulent**")

    night = df["timestamp"].dt.hour < 5
    w(f"- transactions in 00:00-05:00: **{int((night & (truth == 0)).sum()):,} legitimate** vs "
      f"**{int((night & (truth == 1)).sum()):,} fraudulent**")

    legit_amt = df.loc[truth == 0, "amount"]
    fraud_amt = df.loc[truth == 1, "amount"]
    w(f"- amount overlap: legitimate p50 Rs {legit_amt.median():,.0f} / p99 "
      f"Rs {legit_amt.quantile(0.99):,.0f}; fraud p50 Rs {fraud_amt.median():,.0f} / p99 "
      f"Rs {fraud_amt.quantile(0.99):,.0f}")
    w()

    (REPORTS / "data_audit.md").write_text("\n".join(L), encoding="utf-8")
    print(f"\n[written] {REPORTS / 'data_audit.md'}")


def _first_seen_device(df: pd.DataFrame) -> pd.Series:
    """Device is NEW to this account, measured as age-on-account rather than
    first-appearance. First-appearance only ever fires on an episode's opening
    transaction -- which is never yet inside a burst -- so it can never combine with a
    velocity signal. Age-on-account fires for the whole episode, which is the point."""
    first_seen = df.groupby(["account_id", "device_id"])["timestamp"].transform("min")
    acct_first = df.groupby("account_id")["timestamp"].transform("min")
    young = (df["timestamp"] - first_seen) < pd.Timedelta(hours=1)
    has_history = first_seen > acct_first          # not the account's original device
    return young & has_history


def _foreign_city(df: pd.DataFrame) -> pd.Series:
    usual = df.groupby("account_id")["city"].agg(lambda s: s.mode().iat[0])
    return df["city"].values != usual.reindex(df["account_id"]).values


def _amount_spike(df: pd.DataFrame) -> pd.Series:
    med = df.groupby("account_id")["amount"].transform("median")
    return df["amount"] > 3 * med


if __name__ == "__main__":
    main()
