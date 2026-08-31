"""Causal feature engineering, per SPEC.md section 3.

Every feature answers a question a real authorisation system could answer in the
milliseconds before approving a transaction, using only that account's past. The rules:

  * Window counts include the current transaction -- production genuinely knows that this
    is the 5th attempt in 4 minutes.
  * Historical baselines (median, mean, rank) use STRICTLY prior transactions. Letting the
    current amount into its own baseline would dampen exactly the deviation we want to see.
  * No history means NaN, not a filled-in zero. XGBoost splits on missing natively, and a
    fabricated zero would read as "perfectly typical" for a brand-new account.

Features are built over the full time-ordered stream and only then filtered by split,
because a test transaction's trailing window legitimately reaches back into earlier days.
See splits.py for why that is not leakage.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import HIGH_RISK_CATEGORIES

RAW = Path("data/raw")
SPLITS = Path("data/splits")

WINDOWS = {"5m": 300, "15m": 900, "60m": 3600}

# Channel carries no injected signal -- fraud draws it from the same distribution as
# legitimate traffic. It is kept deliberately: near-zero SHAP importance for these columns
# is a cheap sanity check that the model is not inventing structure that is not there.
CHANNELS_ONEHOT = ["card", "upi", "netbanking", "wallet"]


def _circular_hour_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hours are cyclic: 23:00 and 01:00 are two apart, not twenty-two."""
    d = np.abs(a - b) % 24.0
    return np.minimum(d, 24.0 - d)


def _account_features(g: pd.DataFrame) -> pd.DataFrame:
    """All per-account causal features for one account, in time order."""
    n = len(g)
    ts = g["timestamp"].to_numpy("datetime64[s]").astype("int64")
    amt = g["amount"].to_numpy(float)
    log_amt = np.log1p(amt)
    hour = g["hour"].to_numpy(float)
    dev = g["device_id"].to_numpy()
    city = g["city"].to_numpy()
    cat = g["merchant_category"].to_numpy()

    out: dict[str, np.ndarray] = {}

    # ---- velocity: how fast is this account moving right now ----
    for name, secs in WINDOWS.items():
        left = np.searchsorted(ts, ts - secs, side="left")
        idx = np.arange(n)
        out[f"txn_count_{name}"] = (idx - left + 1).astype(float)
        csum = np.concatenate([[0.0], np.cumsum(amt)])
        out[f"amount_sum_{name}"] = csum[idx + 1] - csum[left]

    gaps = np.diff(ts, prepend=np.nan).astype(float)
    out["secs_since_prev"] = gaps
    # Mean of the previous five inter-transaction gaps: a burst collapses this.
    out["mean_gap_prev5"] = pd.Series(gaps).rolling(5, min_periods=2).mean().to_numpy()

    # ---- amount, relative to this account's own history ----
    out["log_amount"] = log_amt
    ratio = np.full(n, np.nan)
    zscore = np.full(n, np.nan)
    pct_rank = np.full(n, np.nan)
    for i in range(1, n):
        prior = amt[:i]
        med = np.median(prior)
        ratio[i] = amt[i] / med if med > 0 else np.nan
        pct_rank[i] = np.mean(prior < amt[i])
        if i >= 3:
            lp = log_amt[:i]
            sd = lp.std(ddof=1)
            zscore[i] = (log_amt[i] - lp.mean()) / sd if sd > 1e-9 else np.nan
    out["amt_ratio_to_prior_median"] = ratio
    out["amt_log_zscore"] = zscore
    out["amt_pct_rank_prior"] = pct_rank

    # The card-testing ramp: small probes, then escalation. Comparing against the max seen
    # earlier in the same 15-minute window is what makes the ramp visible as a shape rather
    # than as one large transaction.
    esc = np.full(n, np.nan)
    left15 = np.searchsorted(ts, ts - WINDOWS["15m"], side="left")
    for i in range(n):
        if left15[i] < i:
            prior_max = amt[left15[i]:i].max()
            esc[i] = amt[i] / prior_max if prior_max > 0 else np.nan
    out["amt_ratio_to_window_max"] = esc

    # ---- device, city, category: novelty measured against this account's own past ----
    out["device_txns_prior"] = _prior_count(dev)
    out["device_age_secs"] = _age_since_first_seen(dev, ts)
    out["is_new_device"] = (out["device_txns_prior"] == 0).astype(float)
    out["distinct_devices_prior"] = _prior_distinct(dev)

    out["city_txns_prior"] = _prior_count(city)
    out["is_new_city"] = (out["city_txns_prior"] == 0).astype(float)
    changed = np.zeros(n)
    changed[1:] = (city[1:] != city[:-1]).astype(float)
    changed[0] = np.nan
    out["city_changed_from_prev"] = changed

    cat_prior = _prior_count(cat)
    out["cat_txns_prior"] = cat_prior
    out["is_new_category"] = (cat_prior == 0).astype(float)
    hist_depth = np.arange(n, dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        out["cat_share_prior"] = np.where(hist_depth > 0, cat_prior / hist_depth, np.nan)
    out["is_high_risk_category"] = np.isin(cat, HIGH_RISK_CATEGORIES).astype(float)

    # ---- time of day, relative to when this account normally transacts ----
    out["hour"] = hour
    out["is_night"] = (hour < 5).astype(float)
    mean_hour = np.full(n, np.nan)
    # Circular mean of prior hours, so a 23:00-and-01:00 account averages to midnight.
    sin_c = np.cumsum(np.sin(hour * np.pi / 12.0))
    cos_c = np.cumsum(np.cos(hour * np.pi / 12.0))
    for i in range(1, n):
        mean_hour[i] = np.arctan2(sin_c[i - 1] / i, cos_c[i - 1] / i) * 12.0 / np.pi % 24.0
    out["hour_dist_from_acct_norm"] = _circular_hour_distance(hour, mean_hour)

    # ---- how much history do we actually have to judge against ----
    out["acct_txns_prior"] = hist_depth
    out["acct_age_days"] = np.concatenate([[np.nan], (ts[1:] - ts[0]) / 86400.0])

    return pd.DataFrame(out, index=g.index)


def _prior_count(values: np.ndarray) -> np.ndarray:
    """For each position, how many times this exact value appeared strictly earlier."""
    seen: dict = {}
    out = np.zeros(len(values))
    for i, v in enumerate(values):
        out[i] = seen.get(v, 0)
        seen[v] = out[i] + 1
    return out


def _prior_distinct(values: np.ndarray) -> np.ndarray:
    """Number of distinct values seen strictly before each position."""
    seen: set = set()
    out = np.zeros(len(values))
    for i, v in enumerate(values):
        out[i] = len(seen)
        seen.add(v)
    return out


def _age_since_first_seen(values: np.ndarray, ts: np.ndarray) -> np.ndarray:
    """Seconds since this value was first seen on the account; 0 the first time.

    This is the device-novelty definition the Phase 1 audit forced on us: measuring age
    rather than first-appearance means the signal stays alive for a whole burst instead of
    firing once and going quiet.
    """
    first: dict = {}
    out = np.zeros(len(values))
    for i, v in enumerate(values):
        if v not in first:
            first[v] = ts[i]
        out[i] = float(ts[i] - first[v])
    return out


def build(txns: pd.DataFrame) -> pd.DataFrame:
    """Build the full causal feature frame for a time-ordered transaction stream."""
    df = txns.sort_values(["timestamp", "txn_id"], kind="mergesort").reset_index(drop=True)
    df["hour"] = df["timestamp"].dt.hour + df["timestamp"].dt.minute / 60.0

    parts = [
        _account_features(g)
        for _, g in df.groupby("account_id", sort=False)
    ]
    feats = pd.concat(parts).sort_index()

    for ch in CHANNELS_ONEHOT:
        feats[f"channel_{ch}"] = (df["channel"] == ch).astype(float)

    feats.insert(0, "txn_id", df["txn_id"].to_numpy())
    return feats


FEATURE_COLUMNS = None  # populated on first build; see main()


def main() -> None:
    txns = pd.read_parquet(RAW / "transactions.parquet")
    feats = build(txns)

    SPLITS.mkdir(parents=True, exist_ok=True)
    feats.to_parquet(SPLITS / "features.parquet", index=False)

    cols = [c for c in feats.columns if c != "txn_id"]
    print(f"rows     : {len(feats):,}")
    print(f"features : {len(cols)}")
    print(f"missing  : {feats[cols].isna().sum().sum():,} cells "
          f"({feats[cols].isna().to_numpy().mean():.2%}) -- expected, early history is unknown")
    print(f"[written] {SPLITS / 'features.parquet'}")
    print()
    for c in cols:
        print(f"  {c}")


if __name__ == "__main__":
    main()
