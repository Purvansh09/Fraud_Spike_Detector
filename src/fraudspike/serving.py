"""Live scoring: turn one incoming transaction into a decision.

The hard part of serving this model is not the HTTP layer, it is that every feature is a
function of the account's history. A scorer that recomputes those features with even
slightly different code than `features.py` used at training time will quietly drift --
the classic train/serve skew, and the usual way a good offline model becomes a bad
production one.

This module therefore does NOT reimplement feature logic. It reconstructs the account's
transaction history, appends the incoming transaction, and calls the exact same
`features.build()` used to train, keeping the final row. `tests/test_serving.py` asserts
that the result is bit-for-bit identical to the batch pipeline.

Decision bands (auto-clear / auto-review / auto-block) are derived on validation, never
on test.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .features import build

RAW = Path("data/raw")
MODELS = Path("models")

# Columns a payment processor genuinely has at authorisation time.
TXN_FIELDS = [
    "txn_id", "account_id", "timestamp", "amount",
    "merchant_id", "merchant_category", "device_id", "city", "channel",
]

# Expanding features (median, mean, percentile rank) look at an account's whole past, so
# history cannot be truncated to the widest rolling window. This cap is far above any
# account's real volume here and exists only to bound worst-case latency.
MAX_HISTORY = 1000


@dataclass
class Decision:
    txn_id: str
    account_id: str
    score: float
    action: str            # ALLOW | REVIEW | BLOCK
    threshold_block: float
    threshold_review: float


class HistoryStore:
    """Per-account transaction history, kept in time order.

    Backed by a plain dict here. In production this is the interesting component -- it
    would be Redis or a feature store, and its freshness would be the main operational
    risk, since a stale history under-counts velocity exactly when a burst is happening.
    """

    def __init__(self) -> None:
        self._by_account: dict[str, pd.DataFrame] = {}

    @classmethod
    def from_parquet(cls, path: Path = RAW / "transactions.parquet") -> "HistoryStore":
        store = cls()
        df = pd.read_parquet(path)[TXN_FIELDS]
        for acct, g in df.groupby("account_id", sort=False):
            store._by_account[acct] = g.sort_values("timestamp").reset_index(drop=True)
        return store

    def history_before(self, account_id: str, ts: pd.Timestamp, exclude_txn: str | None = None):
        g = self._by_account.get(account_id)
        if g is None or g.empty:
            return pd.DataFrame(columns=TXN_FIELDS)
        h = g[g["timestamp"] < ts]
        if exclude_txn is not None:
            h = h[h["txn_id"] != exclude_txn]
        return h.tail(MAX_HISTORY)

    def append(self, txn: dict) -> None:
        """Record a transaction so it becomes history for the next one on that account."""
        acct = txn["account_id"]
        row = pd.DataFrame([{k: txn[k] for k in TXN_FIELDS}])
        prev = self._by_account.get(acct)
        merged = row if prev is None else pd.concat([prev, row], ignore_index=True)
        self._by_account[acct] = merged.sort_values("timestamp").reset_index(drop=True)

    def n_accounts(self) -> int:
        return len(self._by_account)


class Scorer:
    """Model + thresholds + history, with the feature path shared with training."""

    def __init__(self, store: HistoryStore | None = None) -> None:
        bundle = joblib.load(MODELS / "model.joblib")
        self.model = bundle["model"]
        self.features: list[str] = bundle["features"]
        self.threshold_block: float = float(bundle["threshold"])
        # Below the block line but above this, a human should look. Derived on validation
        # in `calibrate_review_band` and stored with the model; falls back to a fraction
        # of the block threshold if an older bundle is loaded.
        self.threshold_review: float = float(bundle.get("threshold_review", self.threshold_block / 5))
        self.store = store if store is not None else HistoryStore.from_parquet()

    def feature_row(self, txn: dict) -> pd.DataFrame:
        """Causal features for one incoming transaction, via the training code path."""
        ts = pd.Timestamp(txn["timestamp"])
        hist = self.store.history_before(txn["account_id"], ts, exclude_txn=txn.get("txn_id"))
        incoming = pd.DataFrame([{k: txn.get(k) for k in TXN_FIELDS}])
        incoming["timestamp"] = pd.to_datetime(incoming["timestamp"])

        stream = pd.concat([hist, incoming], ignore_index=True)
        # An empty history frame carries object dtype, which survives the concat and makes
        # the .dt accessor in build() fail on a brand-new account. Pin the dtypes that
        # feature building depends on.
        stream["timestamp"] = pd.to_datetime(stream["timestamp"])
        stream["amount"] = stream["amount"].astype(float)
        feats = build(stream)
        # build() sorts by (timestamp, txn_id); locate our row by id rather than assuming
        # it landed last, since a same-second collision could reorder it.
        row = feats[feats["txn_id"] == txn["txn_id"]]
        if row.empty:                     # no txn_id supplied: it sorted to the end
            row = feats.tail(1)
        return row[self.features]

    def score(self, txn: dict) -> tuple[float, pd.DataFrame]:
        X = self.feature_row(txn)
        return float(self.model.predict_proba(X)[:, 1][0]), X

    def decide(self, txn: dict) -> tuple[Decision, pd.DataFrame]:
        score, X = self.score(txn)
        if score >= self.threshold_block:
            action = "BLOCK"
        elif score >= self.threshold_review:
            action = "REVIEW"
        else:
            action = "ALLOW"
        return (
            Decision(
                txn_id=txn.get("txn_id", "—"),
                account_id=txn["account_id"],
                score=score,
                action=action,
                threshold_block=self.threshold_block,
                threshold_review=self.threshold_review,
            ),
            X,
        )


def calibrate_review_band(target_recall: float = 0.95) -> float:
    """Pick the review threshold on VALIDATION as the lowest score that still catches
    `target_recall` of fraud. Transactions between here and the block line are too weak
    to decline outright but too strong to wave through.
    """
    from .costs import load_split

    bundle = joblib.load(MODELS / "model.joblib")
    val = load_split("val")
    scores = bundle["model"].predict_proba(val[bundle["features"]])[:, 1]
    fraud_scores = np.sort(scores[val["is_fraud"].to_numpy() == 1])
    if len(fraud_scores) == 0:
        return bundle["threshold"] / 5
    idx = int((1 - target_recall) * len(fraud_scores))
    return float(fraud_scores[min(idx, len(fraud_scores) - 1)])


def calibrate_block_band() -> float:
    """Block at the cost-optimal threshold from Phase 4, recomputed on VALIDATION.

    Derived rather than hardcoded so that changing the cost assumptions in `CostModel`
    automatically moves the deployed operating point, instead of leaving the service
    running against a stale number someone typed in once.
    """
    from .costs import CostModel, evaluate_threshold, load_split

    bundle = joblib.load(MODELS / "model.joblib")
    val = load_split("val")
    scores = bundle["model"].predict_proba(val[bundle["features"]])[:, 1]
    grid = np.unique(np.concatenate([
        np.geomspace(1e-4, 1e-2, 60),
        np.linspace(0.01, 0.99, 200),
    ]))
    cm = CostModel()
    rows = [evaluate_threshold(val, scores, t, cm) for t in grid]
    return float(max(rows, key=lambda d: d["net_saving_inr"])["threshold"])


def main() -> None:
    """Calibrate both decision bands on validation and persist them with the model."""
    thr_block = calibrate_block_band()
    thr_review = calibrate_review_band()
    bundle = joblib.load(MODELS / "model.joblib")
    bundle["threshold_f1"] = bundle["threshold"]        # keep the Phase 3 number visible
    bundle["threshold"] = thr_block
    bundle["threshold_review"] = thr_review
    joblib.dump(bundle, MODELS / "model.joblib")
    print(f"block  threshold (cost-optimal, validation) : {thr_block:.4f}")
    print(f"review threshold (95% fraud recall, val)    : {thr_review:.6f}")
    print(f"(F1-optimal from Phase 3, retained for reference: {bundle['threshold_f1']:.4f})")
    print(f"[updated] {MODELS / 'model.joblib'}")


if __name__ == "__main__":
    main()
