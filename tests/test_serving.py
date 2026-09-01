"""Train/serve parity: the live scorer must compute exactly the batch features.

This is the serving equivalent of the causality test. A model that scores well offline
and badly in production usually does so because the serving path recomputes features
slightly differently -- a window boundary off by one, a missing value filled instead of
left NaN, history sorted differently. Nothing in the offline metrics catches that.

So: take real transactions, score them through the serving path, and assert the feature
row is bit-for-bit identical to the row the batch pipeline produced for the same
transaction.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fraudspike.serving import TXN_FIELDS, HistoryStore, Scorer  # noqa: E402

RAW = ROOT / "data" / "raw"
SPLITS = ROOT / "data" / "splits"


@pytest.fixture(scope="module")
def scorer() -> Scorer:
    return Scorer()


@pytest.fixture(scope="module")
def batch_features() -> pd.DataFrame:
    return pd.read_parquet(SPLITS / "features.parquet").set_index("txn_id")


@pytest.fixture(scope="module")
def sample() -> pd.DataFrame:
    """A mix of fraud and legitimate transactions from accounts with real history."""
    txns = pd.read_parquet(RAW / "transactions.parquet")
    labels = pd.read_parquet(RAW / "labels.parquet")
    df = txns.merge(labels, on="txn_id")
    # Skip each account's earliest rows: those exercise the no-history path, which is
    # covered separately, and we want the history-dependent path here.
    df = df.groupby("account_id", sort=False).tail(20)
    fraud = df[df["is_fraud"] == 1].sample(25, random_state=7)
    legit = df[df["is_fraud"] == 0].sample(25, random_state=7)
    return pd.concat([fraud, legit]).reset_index(drop=True)


def test_serving_features_match_batch_exactly(scorer, batch_features, sample):
    """The whole point. Any drift here is train/serve skew."""
    mismatches = []
    for _, txn in sample.iterrows():
        served = scorer.feature_row({k: txn[k] for k in TXN_FIELDS})
        expected = batch_features.loc[[txn["txn_id"]], scorer.features]

        same_nan = (served.isna().to_numpy() == expected.isna().to_numpy()).all()
        close = np.allclose(
            served.to_numpy(dtype=float), expected.to_numpy(dtype=float),
            rtol=0, atol=0, equal_nan=True,
        )
        if not (same_nan and close):
            diff = [
                c for c in scorer.features
                if not np.array_equal(
                    served[c].to_numpy(dtype=float),
                    expected[c].to_numpy(dtype=float),
                    equal_nan=True,
                )
            ]
            mismatches.append((txn["txn_id"], diff))

    assert not mismatches, f"serving/batch feature drift on {len(mismatches)} txns: {mismatches[:3]}"


def test_scores_match_batch_pipeline(scorer, batch_features, sample):
    """Identical features must of course give identical scores."""
    for _, txn in sample.iterrows():
        served_score, _ = scorer.score({k: txn[k] for k in TXN_FIELDS})
        expected = batch_features.loc[[txn["txn_id"]], scorer.features]
        batch_score = float(scorer.model.predict_proba(expected)[:, 1][0])
        assert served_score == pytest.approx(batch_score, abs=1e-12)


def test_unknown_account_is_handled(scorer):
    """A first-ever transaction must score without history, not crash."""
    decision, X = scorer.decide({
        "txn_id": "TXN_NEW_0001",
        "account_id": "ACC_NEVER_SEEN",
        "timestamp": pd.Timestamp("2026-08-01 03:14:00"),
        "amount": 42000.0,
        "merchant_id": "MRC_GIFT_001",
        "merchant_category": "gift_cards",
        "device_id": "DEV_UNKNOWN",
        "city": "Mumbai",
        "channel": "card",
    })
    assert 0.0 <= decision.score <= 1.0
    assert decision.action in {"ALLOW", "REVIEW", "BLOCK"}
    # No history means unknown, not zero.
    assert pd.isna(X["amt_ratio_to_prior_median"].iloc[0])
    assert X["acct_txns_prior"].iloc[0] == 0


def test_decision_bands_are_ordered(scorer):
    assert 0 < scorer.threshold_review < scorer.threshold_block < 1


def test_appending_history_changes_velocity(scorer):
    """Sanity: the store actually feeds the next scoring call."""
    store = HistoryStore()
    scorer2 = Scorer(store=store)
    base = {
        "account_id": "ACC_TEST", "amount": 900.0, "merchant_id": "MRC_GROC_001",
        "merchant_category": "groceries", "device_id": "DEV_T_0", "city": "Pune",
        "channel": "upi",
    }
    t0 = pd.Timestamp("2026-08-01 12:00:00")
    first = {**base, "txn_id": "T1", "timestamp": t0}
    _, X1 = scorer2.decide(first)
    assert X1["txn_count_5m"].iloc[0] == 1

    scorer2.store.append(first)
    _, X2 = scorer2.decide({**base, "txn_id": "T2", "timestamp": t0 + pd.Timedelta(minutes=1)})
    assert X2["txn_count_5m"].iloc[0] == 2
