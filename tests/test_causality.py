"""Proof, not promise, that the feature builder cannot see the future.

The leakage claim in SPEC.md section 3 is the load-bearing one for this whole project:
if a feature peeks forward, every metric downstream is fiction. These tests check it
empirically rather than by inspection.

The central test is `test_truncating_the_stream_changes_nothing`: rebuild the features
from a stream truncated at time T, and every row at or before T must come out bit-for-bit
identical to its value computed on the full 60-day stream. A feature that looked even one
transaction ahead would necessarily differ.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fraudspike.features import build  # noqa: E402

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"


@pytest.fixture(scope="module")
def txns() -> pd.DataFrame:
    return pd.read_parquet(RAW / "transactions.parquet")


@pytest.fixture(scope="module")
def full_features(txns: pd.DataFrame) -> pd.DataFrame:
    return build(txns).set_index("txn_id").sort_index()


@pytest.mark.parametrize("cut_day", [10, 25, 40, 50])
def test_truncating_the_stream_changes_nothing(txns, full_features, cut_day):
    """Features for rows up to day D must not depend on anything after day D."""
    truncated = txns[txns["day"] <= cut_day]
    partial = build(truncated).set_index("txn_id").sort_index()

    reference = full_features.loc[partial.index]
    assert list(reference.columns) == list(partial.columns)

    # NaN must line up too -- a feature that quietly gains a value once future rows
    # arrive is just as much a leak as one whose number changes.
    mismatched_nan = (reference.isna() != partial.isna())
    assert not mismatched_nan.to_numpy().any(), (
        "NaN pattern differs after truncation in columns: "
        f"{sorted(reference.columns[mismatched_nan.any()])}"
    )

    np.testing.assert_allclose(
        reference.to_numpy(dtype=float),
        partial.to_numpy(dtype=float),
        rtol=0, atol=0, equal_nan=True,
        err_msg=f"feature values changed when the stream was cut at day {cut_day}",
    )


def test_builder_never_receives_labels(txns):
    """Structural guard: the label columns are not even in the builder's input."""
    forbidden = {"is_fraud", "episode_id"}
    assert not forbidden & set(txns.columns)
    assert not forbidden & set(build(txns).columns)


def test_first_transaction_of_account_has_no_history(txns):
    """A brand-new account must report unknown history, not a fabricated zero."""
    feats = build(txns)
    df = txns.sort_values(["timestamp", "txn_id"]).reset_index(drop=True)
    first_rows = df.groupby("account_id", sort=False).head(1)["txn_id"]
    first = feats.set_index("txn_id").loc[first_rows]

    assert first["secs_since_prev"].isna().all()
    assert first["amt_ratio_to_prior_median"].isna().all()
    assert (first["acct_txns_prior"] == 0).all()
    assert (first["device_txns_prior"] == 0).all()
    # Age-on-account is genuinely zero the first time a device is seen, not unknown.
    assert (first["device_age_secs"] == 0).all()


def test_velocity_counts_include_current_transaction(txns):
    """A count of 1 means 'this transaction and nothing else in the window'."""
    feats = build(txns)
    for col in ("txn_count_5m", "txn_count_15m", "txn_count_60m"):
        assert feats[col].min() >= 1, f"{col} should never be below 1"
    # Wider windows can only see more.
    assert (feats["txn_count_60m"] >= feats["txn_count_15m"]).all()
    assert (feats["txn_count_15m"] >= feats["txn_count_5m"]).all()


def test_window_sums_are_consistent_with_counts(txns):
    """Zero-amount transactions do not exist, so any window with n transactions must
    carry a strictly positive sum."""
    feats = build(txns)
    for w in ("5m", "15m", "60m"):
        assert (feats[f"amount_sum_{w}"] > 0).all()
