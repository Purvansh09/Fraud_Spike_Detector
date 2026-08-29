"""Temporal train/validation/test assignment, per SPEC.md section 3.

This module deliberately writes an *assignment* (txn_id -> split) rather than three
separate transaction files. The reason matters:

    Feature building must run over the full, time-ordered stream and only then filter
    by split. A test transaction on day 50 has rolling-window features that reach back
    into day 49 -- which is validation-period history. Cutting the stream into three
    files first would blank out that history and produce features the production system
    would never see.

That is not leakage: every window looks strictly backward in time, exactly as it would
at authorisation. Leakage would be a test row informing a *training* row, which the
temporal ordering makes impossible.

Also pins a hash of the test set so accidental regeneration is detectable.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from .config import GenConfig

RAW = Path("data/raw")
SPLITS = Path("data/splits")
REPORTS = Path("reports")


def assign(df: pd.DataFrame, cfg: GenConfig) -> pd.Series:
    """Label each transaction train / val / test by the day it occurred on."""
    return pd.Series(
        pd.cut(
            df["day"],
            bins=[-1, cfg.train_end_day - 1, cfg.val_end_day - 1, df["day"].max()],
            labels=["train", "val", "test"],
        ),
        index=df.index,
    ).astype(str)


def main() -> None:
    cfg = GenConfig()
    txns = pd.read_parquet(RAW / "transactions.parquet")
    labels = pd.read_parquet(RAW / "labels.parquet")
    df = txns.merge(labels, on="txn_id", validate="one_to_one").sort_values("timestamp")
    df = df.reset_index(drop=True)

    df["split"] = assign(df, cfg)

    SPLITS.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(exist_ok=True)
    df[["txn_id", "split"]].to_parquet(SPLITS / "assignment.parquet", index=False)

    # Pin the test set. If this hash ever changes, the held-out set was regenerated and
    # any previously reported test metric is void.
    test_ids = df.loc[df["split"] == "test", "txn_id"].sort_values().tolist()
    digest = hashlib.sha256("\n".join(test_ids).encode()).hexdigest()

    summary = {
        "seed": cfg.seed,
        "n_transactions": int(len(df)),
        "test_set_sha256": digest,
        "test_set_size": len(test_ids),
        "boundaries": {
            "train": [0, cfg.train_end_day],
            "val": [cfg.train_end_day, cfg.val_end_day],
            "test": [cfg.val_end_day, int(df["day"].max()) + 1],
        },
        "per_split": {
            name: {
                "transactions": int((df["split"] == name).sum()),
                "fraud": int(df.loc[df["split"] == name, "is_fraud"].sum()),
                "episodes": int(
                    df.loc[(df["split"] == name) & (df["is_fraud"] == 1), "episode_id"].nunique()
                ),
            }
            for name in ("train", "val", "test")
        },
    }
    (REPORTS / "split_manifest.json").write_text(json.dumps(summary, indent=2))

    for name, s in summary["per_split"].items():
        print(f"{name:>5}: {s['transactions']:>6,} txns  {s['fraud']:>4} fraud  "
              f"{s['episodes']:>3} episodes")
    print(f"\ntest set sha256: {digest[:16]}...  ({len(test_ids):,} transactions)")
    print(f"[written] {SPLITS / 'assignment.parquet'}, {REPORTS / 'split_manifest.json'}")


if __name__ == "__main__":
    main()
