"""Synthetic transaction generator for velocity-based fraud-spike detection.

Implements the label definition frozen in SPEC.md -- read that first. The one design
rule worth restating here: every individual fraud marker (burst, unseen device, odd
hour, large amount, rare category) is ALSO injected into legitimate traffic, so that
no single marker separates the classes. The model has to learn the conjunction.

Writes two frames, deliberately separated so that label-side bookkeeping cannot leak
into features:
    data/raw/transactions.parquet  -- what a payment processor sees at authorisation
    data/raw/labels.parquet        -- ground truth, joined on txn_id only at eval time
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .config import (
    ALL_CATEGORIES,
    CHANNEL_WEIGHTS,
    CHANNELS,
    CITIES,
    COMMON_CATEGORIES,
    HIGH_RISK_CATEGORIES,
    MERCHANTS_PER_CATEGORY,
    GenConfig,
)

DAY_S = 86_400


def build_merchants(rng: np.random.Generator) -> dict[str, list[str]]:
    """A fixed pool of merchant ids per category, shared across all accounts."""
    return {
        cat: [f"MRC_{cat[:4].upper()}_{i:03d}" for i in range(MERCHANTS_PER_CATEGORY)]
        for cat in ALL_CATEGORIES
    }


def build_accounts(cfg: GenConfig, rng: np.random.Generator) -> list[dict]:
    """One behavioural profile per account: spend level, tempo, hours, category taste."""
    accounts = []
    for i in range(cfg.n_accounts):
        # Most people transact in daylight; a genuine minority are night owls, which is
        # why "3am" on its own must never be treated as evidence of fraud.
        if rng.random() < cfg.night_owl_share:
            peak_hour = float(rng.uniform(0, 5))
        else:
            peak_hour = float(rng.uniform(8, 22))

        # Taste over the everyday categories, plus a small, mostly-zero appetite for the
        # high-risk ones so that a legitimate gift-card purchase is unusual but possible.
        common_w = rng.dirichlet(np.full(len(COMMON_CATEGORIES), 0.8))
        risk_w = rng.dirichlet(np.full(len(HIGH_RISK_CATEGORIES), 0.35)) * rng.uniform(0.01, 0.06)
        weights = np.concatenate([common_w * (1 - risk_w.sum()), risk_w])

        home_city = CITIES[rng.integers(len(CITIES))]

        # Confounder: people travel. Without this, "city != home" is a perfect fraud
        # tell and the model learns one column instead of a behavioural pattern.
        # (The first data audit caught exactly that -- 100% precision on city mismatch.)
        trip_city_by_day: dict[int, str] = {}
        for _ in range(int(rng.integers(0, cfg.trips_per_account_max + 1))):
            dest = str(rng.choice([c for c in CITIES if c != home_city]))
            length = int(rng.integers(cfg.trip_len_min_days, cfg.trip_len_max_days + 1))
            start = int(rng.integers(0, max(1, cfg.n_days - length)))
            for d in range(start, min(start + length, cfg.n_days)):
                trip_city_by_day[d] = dest

        n_devices = int(rng.integers(1, 3))
        accounts.append(
            {
                "account_id": f"ACC{i:05d}",
                "home_city": home_city,
                "trip_city_by_day": trip_city_by_day,
                "devices": [f"DEV_{i:05d}_{d}" for d in range(n_devices)],
                "device_counter": n_devices,
                "log_mu": float(rng.uniform(cfg.amount_log_mu_min, cfg.amount_log_mu_max)),
                "log_sigma": float(rng.uniform(cfg.amount_log_sigma_min, cfg.amount_log_sigma_max)),
                "daily_rate": float(rng.uniform(cfg.daily_rate_min, cfg.daily_rate_max)),
                "peak_hour": peak_hour,
                "hour_spread": float(rng.uniform(1.8, 4.0)),
                "cat_weights": weights / weights.sum(),
                "index": i,
            }
        )
    return accounts


def _sample_hour(acct: dict, rng: np.random.Generator) -> float:
    """Hour-of-day drawn around this account's own peak, wrapped to [0, 24)."""
    # Float modulo returns exactly 24.0 for tiny negative inputs, which would spill a
    # handful of transactions into a phantom extra day. Fold those back to 0.
    h = float(rng.normal(acct["peak_hour"], acct["hour_spread"]) % 24.0)
    return 0.0 if h >= 24.0 else h


def _sample_amount(acct: dict, rng: np.random.Generator) -> float:
    """Lognormal spend with a heavy right tail, so one big purchase is unremarkable."""
    return round(max(10.0, float(rng.lognormal(acct["log_mu"], acct["log_sigma"]))), 2)


def _pick_device(acct: dict, rng: np.random.Generator, cfg: GenConfig) -> str:
    """Usually a known device. Sometimes a genuinely new one -- and that one PERSISTS,
    which is the only thing separating legitimate churn from a fraud device."""
    if rng.random() < cfg.p_new_device_legit:
        dev = f"DEV_{acct['index']:05d}_{acct['device_counter']}"
        acct["device_counter"] += 1
        acct["devices"].append(dev)
        return dev
    # Skew toward the primary device rather than uniform across the account's devices.
    w = np.array([2.0 ** -k for k in range(len(acct["devices"]))])
    return str(rng.choice(acct["devices"], p=w / w.sum()))


def _pick_category(acct: dict, rng: np.random.Generator, cfg: GenConfig) -> str:
    if rng.random() < cfg.p_novel_category:
        return str(rng.choice(ALL_CATEGORIES))
    return str(rng.choice(ALL_CATEGORIES, p=acct["cat_weights"]))


def _legit_txn(acct: dict, ts: float, rng, cfg: GenConfig, merchants) -> dict:
    cat = _pick_category(acct, rng, cfg)
    return {
        "account_id": acct["account_id"],
        "ts": ts,
        "amount": _sample_amount(acct, rng),
        "merchant_category": cat,
        "merchant_id": str(rng.choice(merchants[cat])),
        "device_id": _pick_device(acct, rng, cfg),
        # Away from home if this day falls inside one of the account's trips.
        "city": acct["trip_city_by_day"].get(int(ts // DAY_S), acct["home_city"]),
        "channel": str(rng.choice(CHANNELS, p=CHANNEL_WEIGHTS)),
        "is_fraud": 0,
        "episode_id": None,
    }


def generate_legit(cfg: GenConfig, accounts: list[dict], merchants, rng) -> list[dict]:
    """Day-by-day, in time order, so that device churn accumulates causally."""
    rows: list[dict] = []
    for acct in accounts:
        for day in range(cfg.n_days):
            n = int(rng.poisson(acct["daily_rate"]))
            for _ in range(n):
                ts = day * DAY_S + _sample_hour(acct, rng) * 3600
                rows.append(_legit_txn(acct, ts, rng, cfg, merchants))

            if n == 0:
                continue

            # Confounder: a genuine checkout session -- a real burst, indistinguishable
            # from fraud on the velocity axis alone.
            if rng.random() < cfg.p_legit_session:
                k = int(rng.integers(cfg.legit_session_min, cfg.legit_session_max + 1))
                span = rng.uniform(cfg.legit_session_span_min_s, cfg.legit_session_span_max_s)
                t0 = day * DAY_S + _sample_hour(acct, rng) * 3600
                offsets = np.sort(rng.uniform(0, span, size=k))
                # A real session mostly stays on one device and around one category.
                dev = _pick_device(acct, rng, cfg)
                cat = _pick_category(acct, rng, cfg)
                for off in offsets:
                    r = _legit_txn(acct, t0 + float(off), rng, cfg, merchants)
                    r["device_id"] = dev
                    if rng.random() < 0.7:
                        r["merchant_category"] = cat
                        r["merchant_id"] = str(rng.choice(merchants[cat]))
                    rows.append(r)
    return rows


def generate_fraud(cfg: GenConfig, accounts: list[dict], merchants, rng) -> list[dict]:
    """One takeover episode on each compromised account, per SPEC.md section 1."""
    rows: list[dict] = []
    n_comp = int(round(cfg.compromised_share * cfg.n_accounts))
    victims = rng.choice(len(accounts), size=n_comp, replace=False)

    for ep_no, idx in enumerate(victims):
        acct = accounts[int(idx)]
        episode_id = f"EP{ep_no:04d}"

        day = int(rng.integers(cfg.earliest_episode_day, cfg.n_days))
        hour = rng.uniform(0, 5) if rng.random() < cfg.p_night_episode else rng.uniform(6, 23)
        t0 = day * DAY_S + hour * 3600

        k = int(rng.integers(cfg.burst_min, cfg.burst_max + 1))
        span = rng.uniform(cfg.burst_span_min_s, cfg.burst_span_max_s)
        offsets = np.sort(rng.uniform(0, span, size=k))

        # A device never seen on this account before, and never seen again after.
        device = f"DEVX_{episode_id}"
        if rng.random() < cfg.p_foreign_city:
            city = str(rng.choice([c for c in CITIES if c != acct["home_city"]]))
        else:
            city = acct["home_city"]

        n_probes = int(rng.integers(cfg.n_probes_min, cfg.n_probes_max + 1))
        median_spend = float(np.exp(acct["log_mu"]))

        for j, off in enumerate(offsets):
            if j < n_probes:
                # Card testing: does this credential work at all?
                amount = round(float(rng.uniform(cfg.probe_amount_min, cfg.probe_amount_max)), 2)
            else:
                # Then escalate. Overlaps the legitimate right tail on purpose.
                frac = (j - n_probes) / max(1, k - n_probes - 1)
                factor = cfg.ramp_factor_min + frac * (cfg.ramp_factor_max - cfg.ramp_factor_min)
                amount = round(median_spend * factor * float(rng.uniform(0.75, 1.3)), 2)

            if rng.random() < cfg.p_high_risk_category:
                cat = str(rng.choice(HIGH_RISK_CATEGORIES))
            else:
                cat = str(rng.choice(ALL_CATEGORIES))

            rows.append(
                {
                    "account_id": acct["account_id"],
                    "ts": t0 + float(off),
                    "amount": amount,
                    "merchant_category": cat,
                    "merchant_id": str(rng.choice(merchants[cat])),
                    "device_id": device,
                    "city": city,
                    "channel": str(rng.choice(CHANNELS, p=CHANNEL_WEIGHTS)),
                    "is_fraud": 1,
                    "episode_id": episode_id,
                }
            )
    return rows


def generate(cfg: GenConfig | None = None) -> pd.DataFrame:
    cfg = cfg or GenConfig()
    rng = np.random.default_rng(cfg.seed)

    merchants = build_merchants(rng)
    accounts = build_accounts(cfg, rng)

    rows = generate_legit(cfg, accounts, merchants, rng)
    rows += generate_fraud(cfg, accounts, merchants, rng)

    df = pd.DataFrame(rows)
    df["timestamp"] = pd.Timestamp(cfg.start) + pd.to_timedelta(df["ts"], unit="s")
    df = df.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
    df["txn_id"] = [f"TXN{i:07d}" for i in range(len(df))]
    df["day"] = (df["timestamp"] - pd.Timestamp(cfg.start)).dt.days
    return df.drop(columns=["ts"])


def main() -> None:
    p = argparse.ArgumentParser(description="Generate synthetic transactions (see SPEC.md).")
    p.add_argument("--seed", type=int, default=GenConfig.seed)
    p.add_argument("--out", type=Path, default=Path("data/raw"))
    args = p.parse_args()

    cfg = GenConfig(seed=args.seed)
    df = generate(cfg)
    args.out.mkdir(parents=True, exist_ok=True)

    # The separation that makes the leakage control structural rather than a promise.
    feature_cols = list(cfg.feature_side_columns) + ["day"]
    df[feature_cols].to_parquet(args.out / "transactions.parquet", index=False)
    df[list(cfg.label_side_columns)].to_parquet(args.out / "labels.parquet", index=False)

    n_fraud = int(df["is_fraud"].sum())
    print(f"transactions      : {len(df):,}")
    print(f"accounts          : {df['account_id'].nunique():,}")
    print(f"days              : {int(df['day'].max()) + 1}")
    print(f"fraud transactions: {n_fraud:,}  ({n_fraud / len(df):.2%})")
    print(f"fraud episodes    : {df['episode_id'].nunique():,}")
    print(f"written to        : {args.out}")


if __name__ == "__main__":
    main()
