# Fraud-Spike Detector

Detects **bursts of takeover fraud** — 4–8 transactions on one account inside a few minutes,
from a device that account has never used — and returns a decision with a plain-English reason.

Defence-only. This repository contains no fraud-generation capability beyond the synthetic data
generator used to create a labelled training set, and nothing that describes how to evade the
detector.

---

## Headline numbers

> Held-out test metrics land here at the end of Phase 3. The test set is pinned by SHA-256 in
> `reports/split_manifest.json` and is scored exactly once. Nothing is reported before then.

What is measured today, on the generated dataset:

| Hand-written rule (no model) | Precision | Recall | F1 |
|---|---|---|---|
| 4+ transactions in 10 min | 17.1% | 49.0% | 0.254 |
| Device new to account (<1h) | 31.3% | 99.4% | 0.476 |
| **Burst AND new device** | **81.2%** | **48.9%** | **0.610** |

That last row is the bar the model has to clear. A gradient-boosted model that cannot beat a
two-line rule does not deserve to ship, so the rule is reported alongside the model throughout.

---

## Quickstart

```bash
python -m venv .venv && ./.venv/Scripts/python.exe -m pip install -r requirements.txt
```

Then regenerate everything from the pinned seed:

```bash
PYTHONPATH=src ./.venv/Scripts/python.exe -m fraudspike.generate
```

```bash
PYTHONPATH=src ./.venv/Scripts/python.exe -m fraudspike.audit_data
```

```bash
PYTHONPATH=src ./.venv/Scripts/python.exe -m fraudspike.splits
```

## Layout

| Path | What |
|---|---|
| `SPEC.md` | The frozen label definition, confounders, and leakage controls. **Read this first.** |
| `src/fraudspike/config.py` | Every generator knob, with the reasoning next to it |
| `src/fraudspike/generate.py` | Synthetic transaction + fraud-episode generator |
| `src/fraudspike/audit_data.py` | Adversarial audit: tries to break our own dataset |
| `src/fraudspike/splits.py` | Temporal split assignment, pins the test-set hash |
| `src/fraudspike/features.py` | 33 strictly backward-looking features |
| `src/fraudspike/audit_features.py` | Flags any single feature that looks too good to be true |
| `tests/test_causality.py` | Proves no feature can see the future |
| `reports/data_audit.md` | Latest data audit output |
| `reports/feature_audit.md` | Per-feature discriminative power, train split only |
| `reports/split_manifest.json` | Split sizes and the test-set SHA-256 |

## How we know there is no leakage

Three independent guards, in increasing order of how much they'd convince a sceptic:

1. **Structural.** Labels live in a separate frame. The feature builder's input does not
   contain `is_fraud` or `episode_id`, so it cannot use them even by accident.
2. **Empirical.** `tests/test_causality.py` rebuilds every feature from a stream truncated at
   day 10, 25, 40 and 50, and asserts each row is bit-for-bit identical to the same row built
   from the full 60 days — NaN patterns included. Anything that peeked forward would differ.
3. **The test is itself tested.** We planted a deliberate look-ahead feature (an account's
   all-time mean amount) and confirmed the truncation test fails on all four cut points, then
   removed it. A green suite that cannot go red proves nothing.

```bash
./.venv/Scripts/python.exe -m pytest tests/ -q
```

## Why synthetic data

Public fraud datasets (the ULB Kaggle set, for instance) are single-transaction snapshots with
no account identifiers or timestamp sequences, so velocity and burst features — the entire
subject of this project — cannot be built on them. Generating the data means we control the
ground truth exactly and can guarantee a clean split.

The obvious objection is that a generator can be tuned until the model looks good. Two things
guard against that: `SPEC.md` was frozen *before* any data was generated, and `audit_data.py`
adversarially checks that no single column separates the classes. It has already caught one
real leak — see the SPEC changelog.

## Notes

- `Faker` was in the original plan but is not used. The generator needs behavioural
  distributions, not realistic names, and NumPy covers that without the extra dependency.
