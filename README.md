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
| `reports/data_audit.md` | Latest audit output |
| `reports/split_manifest.json` | Split sizes and the test-set SHA-256 |

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
