# Fraud-Spike Detector

Detects **bursts of takeover fraud** — 4–8 transactions on one account inside a few minutes,
from a device that account has never used — and returns a decision with a plain-English reason.

Defence-only. This repository contains no fraud-generation capability beyond the synthetic data
generator used to create a labelled training set, and nothing that describes how to evade the
detector.

---

## Headline numbers

**Held-out test set, 5,865 transactions, 121 fraud, scored exactly once** at the threshold
chosen on validation. Full report: [`reports/test_report.md`](reports/test_report.md).

| | Precision | Recall | F1 |
|---|---|---|---|
| always-legitimate | 0.0% | 0.0% | 0.000 |
| rule: burst AND new device | 67.9% | 31.4% | 0.429 |
| **XGBoost** | **82.3%** | **84.3%** | **0.833** |

PR-AUC **0.911** (ROC-AUC 0.996, shown only because it is conventional — it flatters every
model at 2% prevalence). **20 of 20 fraud episodes caught.** 22 false positives across 5,744
legitimate transactions, a 0.38% false-positive rate.

Recall is not uniform across attack styles, and the average hides that:

| archetype | txn recall | episodes caught |
|---|---|---|
| `classic_burst` | 100.0% | 6/6 |
| `slow_drain` | 100.0% | 3/3 |
| `blend_in` | 75.7% | 5/5 |
| `session_hijack` | 69.7% | 6/6 |

Session hijacks are the weak point — by construction they carry no device-novelty signal at
all, so the model has only velocity and amount to work with.

### What that is worth

At the operating threshold chosen on validation, over the 11-day test window
([`reports/cost_analysis.md`](reports/cost_analysis.md)):

| | amount |
|---|---|
| Fraud loss avoided | ₹912,201 |
| False-positive cost | ₹34,598 |
| Residual fraud cost | ₹77,018 |
| **Net saving vs no detector** | **₹877,603** |

Savings are episode-aware: blocking the 3rd transaction of an 8-transaction burst prevents
six, not one, because the account is frozen behind the decline.

The threshold is **completely insensitive** to false-positive pricing across every plausible
assumption — because the average fraud ticket (₹6,844) is an order of magnitude larger than
the cost of a wrong decline (~₹2,021). It only starts to move once a false positive costs
about **₹5,000, roughly 73% of a fraudulent transaction**. So the claim is not "this is
robust" but: *robust provided a wrongly declined customer costs you less than three-quarters
of a fraud*. A high-LTV lender should re-run it with their own figures.

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
| `src/fraudspike/train.py` | Model selection + threshold tuning. **Cannot read the test split.** |
| `src/fraudspike/evaluate.py` | The only script that opens the test set. Runs once. |
| `reports/validation_report.md` | Grid, baselines, SMOTE ablation, chosen threshold |
| `reports/test_report.md` | The held-out numbers |
| `src/fraudspike/costs.py` | Cost model, threshold economics, sensitivity + stress test |
| `reports/cost_analysis.md` | Rupee analysis and where the recommendation breaks |
| `src/fraudspike/serving.py` | Live scorer + account history store + decision bands |
| `src/fraudspike/explain.py` | SHAP attribution, Claude reason, template fallback |
| `api/app/main.py` | FastAPI service + dashboard route |
| `dashboard/index.html` | Single-page dashboard, no build step |
| `tests/test_causality.py` | Proves no feature can see the future |
| `tests/test_serving.py` | Proves serving features match training features exactly |
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

## Dashboard

Start the service and open <http://127.0.0.1:8017/>.

The table lists every flagged transaction plus **every fraud the detector missed**, filterable
by Blocked / Review / False positives / Missed fraud. Clicking a row fetches the SHAP
attribution and a one-sentence explanation on demand — the table renders instantly from local
scores, and only the language layer costs a round trip.

Two things the filters make visible that a metrics table hides:

- **The 22 false positives are not 22 unrelated mistakes.** Five of them are one account
  (`ACC00162`) buying gift cards in Pune inside a two-minute window — a genuine shopping burst
  that is close to indistinguishable from card testing. That is the confounder from `SPEC.md`
  section 2 working exactly as designed, and the model losing to it.
- **Raw recall undersells a three-band system.** Of 121 fraud transactions in the test window,
  103 were blocked, 11 were held for human review, and only **7 were waved through entirely** —
  so 94.2% reached a human or were stopped. Those 7 are all `session_hijack` and `blend_in`,
  the two archetypes carrying no device-novelty signal.

Built as one plain HTML file rather than React: it meets every Phase 6 requirement with no
build step, no `node_modules`, and nothing that can break during a live demo. The plan's
rationale for React was reusing JobAgent's setup, which does not exist in this repo.

## Running the service

```bash
PYTHONPATH=src ./.venv/Scripts/python.exe -m uvicorn api.app.main:app --reload
```

`POST /score` returns a risk score, a decision band, the SHAP factors behind it, and one
sentence of plain English. `GET /model` reports the deployed thresholds and where they came
from. `GET /demo/flagged` serves the Phase 6 dashboard.

Decision bands, both calibrated on validation and never on test:

| band | threshold | meaning |
|---|---|---|
| `ALLOW` | < 0.0071 | clear automatically |
| `REVIEW` | 0.0071 – 0.1282 | hold for a human; the band retains 95% fraud recall |
| `BLOCK` | >= 0.1282 | decline; this is the Phase 4 cost-optimal point |

Explanations are optional by design. Set `ANTHROPIC_API_KEY` in `.env` (see `.env.example`)
and Claude phrases the SHAP attributions; without a key, a deterministic template renders the
same facts. **The decision never depends on the API** — Claude is given the attributions and
asked to phrase them, never asked whether the transaction is fraud, and its output is
generated after the decision is already made.

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
