# Fraud-Spike Detector

Detects **bursts of takeover fraud**: four to eight transactions on one account within a few
minutes, from a device the account has never used. Every flag comes back with a decision and a
plain-English reason.

Defence-only. The only thing here that produces fraud is the synthetic data generator that
builds the training set. Nothing in this repo explains how to get past the detector.

---

## Headline numbers

**Held-out test set, 5,865 transactions, 121 fraud, scored exactly once**, reported at the
deployed threshold (0.1282, chosen on validation in Phase 4).

| | Precision | Recall | F1 |
|---|---|---|---|
| always-legitimate | 0.0% | 0.0% | 0.000 |
| rule: burst AND new device | 67.9% | 31.4% | 0.429 |
| **XGBoost** | **82.4%** | **85.1%** | **0.837** |

PR-AUC is **0.911**. ROC-AUC is 0.996, which we report only because people expect it; at 2%
prevalence it makes every model look good. **All 20 fraud episodes were caught.** There were 22
false positives across 5,744 legitimate transactions, a false-positive rate of 0.38%.

Of the 121 fraud transactions, **103 were blocked, 11 were held for human review, and 7 got
through**. That means 94.2% were either stopped or put in front of a person. With three
decision bands, raw recall undersells what the system actually does.

> Two thresholds exist in this project and both were chosen on validation: 0.1349 (F1-optimal,
> Phase 3) and 0.1282 (cost-optimal, deployed). The test scores were computed once and read at
> both. [`reports/operating_point.md`](reports/operating_point.md) reconciles them;
> [`reports/test_report.md`](reports/test_report.md) is the original Phase 3 record, kept
> unedited.

Recall is not uniform across attack styles, and the average hides that:

| archetype | txn recall | episodes caught |
|---|---|---|
| `classic_burst` | 100.0% | 6/6 |
| `slow_drain` | 100.0% | 3/3 |
| `blend_in` | 75.7% | 5/5 |
| `session_hijack` | 72.7% | 6/6 |

Session hijacks give us the most trouble. The attacker rides a device the account already
trusts, so there's no new device to notice. That leaves the model reading speed and spend, and
not much else.

### What that is worth

At the operating threshold chosen on validation, over the 11-day test window
([`reports/cost_analysis.md`](reports/cost_analysis.md)):

| | amount |
|---|---|
| Fraud loss avoided | ₹912,201 |
| False-positive cost | ₹34,598 |
| Residual fraud cost | ₹77,018 |
| **Net saving vs no detector** | **₹877,603** |

Savings are counted per episode, not per transaction. Block the third transaction of an
eight-transaction burst and you prevent six, because the account is frozen behind the decline.

The threshold barely moves no matter how we price false positives. The average fraud ticket is
₹6,844 while a wrong decline costs around ₹2,021, so fraud dominates the arithmetic. It only
starts to shift once a false positive costs about **₹5,000, roughly 73% of a fraudulent
transaction**.

So we won't claim the threshold is simply robust. It's robust as long as a wrongly declined
customer costs you less than three-quarters of a fraud. A lender with high-value customers
should re-run this with their own numbers.

---

## Quickstart

The data, the trained model and every report are committed, so the service runs immediately:

```bash
python -m venv .venv && ./.venv/Scripts/python.exe -m pip install -r requirements.txt
```

```bash
./.venv/Scripts/python.exe -m uvicorn api.app.main:app --port 8017
```

Then open <http://127.0.0.1:8017/>. Demo script: [`DEMO.md`](DEMO.md).

### Reproducing every number

You don't need this to run anything. It's here so you can check the numbers instead of
taking our word for them. One script rebuilds the whole pipeline from the pinned seed, in
order:

```bash
bash reproduce.sh
```

**Verified on 2026-09-05:** regenerating gives byte-identical data files and the same test-set
SHA-256 (`622d42b4fde8ed01...`) recorded in `reports/split_manifest.json`. If that hash ever
changes, someone rebuilt the held-out set, and every test metric here is void.

## Layout

| Path | What |
|---|---|
| `DEMO.md` | Five-minute walkthrough script and the questions to expect |
| `reproduce.sh` | Rebuilds everything from the seed, in order |
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
| `reports/test_report.md` | The held-out numbers, as recorded in Phase 3 |
| `reports/operating_point.md` | Reconciles the F1 and cost-optimal thresholds |
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

Three separate guards, roughly in order of how convincing they are:

1. **Structural.** Labels live in a separate frame that the feature builder never receives. It
   can't use `is_fraud` or `episode_id` even by accident.
2. **Empirical.** `tests/test_causality.py` rebuilds every feature from streams cut off at days
   10, 25, 40 and 50, then checks each row matches the version built from all 60 days, right
   down to which values are missing. A feature that looked ahead would come out different.
3. **We tested the test.** We planted a look-ahead feature on purpose (an account's all-time
   mean amount) and confirmed the truncation test failed at all four cut points, then took it
   out. A suite that can't go red isn't proving anything.

```bash
./.venv/Scripts/python.exe -m pytest tests/ -q
```

## Dashboard

Start the service and open <http://127.0.0.1:8017/>.

The table lists every flagged transaction and **every fraud the detector missed**, filterable
by Blocked, Review, False positives, and Missed fraud. Click a row and it fetches the SHAP
attribution and a one-sentence explanation. The table renders instantly from local scores; only
the wording costs a round trip.

Two things the filters show that a metrics table can't:

- **The 22 false positives aren't 22 separate mistakes.** Five of them are one account
  (`ACC00162`) buying gift cards in Pune inside a two-minute window. That's a real shopping
  burst, and it looks almost exactly like card testing. The confounder from `SPEC.md` section 2
  is doing its job, and the model is losing to it fairly.
- **Raw recall undersells three bands.** Of 121 fraud transactions, 103 were blocked, 11 went
  to human review, and only **7 got through**. So 94.2% were stopped or seen by a person. All 7
  misses are `session_hijack` and `blend_in`, the two archetypes with no device-novelty signal.

This is one plain HTML file, not React. It does everything Phase 6 asked for with no build
step, no `node_modules`, and nothing that can break mid-demo. The original plan chose React to
reuse JobAgent's setup, but that setup doesn't exist in this repo.

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

Explanations are optional on purpose. Put `ANTHROPIC_API_KEY` in `.env` (see `.env.example`)
and Claude will phrase the SHAP attributions. Without a key, a fixed template states the same
facts.

**The decision never depends on the API.** Claude gets the attributions and is asked to word
them. It is never asked whether the transaction is fraud, and it runs after the decision has
already been made.

## Why synthetic data

Public fraud datasets, like the ULB Kaggle set, are single-transaction snapshots with no
account IDs and no timestamp sequences. Velocity and burst features are the whole point of this
project, and you cannot build them on that data. Generating our own means we know the ground
truth exactly and can guarantee a clean split.

The obvious objection is that you can keep tuning a generator until the model looks good. Two
things stop that. `SPEC.md` was frozen *before* any data existed, and `audit_data.py` attacks
our own dataset to check that no single column separates the classes. It has already caught one
real leak; the SPEC changelog has the details.

## Notes

- `Faker` was in the original plan but is not used. The generator needs behavioural
  distributions, not realistic names, and NumPy covers that without the extra dependency.
