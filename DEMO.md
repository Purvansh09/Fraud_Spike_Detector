# Demo Walkthrough

Target: **5 minutes**, then questions. The structure is deliberate — problem, model, honest
numbers, money, live flag. Do not open with the architecture; open with the problem.

---

## Before you start

```bash
./.venv/Scripts/python.exe -m uvicorn api.app.main:app --port 8017
```

- [ ] Service up, <http://127.0.0.1:8017/> loaded and showing the metric cards
- [ ] One row already clicked once, so its explanation is **cached** and returns instantly
      (a cold Claude call takes 2–5 seconds; do not spend that live)
- [ ] `reports/cost_analysis.md` open in a second tab
- [ ] `SPEC.md` open in a third — you will want it if leakage comes up
- [ ] Know your fallback: if the network dies, explanations degrade to the template
      automatically. Say so out loud rather than hiding it; it is a design feature.

---

## The five beats

### 1 — The problem (30s)

> "Account takeover doesn't look like one bad transaction. It looks like four to eight in a
> few minutes, from a device that account has never used — a small card-testing probe, then
> escalation. We detect that burst, and we explain every decision in a sentence a fraud
> analyst can act on."

Say **defence-only** once, early. The track requires it.

### 2 — Why the data is synthetic, and why that isn't a cop-out (45s)

This is where most teams get challenged, so get ahead of it.

> "Public fraud datasets are single-transaction snapshots with no account IDs or timestamps,
> so you cannot build velocity features on them at all. We generated data — which means the
> obvious question is whether we tuned the generator until the model looked good."

Then show `SPEC.md`:

> "The label definition was frozen before any data existed. Every change since is in the
> changelog at the bottom. And we wrote an audit that attacks our own dataset — it checks
> that no single column separates fraud from legitimate."

**The line that lands:** it caught a real leak.

> "Our first run scored 100% precision on 'city different from home'. Legitimate accounts
> never travelled, so one column was a perfect classifier. We added travel, regenerated, and
> it dropped to 16.8%. That's in the changelog. No model had been trained yet."

### 3 — The numbers (60s)

Point at the dashboard cards.

| | |
|---|---|
| Precision | **82.4%** |
| Recall | **85.1%** |
| PR-AUC | **0.911** |
| Episodes caught | **20/20** |
| False positives | **22** of 5,744 legitimate |

> "PR-AUC is the number to read at 2% prevalence. ROC-AUC is 0.996 and we report it, but it
> flatters everything on imbalanced data and nobody should quote it."

Then the comparison that justifies the model existing at all:

> "A hand-written rule — burst AND new device — gets 67.9% precision and 31.4% recall. The
> model roughly triples recall at higher precision. If it hadn't beaten the rule, we'd have
> shipped the rule."

And the framing that's better than raw recall:

> "Of 121 fraud transactions: 103 blocked, 11 held for human review, 7 waved through
> entirely. So 94.2% were stopped or put in front of a person."

### 4 — What it's worth, and where the claim breaks (75s)

Switch to `reports/cost_analysis.md`.

> "Over the 11-day test window: ₹912k of fraud prevented, ₹35k of false-positive cost,
> ₹878k net saving. Savings are episode-aware — blocking the third transaction of an
> eight-transaction burst prevents six, because the account is frozen behind the decline."

Then the part almost nobody else will have:

> "The threshold was chosen on validation, never on test. And we stress-tested it: it doesn't
> move at all across any plausible cost assumption, because the average fraud ticket is
> ₹6,844 and a wrong decline costs about ₹2,000. It only starts moving once a false positive
> costs about ₹5,000 — roughly three-quarters of a fraud. So the claim isn't 'this is robust',
> it's 'this is robust provided a wrongly declined customer costs you less than
> three-quarters of a fraudulent transaction.'"

If asked whose money: **be straight** — it's an ecosystem view (merchant + issuer), not
Razorpay's own P&L. On a strict PSP P&L you'd use MDR (~2%) instead of merchant margin, and
under 3DS liability shift the issuer bears much of the fraud loss.

### 5 — Live flag (60s)

Click a `classic_burst` row.

> "This is the first transaction of a burst — an ₹80 probe. Blocked at 0.997. The model saw a
> device zero seconds old on this account, 2am, far outside this account's normal hours."

Then the SHAP bars, then the sentence:

> "SHAP produces the attribution locally. Claude only phrases it — it's never asked whether
> this is fraud, and it runs after the decision is already made. If the API is down, a
> deterministic template says the same thing. The decision never depends on the network."

**Close on the failures, not the wins.** Click **False positives**:

> "Five of our 22 false positives are one account buying gift cards in Pune inside two
> minutes. That's a real customer having a genuine shopping burst that looks exactly like
> card testing. We'd rather show you that than a curated success list."

---

## Questions you should expect

**"How do we know there's no leakage?"**
Three guards. Labels live in a separate frame the feature builder never receives. Every
feature is rebuilt from streams truncated at days 10/25/40/50 and asserted bit-identical to
the full-stream values. And we planted a deliberate look-ahead feature to confirm that test
actually fails — a green suite that can't go red proves nothing.

**"Did you touch the test set more than once?"**
Scores were computed once. Two thresholds are reported, 0.1349 and 0.1282, and **both were
chosen on validation**. `reports/operating_point.md` reconciles them. We also report the
generalisation gap: the retrospective test-optimal threshold would have saved ₹20,678 more,
a 2.3% shortfall. That's the price of not tuning on test.

**"20% of accounts compromised is unrealistic."**
Correct, and it's stated in SPEC §4. It's set high so the test window holds ~18 episodes
instead of ~4 — below a dozen, episode recall swings double digits on one episode. The
quantity that actually drives precision/recall is transaction-level prevalence, held at a
realistic 2.06%.

**"What's the weakest part?"**
Session hijacks: 72.7% recall versus 100% on classic bursts. By construction they reuse a
known device, so the strongest single feature is unavailable and the model has only velocity
and amount. The 7 fraud transactions that got through entirely are all `session_hijack` and
`blend_in`.

**"Would this work on real data?"**
Unknown, and we won't claim otherwise. The feature logic is standard and the serving path is
proven identical to training, but the generator is our own model of fraud. The honest next
step is a shadow deployment against real traffic.

---

## If something breaks

| Symptom | Do this |
|---|---|
| Explanation slow | It's a live API call. Talk over it, or use a pre-clicked cached row. |
| Explanations say `template` | No API key loaded. Fine — say it's the offline fallback, keep going. |
| Service won't start | Check cwd is the repo root; paths are anchored, but the venv isn't. |
| Dashboard empty | `data/splits/features.parquet` missing — run `bash reproduce.sh`. |
