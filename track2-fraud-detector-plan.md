# Track 2 — AI Risk Manager: Fraud-Spike Detector
### Build Plan

---

## 1. Initial Approach

**What we're building:** A detector that flags bursts of suspicious transaction activity — a "fraud spike" — from an account or merchant within a short time window, backed by a real machine-learned model, an honest precision/recall report on held-out data, and an AI-generated explanation for every flag.

**Definition of done for the hackathon:**
- A trained classifier with reported precision, recall, F1, and ROC-AUC on a held-out test set
- A false-positive cost analysis translating the metrics into business terms
- A working API that scores a transaction and returns a decision + plain-English reason
- A minimal dashboard showing flagged transactions and running accuracy
- Strictly defense-only — no fraud-generation logic exposed as a "feature," only detection

**Scope discipline:** Resist the urge to detect every type of fraud. One well-defined pattern (velocity/burst-based spikes), done rigorously, beats five patterns done shakily. This directly serves "the bar" the track sets: honest, measured, not cherry-picked.

---

## 2. Tech Stack

| Layer | Choice | Why |
|---|---|---|
| Data & ML | Python, pandas, scikit-learn, XGBoost, imbalanced-learn (SMOTE), SHAP | Standard, fast, you already know XGBoost well |
| Explainability | SHAP values → fed into Claude API for natural-language reasons | Turns raw feature importances into human-readable flags |
| Backend | FastAPI | You already use this in JobAgent — reuse patterns |
| AI reasoning layer | Claude API | Explanation generation + optional Q&A on flags |
| Frontend | React + TypeScript | Reuse JobAgent's existing setup where possible |
| Storage | Supabase (Postgres) or just local CSV/SQLite | Supabase only if time allows — CSV is fine for a hackathon demo |
| Version control | GitHub | Push early, commit often — judges may check history |

---

## 3. Where to Get the Data

**Recommended: generate synthetic transaction data yourself.** Here's why this beats downloading a public dataset for this specific track:

- Public fraud datasets (e.g. the ULB Kaggle credit card dataset) are **single-transaction snapshots with no account IDs or timest<br>sequences** — you can't build velocity/burst features on them, which is the whole point of "spike detection."
- Synthetic data gives you full control: you decide account IDs, transaction timing, amount distributions, and where to inject fraud bursts — which means you can *guarantee* a clean, defensible train/test split and known ground truth.
- It directly matches the track's language: "synthetic data" is explicitly fine (Track 4 says it outright; Track 2 doesn't forbid it either).

**How to generate it:**
1. Use `Faker` (Python) to generate ~5,000–10,000 normal transactions across a few hundred synthetic accounts — realistic amounts, timestamps, merchant categories, devices.
2. Inject fraud patterns programmatically into a subset: bursts of 4–8 transactions within a few minutes, unusual amounts, a new/unseen device, odd hours. Label these `is_fraud=1`.
3. Keep the fraud rate realistic and low (1–3%) — this preserves the class-imbalance challenge that makes precision/recall reporting meaningful (a trivial "always predict legit" baseline should score badly).

**Optional credibility booster (if time allows):** Cross-check your synthetic model's behavior against the public ULB or IEEE-CIS Kaggle fraud datasets by validating that your feature engineering approach (amount deviation, unusual timing) also holds up there — even a small side validation shows judges you're not just overfitting to your own synthetic generator.

---

## 4. Phase-by-Phase Plan

### Phase 0 — Setup (few hours)
- Repo scaffold: FastAPI backend, React frontend, `/data`, `/notebooks` folders
- Decide final scope: velocity-based fraud-spike detection, nothing broader
- Write one paragraph defining exactly what counts as a "spike" for your synthetic generator (so your labels are principled, not arbitrary)

### Phase 1 — Data Generation
- Build the Faker-based synthetic transaction generator
- Inject fraud-spike patterns with clear, documented rules
- Split into train / validation / **held-out test** (test set touched only once, at the very end)

### Phase 2 — Feature Engineering
- Rolling-window velocity features (transaction count & sum in last 5/15/60 min per account)
- Amount deviation from that account's historical average
- New-device / new-location flags
- Time-of-day anomaly score
- Merchant-category rarity for that account

### Phase 3 — Model Training & Validation
- Train XGBoost with class weighting or SMOTE for the imbalance
- Tune threshold using validation set only
- Run once against held-out test set → report precision, recall, F1, ROC-AUC
- Save the trained model artifact for serving

### Phase 4 — False-Positive Cost Analysis
- Pick 2–3 candidate decision thresholds
- For each, compute: fraud caught (₹ saved) vs. legitimate transactions wrongly blocked (₹ lost sales + friction cost)
- Present as a small table — this is a differentiator almost no other team will do properly

### Phase 5 — Service Layer
- FastAPI endpoint: takes a transaction → returns risk score, flag decision, top contributing features
- Feed top SHAP features into Claude API → generate a one-sentence human-readable reason
- (Optional, if time allows) simple decision layer: auto-hold / auto-escalate / auto-clear based on score + rule bounds — this is the "auto-responder" framing from the track's own example directions

### Phase 6 — Dashboard
- Table of flagged transactions: score, reason, decision
- Running precision/recall readout against the test batch
- Keep the UI plain — function over polish, this isn't what's being judged

### Phase 7 — Polish & Demo Prep
- Write up the metrics + cost analysis clearly (this doc is probably your strongest asset)
- Rehearse a tight walkthrough: define the problem → show the model → show the numbers → show the cost tradeoff → live demo a flagged transaction with its explanation
- Push final commit, README with setup instructions, and the metrics table front and center

---

## Day-by-Day Schedule (Aug 29 → Sep 5)

| Date | Phase | Focus |
|---|---|---|
| **Sat Aug 29** | Phase 0 + start Phase 1 | Repo scaffold, define the "spike" rule precisely, start the synthetic data generator |
| **Sun Aug 30** | Finish Phase 1 | Complete data generation, fraud injection, train/val/test split locked and untouched |
| **Mon Aug 31** | Phase 2 | Feature engineering — velocity windows, deviation scores, device/time anomalies |
| **Tue Sep 1** | Phase 3 | Train XGBoost, tune on validation only, run the held-out test once, record metrics |
| **Wed Sep 2** | Phase 4 + start Phase 5 | Cost analysis table; begin FastAPI scoring endpoint + Claude explanation layer |
| **Thu Sep 3** | Finish Phase 5 + Phase 6 | Complete service layer, build the dashboard (flag list + running precision/recall) |
| **Fri Sep 4** | Phase 7 | Polish writeup, rehearse the demo walkthrough, fix rough edges — **no new features today** |
| **Sat Sep 5** | Buffer / submission | Final commit, README, submit early in the day — don't cut it to the deadline |

**Key guardrail:** if you're behind schedule by Sep 2, cut dashboard polish (Phase 6) before you cut the cost analysis (Phase 4) or metrics rigor (Phase 3) — those are what's actually being judged. A plain table beats a fancy UI with weak numbers.

---

## Things to keep front-of-mind throughout
- **Honesty over polish.** A slightly worse model with clearly reported, unmanipulated metrics beats a suspiciously perfect one.
- **Held-out test set discipline.** Touch it once, at the end. Judges may ask how you avoided leakage — have an answer.
- **Defense-only.** Never show or describe how to *evade* the detector — the track explicitly disqualifies anything offense-capable.
