# Fraud-Spike Detector — Data & Label Specification

**Status:** frozen for Phase 1. Changes after data generation must be recorded in the changelog at the bottom.
**Scope:** velocity/burst-based account-takeover spikes. Nothing else. Defense-only.

---

## 1. What counts as a "fraud spike"

A **fraud spike** is a short burst of transactions on a single account, executed by someone
who is not the account holder, after credential or device compromise.

Formally, an account experiences a spike episode when **all** of the following hold:

| Property | Value in the generator |
|---|---|
| Burst size | 4–8 transactions |
| Burst duration | 3–12 minutes, wall-clock, on one account |
| Device | a `device_id` never previously observed on that account, and never observed again after the episode |
| Amount shape | a card-testing ramp: 1–2 probe transactions (₹10–₹150) followed by escalating high-value attempts drawn from a distribution well above the account's own history |
| Merchant mix | skewed toward liquidation-friendly categories (`digital_goods`, `gift_cards`, `electronics`, `crypto`) that are rare or unseen for that account |
| Hour | 60% of episodes fall in 00:00–05:00 local; 40% deliberately do not |

**Every transaction inside an episode is labelled `is_fraud = 1`.** The label is carried at the
transaction level because that is the unit the API scores in production, but the underlying
phenomenon is episode-level — this is stated explicitly so that the evaluation section below
can also report episode-level recall, which is the number a risk team actually cares about.

**Anything not inside an episode is `is_fraud = 0`.** There is no third class, no "suspicious
but unlabelled" bucket. Ambiguity in labels would make the precision/recall report meaningless.

---

## 2. What is deliberately NOT a spike (the confounders)

This is the most important section in the document. If fraud were the only source of bursts,
new devices, odd hours, or unusual amounts, then a single `if` statement would score a perfect
F1 and the model would be theatre. The generator therefore injects each individual fraud signal
into legitimate traffic as well, at rates that overlap the fraud distribution:

1. **Legitimate shopping sessions.** A real user checking out repeatedly — 3–6 transactions in
   2–10 minutes. These occur on ~6% of active account-days. Burst velocity alone therefore
   cannot separate the classes.
2. **Legitimate device churn.** Users buy new phones, reinstall browsers, clear cookies. A new
   `device_id` appears on a legitimate account with ~3% probability per transaction, and unlike
   fraud devices it *persists* into that account's later traffic.
3. **Legitimate night owls.** Each account has its own preferred activity hours; some accounts
   genuinely transact at 03:00. Hour-of-day is only anomalous *relative to that account*.
4. **Legitimate large purchases.** Amounts are lognormal per account with heavy right tails, so
   a single large transaction is unremarkable and overlaps the fraud amount range.
5. **Legitimate category exploration.** Accounts occasionally transact in a category they have
   never used before.
6. **Legitimate travel.** Each account takes 0-3 trips of 1-5 days to another city and transacts
   normally while away. Without this, "city != home city" is a *perfect* fraud tell -- the first
   data audit measured it at 100% precision, and it was fixed before any model was trained.

The model's job is to learn the **conjunction** — burst *and* unseen non-persistent device *and*
amount ramp *and* account-relative category rarity — not any single marker.

---

## 3. Leakage controls

These are the answers to "how do we know you didn't cheat", and they are binding:

- **No generator-side column may become a feature.** `episode_id`, `is_fraud`, and any injection
  bookkeeping live in a separate label frame. The feature builder is given only the columns a
  real payment processor would have at authorisation time: `txn_id, account_id, timestamp,
  amount, merchant_id, merchant_category, device_id, city, channel`.
- **All features are strictly backward-looking.** Every rolling aggregate over an account is
  computed on transactions with `timestamp < t` for the transaction at time `t`. No centred
  windows, no full-dataset group statistics, no target encoding.
- **The split is temporal**, matching how the system would actually deploy: train on the past,
  predict the future.
  - Days 0–39 → **train** (22,775 txns, 460 fraud, 79 episodes)
  - Days 40–49 → **validation** (5,704 txns, 135 fraud, 23 episodes) — all threshold and
    hyperparameter choice happens here and nowhere else
  - Days 50–end → **held-out test** (5,857 txns, 113 fraud, 18 episodes), scored exactly once
    at the end of Phase 3
  - Transactions straddling a boundary are assigned by their own timestamp; an account may
    appear in more than one split, which is realistic — its features are still causal.
- **Secondary robustness check (Phase 3, if time allows):** an account-disjoint split, where no
  account appears in both train and test. This tests generalisation to unseen accounts and
  guards against the model memorising account identity. Reported alongside, not instead of.
- **The test set is written once and its hash recorded.** `reports/test_set_hash.txt` pins it so
  that any accidental re-generation is detectable.

---

## 4. Volume and class balance

| Quantity | Target |
|---|---|
| Accounts | 600 |
| Period | 60 days |
| Total transactions | ~34,300 |
| Compromised accounts | 20% of accounts, one episode each |
| Fraud transactions | 1.5–2.5% of all rows (actual: 2.06%) |

**Honest caveat on the compromise rate.** Twenty percent of accounts suffering a takeover in
60 days is far above any real portfolio. It is set deliberately high so that the held-out test
window contains ~18 fraud episodes rather than ~4 — below roughly a dozen, episode-level recall
swings by double digits on a single episode and any number we reported would be noise. The
quantity that actually governs the precision/recall trade-off is the **transaction-level
prevalence**, and that is held at a realistic 2.06%. We state this rather than quietly tuning
it away.

A trivial "always predict legitimate" baseline therefore scores ~98% accuracy and **0% recall** —
which is exactly why accuracy is not reported as a headline metric anywhere in this project.

---

## 5. Metrics contract

Reported on the held-out test set, once:

- **Precision, recall, F1** at the operating threshold chosen on validation
- **ROC-AUC** and **PR-AUC** (PR-AUC is the honest one under 2% prevalence; both are shown)
- **Episode-level recall** — fraction of fraud episodes with at least one transaction flagged
- **Confusion matrix** in raw counts, not percentages
- **Baselines for comparison:** (a) always-legitimate, (b) a hand-written velocity rule
  (≥4 transactions in 10 minutes), (c) logistic regression. The gradient-boosted model has to
  beat the rule to justify existing.

---

## Changelog

- 2026-08-29 — initial specification, frozen before any data was generated.
- 2026-08-29 — **city leak found and fixed.** The first `audit_data` run measured
  "city != account's usual city" at **100% precision / 60.9% recall** — legitimate accounts never
  left home, so that one column was a perfect classifier. Added the legitimate-travel confounder
  (section 2, item 6) and regenerated; the marker now sits at 16.8% precision. No model had been
  trained at the point this was caught.
- 2026-08-29 — device novelty redefined as *age of the device on the account*, not first
  appearance. First-appearance fires only on an episode's opening transaction, which is never yet
  inside a burst, so it could never combine with a velocity signal. This shaped the Phase 2
  feature set.
- 2026-08-29 — scale raised from 400 to 600 accounts and the compromise rate from 8% to 20%, to
  put enough fraud episodes in the test window to report on. See the caveat in section 4.
- 2026-08-29 — the dataset spans a 61st day holding 6 transactions: one legitimate checkout
  session that began at 23:5x on day 59 and ran past midnight. Realistic, left in place.
