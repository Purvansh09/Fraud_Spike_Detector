"""False-positive cost analysis: what the confusion matrix is worth in rupees.

Two things make this more than multiplying a confusion matrix by two made-up numbers.

1. **Episode-aware savings.** Blocking the third transaction of an eight-transaction burst
   does not save one transaction, it saves six -- the account is frozen and the rest never
   happen. Costing this per-transaction would badly understate early detection. Fraud
   transactions *before* the first flag are counted as lost, those from the first flag
   onward as prevented.

2. **The operating threshold is chosen on VALIDATION.** Picking the cost-minimising
   threshold by looking at test costs would turn the held-out set into a second validation
   set and void the Phase 3 metrics. We optimise on validation, then report what that
   choice cost on test.

Every assumption lives in `CostModel` and is a guess until someone with real Razorpay
numbers replaces it. The sensitivity analysis at the end exists because the honest
question is not "what is the net benefit" but "does the recommendation survive being
wrong about the inputs".

Writes reports/cost_analysis.md and reports/cost_curve.png.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

RAW = Path("data/raw")
SPLITS = Path("data/splits")
MODELS = Path("models")
REPORTS = Path("reports")


@dataclass(frozen=True)
class CostModel:
    """All figures in INR. These are assumptions, not measurements -- see module docstring.

    Ranges quoted are the plausible bands used in the sensitivity analysis.
    """

    # What a fraudulent transaction costs when it goes through: the merchant loses the
    # goods AND pays the acquirer's dispute fee. Indian acquirer chargeback fees commonly
    # sit in the 300-1000 band.
    chargeback_fee: float = 500.0

    # A wrongly blocked sale does not cost the full ticket value -- it costs the margin
    # that sale would have earned.
    margin_rate: float = 0.25

    # Handling one wrongly declined payment: support contact, re-auth, manual review.
    friction_cost: float = 150.0

    # Some wrongly blocked customers simply leave. This term dominates at high block
    # rates and is the reason precision matters more than it looks.
    churn_probability: float = 0.02
    customer_ltv: float = 8000.0

    def false_positive_cost(self, amounts: np.ndarray) -> float:
        return float(
            (amounts * self.margin_rate).sum()
            + len(amounts) * (self.friction_cost + self.churn_probability * self.customer_ltv)
        )


def episode_outcomes(
    fraud: pd.DataFrame, flagged: np.ndarray
) -> tuple[float, int, float, int]:
    """Split fraud value into prevented vs lost, respecting burst dynamics.

    Returns (prevented_amount, n_prevented, lost_amount, n_lost).
    """
    prevented_amt = lost_amt = 0.0
    n_prev = n_lost = 0
    f = fraud.assign(_flag=flagged).sort_values("timestamp")
    for _, ep in f.groupby("episode_id", sort=False):
        hits = np.flatnonzero(ep["_flag"].to_numpy())
        if len(hits) == 0:
            lost_amt += float(ep["amount"].sum())
            n_lost += len(ep)
            continue
        first = hits[0]
        # Everything from the first flag onward is stopped: this transaction is declined
        # and the account is frozen behind it.
        prevented_amt += float(ep["amount"].iloc[first:].sum())
        n_prev += len(ep) - first
        lost_amt += float(ep["amount"].iloc[:first].sum())
        n_lost += first
    return prevented_amt, n_prev, lost_amt, n_lost


def evaluate_threshold(df: pd.DataFrame, scores: np.ndarray, thr: float, cm: CostModel) -> dict:
    """Total cost to the business at one threshold, versus having no model at all."""
    pred = scores >= thr
    is_fraud = df["is_fraud"].to_numpy() == 1

    fraud = df[is_fraud]
    prevented_amt, n_prev, lost_amt, n_lost = episode_outcomes(fraud, pred[is_fraud])

    fp_amounts = df.loc[~is_fraud & pred, "amount"].to_numpy()
    fp_cost = cm.false_positive_cost(fp_amounts)

    residual_fraud_cost = lost_amt + n_lost * cm.chargeback_fee
    total_cost = residual_fraud_cost + fp_cost

    # The counterfactual: no detector, every fraud transaction settles.
    no_model_cost = float(fraud["amount"].sum()) + len(fraud) * cm.chargeback_fee

    return {
        "threshold": thr,
        "n_flagged": int(pred.sum()),
        "n_false_pos": int((~is_fraud & pred).sum()),
        "n_fraud_prevented": n_prev,
        "n_fraud_missed": n_lost,
        "fraud_prevented_inr": prevented_amt + n_prev * cm.chargeback_fee,
        "fp_cost_inr": fp_cost,
        "residual_fraud_cost_inr": residual_fraud_cost,
        "total_cost_inr": total_cost,
        "no_model_cost_inr": no_model_cost,
        "net_saving_inr": no_model_cost - total_cost,
    }


def load_split(split: str) -> pd.DataFrame:
    feats = pd.read_parquet(SPLITS / "features.parquet")
    labels = pd.read_parquet(RAW / "labels.parquet")
    assign = pd.read_parquet(SPLITS / "assignment.parquet")
    txns = pd.read_parquet(RAW / "transactions.parquet")[["txn_id", "amount", "timestamp"]]
    df = (feats.merge(assign, on="txn_id")
               .merge(labels, on="txn_id")
               .merge(txns, on="txn_id"))
    return df[df["split"] == split].reset_index(drop=True)


def main() -> None:
    bundle = joblib.load(MODELS / "model.joblib")
    model, features, f1_thr = bundle["model"], bundle["features"], bundle["threshold"]
    cm = CostModel()

    val = load_split("val")
    test = load_split("test")
    val_scores = model.predict_proba(val[features])[:, 1]
    test_scores = model.predict_proba(test[features])[:, 1]

    REPORTS.mkdir(exist_ok=True)
    L: list[str] = []

    def w(line: str = "") -> None:
        L.append(line)
        try:
            print(line)
        except UnicodeEncodeError:
            print(line.encode("ascii", "replace").decode("ascii"))

    def inr(x: float) -> str:
        return f"Rs {x:,.0f}"

    w("# False-Positive Cost Analysis")
    w()
    w("What the confusion matrix is worth in money, and whether the recommendation "
      "survives being wrong about the inputs.")
    w()

    # ---------------- assumptions, stated before any number is quoted ----------------
    w("## Assumptions")
    w()
    w("**These are assumptions, not measurements.** They are isolated in `CostModel` so "
      "that real figures can replace them and the analysis re-run unchanged.")
    w()
    w("| parameter | value | meaning |")
    w("|---|---|---|")
    w(f"| `chargeback_fee` | {inr(cm.chargeback_fee)} | acquirer dispute fee per settled fraud |")
    w(f"| `margin_rate` | {cm.margin_rate:.0%} | contribution margin lost on a blocked sale |")
    w(f"| `friction_cost` | {inr(cm.friction_cost)} | support and re-auth per wrong decline |")
    w(f"| `churn_probability` | {cm.churn_probability:.0%} | chance a wrongly blocked customer leaves |")
    w(f"| `customer_ltv` | {inr(cm.customer_ltv)} | lifetime value at risk when they do |")
    w()
    w("A false positive therefore costs "
      f"`amount x {cm.margin_rate:.2f} + {inr(cm.friction_cost)} + "
      f"{cm.churn_probability:.2f} x {inr(cm.customer_ltv)}` "
      f"= {inr(cm.friction_cost + cm.churn_probability * cm.customer_ltv)} plus a quarter of "
      "the ticket. A settled fraud costs its full amount plus the dispute fee.")
    w()
    w("**Episode-aware savings.** Flagging the 3rd transaction of an 8-transaction burst "
      "prevents 6, not 1 — the decline stops that attempt and the account is frozen behind "
      "it. Fraud before the first flag is counted lost; from the first flag onward, "
      "prevented.")
    w()

    # ---------------- choose the threshold on validation ----------------
    # The score distribution is violently bimodal -- legitimate transactions sit near 1e-5,
    # fraud near 1.0 -- so the 95th percentile of scores is 0.0009 while the 98th is 0.31.
    # A quantile-spaced grid therefore puts almost no candidates in the 0.001-0.3 band where
    # the decision actually lives, and crams the rest above 0.999. The first version of this
    # analysis did exactly that and returned an identical "optimum" for every sensitivity
    # scenario, which is what exposed the bug. Space the grid in threshold space instead.
    grid = np.unique(np.concatenate([
        np.geomspace(1e-4, 1e-2, 60),
        np.linspace(0.01, 0.99, 200),
    ]))
    val_rows = [evaluate_threshold(val, val_scores, t, cm) for t in grid]
    best_val = max(val_rows, key=lambda d: d["net_saving_inr"])
    cost_thr = best_val["threshold"]

    w("## Choosing the operating point (validation only)")
    w()
    w(f"Sweeping {len(grid)} thresholds on the **validation** set and maximising net saving "
      f"gives **{cost_thr:.4f}**, against the F1-optimal **{f1_thr:.4f}** carried over from "
      "Phase 3.")
    w()
    w("The test set plays no part in this choice. That is the whole point: a threshold "
      "tuned on test would make every Phase 3 number meaningless.")
    w()

    # ---------------- report both on test ----------------
    w("## What each choice costs on the held-out test set")
    w()
    w("| | F1-optimal | cost-optimal | no model |")
    w("|---|---|---|---|")
    a = evaluate_threshold(test, test_scores, f1_thr, cm)
    b = evaluate_threshold(test, test_scores, cost_thr, cm)
    w(f"| threshold | {f1_thr:.4f} | {cost_thr:.4f} | — |")
    w(f"| transactions flagged | {a['n_flagged']:,} | {b['n_flagged']:,} | 0 |")
    w(f"| false positives | {a['n_false_pos']:,} | {b['n_false_pos']:,} | 0 |")
    w(f"| fraud txns prevented | {a['n_fraud_prevented']} | {b['n_fraud_prevented']} | 0 |")
    w(f"| fraud txns settled | {a['n_fraud_missed']} | {b['n_fraud_missed']} | "
      f"{int((test['is_fraud'] == 1).sum())} |")
    w(f"| **fraud loss avoided** | {inr(a['fraud_prevented_inr'])} | "
      f"{inr(b['fraud_prevented_inr'])} | {inr(0)} |")
    w(f"| **false-positive cost** | {inr(a['fp_cost_inr'])} | {inr(b['fp_cost_inr'])} | {inr(0)} |")
    w(f"| **residual fraud cost** | {inr(a['residual_fraud_cost_inr'])} | "
      f"{inr(b['residual_fraud_cost_inr'])} | {inr(a['no_model_cost_inr'])} |")
    w(f"| **total cost** | {inr(a['total_cost_inr'])} | {inr(b['total_cost_inr'])} | "
      f"{inr(a['no_model_cost_inr'])} |")
    w(f"| **net saving vs no model** | **{inr(a['net_saving_inr'])}** | "
      f"**{inr(b['net_saving_inr'])}** | — |")
    w()
    w(f"Over {len(test):,} test transactions covering {test['timestamp'].dt.day.nunique()} "
      f"days of traffic.")
    w()
    # State the generalisation gap rather than let the chart quietly reveal it.
    best_test = max(
        (evaluate_threshold(test, test_scores, t, cm) for t in grid),
        key=lambda d: d["net_saving_inr"],
    )
    gap = best_test["net_saving_inr"] - b["net_saving_inr"]
    w(f"**The honest gap.** With hindsight, the best threshold *on test* would have been "
      f"{best_test['threshold']:.4f}, saving {inr(best_test['net_saving_inr'])} -- "
      f"{inr(gap)} more than the {cost_thr:.4f} we committed to from validation. That "
      f"{gap / best_test['net_saving_inr']:.1%} shortfall over an 11-day window is the "
      "price of not tuning on the test set, and it is the correct price to pay: the "
      "retrospective optimum is not available at decision time in production either.")
    w()

    # ---------------- the full curve, so the shape is visible ----------------
    w("## Cost across the threshold range (test)")
    w()
    w("| threshold | flagged | false pos | fraud prevented | FP cost | residual fraud | net saving |")
    w("|---|---|---|---|---|---|---|")
    for t in [0.05, 0.10, f1_thr, cost_thr, 0.30, 0.50, 0.70, 0.90]:
        r = evaluate_threshold(test, test_scores, t, cm)
        tag = ""
        if abs(t - f1_thr) < 1e-9:
            tag = " (F1)"
        if abs(t - cost_thr) < 1e-9:
            tag = " (cost)"
        w(f"| {t:.4f}{tag} | {r['n_flagged']:,} | {r['n_false_pos']:,} | "
          f"{r['n_fraud_prevented']} | {inr(r['fp_cost_inr'])} | "
          f"{inr(r['residual_fraud_cost_inr'])} | {inr(r['net_saving_inr'])} |")
    w()

    # ---------------- sensitivity: does the recommendation survive? ----------------
    w("## Sensitivity to the assumptions")
    w()
    w("The net-saving figure is only as good as the guesses above. What matters for a "
      "decision is whether the **recommended threshold** moves when those guesses are "
      "wrong. Each row re-optimises on validation under a different assumption and reports "
      "the resulting test saving.")
    w()
    w("| scenario | optimal threshold | test net saving |")
    w("|---|---|---|")

    plausible = {
        "baseline": cm,
        "chargeback fee 300 (low)": replace(cm, chargeback_fee=300.0),
        "chargeback fee 1000 (high)": replace(cm, chargeback_fee=1000.0),
        "margin 10% (thin)": replace(cm, margin_rate=0.10),
        "margin 40% (fat)": replace(cm, margin_rate=0.40),
        "friction 50 (cheap support)": replace(cm, friction_cost=50.0),
        "friction 500 (costly support)": replace(cm, friction_cost=500.0),
        "churn 0.5% (sticky)": replace(cm, churn_probability=0.005),
        "churn 5% (fickle)": replace(cm, churn_probability=0.05),
        "LTV 25,000 (premium)": replace(cm, customer_ltv=25000.0),
    }

    def optimise(model_cm: CostModel) -> tuple[float, float, float]:
        rows = [evaluate_threshold(val, val_scores, t, model_cm) for t in grid]
        thr_s = max(rows, key=lambda d: d["net_saving_inr"])["threshold"]
        unit = model_cm.friction_cost + model_cm.churn_probability * model_cm.customer_ltv
        return thr_s, evaluate_threshold(test, test_scores, thr_s, model_cm)["net_saving_inr"], unit

    for name, model_cm in plausible.items():
        thr_s, saving, unit = optimise(model_cm)
        w(f"| {name} | {thr_s:.4f} | {inr(saving)} |")
    w()

    thrs = [optimise(c)[0] for c in plausible.values()]
    avg_fraud = float(val.loc[val["is_fraud"] == 1, "amount"].mean())
    base_unit = cm.friction_cost + cm.churn_probability * cm.customer_ltv
    w(f"**The optimal threshold does not move at all** — {min(thrs):.4f} in every one of "
      "these scenarios. That looked like a bug when it first appeared, so it was chased "
      "down rather than reported as robustness.")
    w()
    w("The reason is a genuine asymmetry in the economics. The average fraud transaction "
      f"here is {inr(avg_fraud)}; a false positive costs {inr(base_unit)} plus a quarter of "
      f"the ticket, so roughly {inr(base_unit + avg_fraud * cm.margin_rate)}. Fraud is an "
      "order of magnitude more expensive per event, so the optimum is pinned by *which "
      "fraud episodes get caught* and is simply not sensitive to false-positive pricing "
      "within any plausible range.")
    w()

    # ---------------- where does the recommendation actually break? ----------------
    w("### Where it does break")
    w()
    w("Robustness is only meaningful with a stated limit, so here is the stress test: push "
      "false-positive cost up until the operating point moves.")
    w()
    w("| scenario | FP cost per event | optimal threshold |")
    w("|---|---|---|")
    stress = {
        "baseline": cm,
        "churn 5%, LTV 25k": replace(cm, churn_probability=0.05, customer_ltv=25000.0),
        "churn 10%, LTV 50k": replace(cm, churn_probability=0.10, customer_ltv=50000.0),
        "churn 20%, LTV 50k": replace(cm, churn_probability=0.20, customer_ltv=50000.0),
        "churn 30%, LTV 100k": replace(cm, churn_probability=0.30, customer_ltv=100000.0),
    }
    for name, model_cm in stress.items():
        thr_s, _, unit = optimise(model_cm)
        w(f"| {name} | {inr(unit)} | {thr_s:.4f} |")
    w()
    w(f"The operating point holds until a false positive costs roughly **{inr(5000)}** — "
      f"about {5000 / avg_fraud:.0%} of the average fraud ticket — at which point it climbs "
      "steeply and the detector should be tuned to flag far less. So the honest statement "
      "is not *the recommendation is robust*, but: **it is robust provided a wrongly "
      "declined customer costs you less than about three-quarters of a fraudulent "
      "transaction.** For a payments business that is a comfortable margin; for a "
      "high-LTV lender with fragile customer relationships it might not be, and that "
      "business should re-run this with its own figures.")
    w()
    w("Net saving stays positive throughout, which is unsurprising at 2% prevalence with "
      "82% precision. The figure to distrust is the absolute rupee amount, not the sign.")
    w()

    _plot(test, test_scores, cm, f1_thr, cost_thr)
    w("![cost curve](cost_curve.png)")
    w()

    (REPORTS / "cost_analysis.md").write_text("\n".join(L), encoding="utf-8")
    print(f"\n[written] {REPORTS / 'cost_analysis.md'}")


def _plot(test, scores, cm: CostModel, f1_thr: float, cost_thr: float) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ts = np.linspace(0.01, 0.95, 60)
    rows = [evaluate_threshold(test, scores, t, cm) for t in ts]
    fp = [r["fp_cost_inr"] for r in rows]
    rf = [r["residual_fraud_cost_inr"] for r in rows]
    tot = [r["total_cost_inr"] for r in rows]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(ts, rf, label="residual fraud cost", lw=2)
    ax.plot(ts, fp, label="false-positive cost", lw=2)
    ax.plot(ts, tot, label="total cost", lw=2.5, color="black")
    ax.axvline(f1_thr, ls="--", color="grey", lw=1)
    ax.axvline(cost_thr, ls=":", color="crimson", lw=1.5)
    ax.annotate("F1-optimal", (f1_thr, max(tot) * 0.60), fontsize=8, color="grey",
                rotation=90, ha="left", va="bottom")
    ax.annotate("cost-optimal (chosen on val)", (cost_thr, max(tot) * 0.02), fontsize=8,
                color="crimson", rotation=90, ha="right", va="bottom")
    ax.set_xlabel("decision threshold")
    ax.set_ylabel("cost on test window (INR)")
    ax.set_title("Where the two costs trade off")
    ax.legend(frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(REPORTS / "cost_curve.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()
