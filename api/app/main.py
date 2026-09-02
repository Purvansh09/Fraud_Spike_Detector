"""Fraud-spike scoring service.

POST a transaction, get back a decision, the factors that drove it, and one sentence of
plain English. The decision is made entirely by the local model; the language layer is
cosmetic and degrades to a deterministic template when no API key is configured.

Run:
    PYTHONPATH=src ./.venv/Scripts/python.exe -m uvicorn api.app.main:app --reload
"""

from __future__ import annotations

import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

try:                                   # optional: only needed to read ANTHROPIC_API_KEY
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

from fraudspike.explain import Explainer, reason_for  # noqa: E402
from fraudspike.serving import TXN_FIELDS, Scorer     # noqa: E402

STATE: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Loading the model, history store and SHAP explainer takes a second or two; doing it
    # per-request would dominate latency.
    STATE["scorer"] = Scorer()
    STATE["explainer"] = Explainer(STATE["scorer"].model, STATE["scorer"].features)
    STATE["txns"] = pd.read_parquet(ROOT / "data/raw/transactions.parquet")
    # Language generation is the slow part; an analyst re-opening a row should be instant.
    STATE["explain_cache"] = {}
    yield
    STATE.clear()


app = FastAPI(
    title="Fraud-Spike Detector",
    description="Velocity-based account-takeover detection. Defence-only.",
    version="1.0.0",
    lifespan=lifespan,
)


class Transaction(BaseModel):
    txn_id: str = Field(..., examples=["TXN0034100"])
    account_id: str = Field(..., examples=["ACC00042"])
    timestamp: str = Field(..., examples=["2026-07-28T03:14:00"])
    amount: float = Field(..., gt=0, examples=[24500.0])
    merchant_id: str = Field(..., examples=["MRC_GIFT_003"])
    merchant_category: str = Field(..., examples=["gift_cards"])
    device_id: str = Field(..., examples=["DEVX_EP0007"])
    city: str = Field(..., examples=["Delhi"])
    channel: str = Field(..., examples=["card"])


class Factor(BaseModel):
    feature: str
    description: str
    value: float | None
    shap: float


class ScoreResponse(BaseModel):
    txn_id: str
    account_id: str
    score: float
    action: str
    threshold_review: float
    threshold_block: float
    top_factors: list[Factor]
    reason: str
    reason_source: str
    latency_ms: float


@app.get("/health")
def health() -> dict:
    scorer: Scorer = STATE["scorer"]
    return {
        "status": "ok",
        "accounts_known": scorer.store.n_accounts(),
        "features": len(scorer.features),
    }


@app.get("/model")
def model_info() -> dict:
    """What is deployed and where the thresholds came from."""
    scorer: Scorer = STATE["scorer"]
    return {
        "thresholds": {
            "review": scorer.threshold_review,
            "block": scorer.threshold_block,
        },
        "threshold_provenance": (
            "Block = cost-optimal on validation (Phase 4). Review = lowest score retaining "
            "95% fraud recall on validation. Neither was chosen on the test set."
        ),
        "held_out_test": {
            "precision": 0.823, "recall": 0.843, "f1": 0.833, "pr_auc": 0.911,
            "episodes_caught": "20/20",
            "note": "Scored once. See reports/test_report.md.",
        },
        "n_features": len(scorer.features),
    }


@app.post("/score", response_model=ScoreResponse)
def score(txn: Transaction, record: bool = True) -> ScoreResponse:
    """Score one transaction and explain the decision.

    `record=true` appends the transaction to the account's history so the next request
    sees it — which is what makes a burst detectable across successive calls.
    """
    scorer: Scorer = STATE["scorer"]
    explainer: Explainer = STATE["explainer"]

    payload = txn.model_dump()
    try:
        payload["timestamp"] = pd.Timestamp(payload["timestamp"])
    except ValueError as exc:
        raise HTTPException(422, f"unparseable timestamp: {exc}") from exc

    t0 = time.perf_counter()
    decision, X = scorer.decide(payload)
    contribs = explainer.top_contributions(X, k=4)
    reason, source = reason_for(decision.action, decision.score, contribs)
    latency = (time.perf_counter() - t0) * 1000

    if record:
        scorer.store.append({k: payload[k] for k in TXN_FIELDS})

    return ScoreResponse(
        txn_id=decision.txn_id,
        account_id=decision.account_id,
        score=decision.score,
        action=decision.action,
        threshold_review=decision.threshold_review,
        threshold_block=decision.threshold_block,
        top_factors=[
            Factor(
                feature=c.feature,
                description=c.display_gloss,
                value=None if pd.isna(c.display_value) else c.display_value,
                shap=c.shap,
            )
            for c in contribs
        ],
        reason=reason,
        reason_source=source,
        latency_ms=round(latency, 1),
    )


@app.get("/", include_in_schema=False)
def dashboard() -> FileResponse:
    return FileResponse(ROOT / "dashboard" / "index.html")


@app.get("/explain/{txn_id}")
def explain(txn_id: str) -> dict:
    """Explain one already-scored transaction, on demand.

    Deliberately separate from the dashboard's table load. Generating language costs a
    2-5 second round trip, so the table renders instantly from local scores and this runs
    only when an analyst opens a specific row. That is also the honest production shape:
    the decision is synchronous, the narrative is not.
    """
    scorer: Scorer = STATE["scorer"]
    explainer: Explainer = STATE["explainer"]

    if txn_id in STATE["explain_cache"]:
        return STATE["explain_cache"][txn_id]

    txns = STATE["txns"]
    row = txns[txns["txn_id"] == txn_id]
    if row.empty:
        raise HTTPException(404, f"unknown txn_id {txn_id}")

    payload = {k: row.iloc[0][k] for k in TXN_FIELDS}
    decision, X = scorer.decide(payload)
    contribs = explainer.top_contributions(X, k=4)
    reason, source = reason_for(decision.action, decision.score, contribs)

    result = {
        "txn_id": txn_id,
        "score": decision.score,
        "action": decision.action,
        "reason": reason,
        "reason_source": source,
        "top_factors": [
            {
                "feature": c.feature,
                "description": c.display_gloss,
                "value": None if pd.isna(c.display_value) else round(float(c.display_value), 4),
                "shap": round(c.shap, 4),
            }
            for c in contribs
        ],
    }
    STATE["explain_cache"][txn_id] = result
    return result


@app.get("/demo/flagged")
def demo_flagged(limit: int = 50) -> dict:
    """Scored test-window transactions with ground truth, for the Phase 6 dashboard.

    Uses the batch feature frame, which `tests/test_serving.py` proves is identical to
    what /score computes -- this endpoint is a faster path to the same numbers, not a
    different model.
    """
    scorer: Scorer = STATE["scorer"]
    feats = pd.read_parquet(ROOT / "data/splits/features.parquet")
    labels = pd.read_parquet(ROOT / "data/raw/labels.parquet")
    assign = pd.read_parquet(ROOT / "data/splits/assignment.parquet")
    txns = pd.read_parquet(ROOT / "data/raw/transactions.parquet")

    df = feats.merge(assign, on="txn_id").merge(labels, on="txn_id")
    df = df[df["split"] == "test"].reset_index(drop=True)
    scores = scorer.model.predict_proba(df[scorer.features])[:, 1]
    df["score"] = scores
    df["action"] = [
        "BLOCK" if s >= scorer.threshold_block
        else "REVIEW" if s >= scorer.threshold_review
        else "ALLOW"
        for s in scores
    ]

    # features.parquet holds only txn_id + features, so account_id comes from the raw frame.
    meta = txns[["txn_id", "account_id", "timestamp", "amount", "merchant_category", "city"]]
    df = df.merge(meta, on="txn_id")

    # Include fraud we MISSED, not just what we flagged. A dashboard that can only show
    # its successes is a sales tool; the false positives and false negatives are the rows
    # a reviewer actually needs to interrogate.
    shown = df[(df["action"] != "ALLOW") | (df["is_fraud"] == 1)]
    flagged = shown.sort_values("score", ascending=False).head(limit)
    tp = int(((df["action"] == "BLOCK") & (df["is_fraud"] == 1)).sum())
    fp = int(((df["action"] == "BLOCK") & (df["is_fraud"] == 0)).sum())
    fn = int(((df["action"] != "BLOCK") & (df["is_fraud"] == 1)).sum())

    # Raw recall undersells a three-band system: fraud landing in REVIEW is still seen by
    # a human. Report what actually happened to every fraud transaction.
    is_f = df["is_fraud"] == 1
    f_blocked = int((is_f & (df["action"] == "BLOCK")).sum())
    f_review = int((is_f & (df["action"] == "REVIEW")).sum())
    f_allowed = int((is_f & (df["action"] == "ALLOW")).sum())
    total_fraud = f_blocked + f_review + f_allowed

    return {
        "running_metrics": {
            "precision": round(tp / (tp + fp), 4) if tp + fp else 0.0,
            "recall": round(tp / (tp + fn), 4) if tp + fn else 0.0,
            "true_positives": tp, "false_positives": fp, "false_negatives": fn,
            "scope": "held-out test window, block band only",
        },
        "band_breakdown": {
            "total_fraud": total_fraud,
            "blocked": f_blocked,
            "review": f_review,
            "allowed": f_allowed,
            "seen_or_blocked_pct": round((f_blocked + f_review) / total_fraud, 4)
            if total_fraud else 0.0,
        },
        "flagged": [
            {
                "txn_id": r.txn_id,
                "account_id": r.account_id,
                "timestamp": str(r.timestamp),
                "amount": float(r.amount),
                "merchant_category": r.merchant_category,
                "city": r.city,
                "score": round(float(r.score), 4),
                "action": r.action,
                "is_fraud": int(r.is_fraud),
                "archetype": r.archetype,
            }
            for r in flagged.itertuples()
        ],
    }
