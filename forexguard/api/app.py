"""
FastAPI app — the main API layer for ForexGuard.

I load all the pre-computed scores at startup so responses are instant
(no model inference on the hot path for the common case). The endpoints
fall back to live inference if you send feature values directly, which
is useful for testing and for real-time scoring from the stream.

Five endpoints:
  GET  /health                — liveness probe
  POST /score                 — IF score + SHAP explanation
  POST /score_seq             — LSTM reconstruction error score
  GET  /graph_risk/{user_id}  — graph risk features
  GET  /alerts                — paginated alert queue with tier filter
  GET  /alerts/{user_id}      — single user's alert details
"""

import logging
import pickle
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from forexguard.log_utils import setup_logger
log = setup_logger("forexguard.api", "forexguard_api.log")

# ──────────────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────────────

BASE_DIR   = Path(__file__).parent.parent
RAW_DIR    = BASE_DIR / "data" / "raw"
MODELS_DIR = BASE_DIR / "data" / "models"

# ──────────────────────────────────────────────────────────────────────────────
# App state — loaded once at startup
# ──────────────────────────────────────────────────────────────────────────────

class AppState:
    final_scores:  pd.DataFrame | None = None
    alerts:        pd.DataFrame | None = None
    graph_feats:   pd.DataFrame | None = None
    if_model     = None
    if_scaler    = None
    if_features: list[str] = []
    lstm_model   = None
    lstm_cfg:    dict = {}


state = AppState()

# ──────────────────────────────────────────────────────────────────────────────
# FastAPI app
# ──────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="ForexGuard Anomaly Detection API",
    description="Real-time user/trader anomaly detection engine for forex brokerages.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    log.info("Loading model artifacts and pre-computed scores...")

    # Final scores + alerts
    state.final_scores = pd.read_parquet(RAW_DIR / "final_scores.parquet")
    state.alerts       = pd.read_parquet(RAW_DIR / "alerts.parquet")
    state.graph_feats  = pd.read_parquet(RAW_DIR / "features_graph.parquet")

    # IF model
    with open(MODELS_DIR / "if_model.pkl",        "rb") as f: state.if_model    = pickle.load(f)
    with open(MODELS_DIR / "if_scaler.pkl",       "rb") as f: state.if_scaler   = pickle.load(f)
    with open(MODELS_DIR / "if_feature_names.pkl","rb") as f: state.if_features = pickle.load(f)

    # LSTM model
    with open(MODELS_DIR / "lstm_config.pkl", "rb") as f:
        state.lstm_cfg = pickle.load(f)

    # Import here to avoid circular import
    from forexguard.models.lstm_autoencoder import LSTMAutoencoder
    lstm = LSTMAutoencoder(
        input_size   = state.lstm_cfg["input_size"],
        hidden_size  = state.lstm_cfg["hidden_size"],
        num_layers   = state.lstm_cfg["num_layers"],
        dropout      = state.lstm_cfg["dropout"],
    )
    lstm.load_state_dict(torch.load(MODELS_DIR / "lstm_ae.pt", map_location="cpu"))
    lstm.eval()
    state.lstm_model = lstm

    log.info(
        "Startup complete. %d users in score table | %d active alerts.",
        len(state.final_scores), len(state.alerts),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Request / Response schemas
# ──────────────────────────────────────────────────────────────────────────────

class TabularScoreRequest(BaseModel):
    """Feature vector for IF scoring (all numeric tabular features)."""
    user_id:  str
    features: dict[str, float] = Field(
        description="Feature name -> value. Missing features default to 0."
    )


class SequenceScoreRequest(BaseModel):
    """Sequence of event feature vectors for LSTM scoring."""
    user_id: str
    # Each element is a flat dict of feature_name -> value
    events: list[dict[str, float]] = Field(
        description="Ordered list of event feature dicts (up to 20 timesteps)."
    )


class ScoreResponse(BaseModel):
    user_id:         str
    score:           float
    is_anomaly:      bool
    alert_tier:      str
    top_features:    str
    explanation:     str


class AlertSummary(BaseModel):
    user_id:         str
    composite_score: float
    if_score:        float
    lstm_score:      float
    alert_tier:      str
    explanation:     str


# ──────────────────────────────────────────────────────────────────────────────
# Helper: lookup pre-computed score
# ──────────────────────────────────────────────────────────────────────────────

def _get_precomputed(user_id: str) -> pd.Series | None:
    if state.final_scores is None:
        return None
    if user_id in state.final_scores.index:
        return state.final_scores.loc[user_id]
    return None


# ──────────────────────────────────────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "users_loaded": len(state.final_scores) if state.final_scores is not None else 0}


@app.post("/score", response_model=ScoreResponse)
async def score_tabular(request: TabularScoreRequest):
    """
    Return IF anomaly score for a user.
    First checks pre-computed results; falls back to live inference
    if the feature dict is provided.
    """
    precomp = _get_precomputed(request.user_id)

    if precomp is not None and not request.features:
        # Return cached result
        return ScoreResponse(
            user_id      = request.user_id,
            score        = float(precomp.get("if_score", 0.0)),
            is_anomaly   = bool(precomp.get("if_is_anomaly", False)),
            alert_tier   = str(precomp.get("alert_tier", "LOW")),
            top_features = str(precomp.get("shap_top_features", "")),
            explanation  = str(precomp.get("explanation", "")),
        )

    # Live inference from provided features
    if not request.features:
        raise HTTPException(status_code=404, detail=f"User {request.user_id} not found in precomputed scores. Provide features for live scoring.")

    feat_vec = np.array(
        [request.features.get(f, 0.0) for f in state.if_features],
        dtype=np.float32,
    ).reshape(1, -1)
    X_scaled = state.if_scaler.transform(feat_vec)
    raw_score = state.if_model.decision_function(X_scaled)[0]
    score = float(np.clip((-raw_score + 0.5) / 1.0, 0.0, 1.0))  # rough normalisation
    is_anomaly = state.if_model.predict(X_scaled)[0] == -1

    tier = "LOW"
    if score >= 0.80: tier = "CRITICAL"
    elif score >= 0.60: tier = "HIGH"
    elif score >= 0.40: tier = "MEDIUM"

    return ScoreResponse(
        user_id      = request.user_id,
        score        = score,
        is_anomaly   = bool(is_anomaly),
        alert_tier   = tier,
        top_features = "live-inference (SHAP not computed)",
        explanation  = "Live inference — run full pipeline for SHAP explanations.",
    )


@app.post("/score_seq", response_model=ScoreResponse)
async def score_sequence(request: SequenceScoreRequest):
    """
    Return LSTM reconstruction-error anomaly score for a user.
    Accepts up to 20 timesteps; pads with zeros if fewer.
    Falls back to pre-computed score if no events provided.
    """
    precomp = _get_precomputed(request.user_id)

    if not request.events:
        if precomp is not None:
            return ScoreResponse(
                user_id      = request.user_id,
                score        = float(precomp.get("lstm_score", 0.0)),
                is_anomaly   = float(precomp.get("lstm_score", 0.0)) >= 0.5,
                alert_tier   = str(precomp.get("alert_tier", "LOW")),
                top_features = str(precomp.get("lstm_top_features", "")),
                explanation  = str(precomp.get("explanation", "")),
            )
        raise HTTPException(status_code=404, detail=f"User {request.user_id} not found. Provide events for live scoring.")

    feat_names = state.lstm_cfg.get("seq_feature_names", [])
    seq_len    = state.lstm_cfg.get("seq_len", 20)
    n_feats    = state.lstm_cfg.get("input_size", 21)

    # Build sequence tensor
    rows = []
    for ev in request.events[-seq_len:]:
        vec = np.array([ev.get(f, 0.0) for f in feat_names], dtype=np.float32)
        rows.append(vec)

    seq = np.array(rows, dtype=np.float32)
    if len(seq) < seq_len:
        pad = np.zeros((seq_len - len(seq), n_feats), dtype=np.float32)
        seq = np.vstack([pad, seq])

    X = torch.tensor(seq, dtype=torch.float32).unsqueeze(0)  # (1, T, F)
    with torch.no_grad():
        recon = state.lstm_model(X)
    mse = float(((recon - X) ** 2).mean().item())

    # Rough normalisation using dataset stats (mean=0.058 scaled to 1.0)
    score = float(np.clip(mse / 0.5, 0.0, 1.0))
    tier = "LOW"
    if score >= 0.80: tier = "CRITICAL"
    elif score >= 0.60: tier = "HIGH"
    elif score >= 0.40: tier = "MEDIUM"

    return ScoreResponse(
        user_id      = request.user_id,
        score        = score,
        is_anomaly   = score >= 0.5,
        alert_tier   = tier,
        top_features = "live-inference",
        explanation  = f"LSTM reconstruction MSE={mse:.4f}",
    )


@app.get("/graph_risk/{user_id}")
async def graph_risk(user_id: str):
    """Return graph-based risk features for a user."""
    if state.graph_feats is None:
        raise HTTPException(status_code=503, detail="Graph features not loaded.")
    if user_id not in state.graph_feats.index:
        raise HTTPException(status_code=404, detail=f"User {user_id} not found in graph features.")
    row = state.graph_feats.loc[user_id]
    return {
        "user_id":              user_id,
        "shared_ip_user_count": int(row.get("shared_ip_user_count", 0)),
        "degree_centrality":    float(row.get("degree_centrality", 0.0)),
        "pagerank_score":       float(row.get("pagerank_score", 0.0)),
        "component_size":       int(row.get("component_size", 1)),
        "community_id":         int(row.get("community_id", -1)),
        "community_size":       int(row.get("community_size", 1)),
        "is_hub_neighbor":      int(row.get("is_hub_neighbor", 0)),
    }


@app.get("/alerts", response_model=list[AlertSummary])
async def get_alerts(
    tier: Literal["ALL", "CRITICAL", "HIGH", "MEDIUM"] = Query("ALL"),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
):
    """Return paginated list of current alerts, optionally filtered by tier."""
    if state.alerts is None:
        raise HTTPException(status_code=503, detail="Alerts not loaded.")

    df = state.alerts.copy()
    if tier != "ALL":
        df = df[df["alert_tier"] == tier]

    df = df.sort_values("composite_score", ascending=False).iloc[offset : offset + limit]

    results = []
    for uid, row in df.iterrows():
        results.append(AlertSummary(
            user_id         = str(uid),
            composite_score = float(row.get("composite_score", 0.0)),
            if_score        = float(row.get("if_score", 0.0)),
            lstm_score      = float(row.get("lstm_score", 0.0)),
            alert_tier      = str(row.get("alert_tier", "MEDIUM")),
            explanation     = str(row.get("explanation", "")),
        ))
    return results


@app.get("/alerts/{user_id}", response_model=AlertSummary)
async def get_user_alert(user_id: str):
    """Return the alert and explanation for a single user."""
    precomp = _get_precomputed(user_id)
    if precomp is None:
        raise HTTPException(status_code=404, detail=f"User {user_id} not found.")

    return AlertSummary(
        user_id         = user_id,
        composite_score = float(precomp.get("composite_score", 0.0)),
        if_score        = float(precomp.get("if_score", 0.0)),
        lstm_score      = float(precomp.get("lstm_score", 0.0)),
        alert_tier      = str(precomp.get("alert_tier", "LOW")),
        explanation     = str(precomp.get("explanation", "")),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(
        "forexguard.api.app:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
    )
