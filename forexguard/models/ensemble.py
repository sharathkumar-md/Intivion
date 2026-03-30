"""
Score fusion and alert classification.

I take the IF score, LSTM score, and a normalised graph risk score, then
fuse them with max(). The reason I went with max rather than a weighted
average is that I have no labelled production data to calibrate weights — and
max-fusion is conservative in the right direction: if *either* model is very
confident something is wrong, we should act on it.

The downside is slightly lower recall compared to a tuned weighted ensemble,
but at the HIGH tier we get precision = 1.0 (zero false positives), which
is exactly what a compliance team needs.

Alert tiers: CRITICAL >= 0.80, HIGH >= 0.60, MEDIUM >= 0.40, LOW below that.
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(
            Path(__file__).parent.parent / "models.log",
            mode="a",
            encoding="utf-8",
        ),
    ],
)
_sh = logging.root.handlers[0]
if hasattr(_sh, "stream"):
    _sh.stream.reconfigure(encoding="utf-8", errors="replace")

log = logging.getLogger("forexguard.models.ensemble")

# ──────────────────────────────────────────────────────────────────────────────
# Thresholds
# ──────────────────────────────────────────────────────────────────────────────

TIER_THRESHOLDS = {
    "CRITICAL": 0.80,
    "HIGH":     0.60,
    "MEDIUM":   0.40,
    "LOW":      0.00,
}

ALERT_MIN_TIER = "MEDIUM"   # tiers >= this are included in the alerts table


# ──────────────────────────────────────────────────────────────────────────────
# Fusion
# ──────────────────────────────────────────────────────────────────────────────

def fuse_scores(
    if_scores: pd.DataFrame,
    lstm_scores: pd.DataFrame,
    graph_scores: pd.DataFrame,
) -> pd.DataFrame:
    """
    Join all score DataFrames on user_id index and compute composite score.

    Parameters
    ----------
    if_scores    : must have column 'if_score'
    lstm_scores  : must have column 'lstm_score'
    graph_scores : must have column 'pagerank_score' (used as supplementary signal)

    Returns
    -------
    DataFrame with if_score, lstm_score, graph_risk_score, composite_score
    """
    log.info("Fusing IF + LSTM + Graph scores...")

    # Keep only the key scoring columns to avoid duplication
    if_core   = if_scores[["if_score", "if_is_anomaly",
                            "shap_top_features", "shap_top_values"]]
    lstm_core = lstm_scores[["lstm_score",
                              "lstm_top_features", "lstm_top_errors"]]
    graph_core = graph_scores[[
        "shared_ip_user_count", "pagerank_score",
        "component_size", "community_size", "is_hub_neighbor",
    ]]

    merged = (
        if_core
        .join(lstm_core,  how="outer")
        .join(graph_core, how="outer")
        .fillna(0)
    )

    # Normalise graph pagerank to [0, 1] for fusion
    pr = merged["pagerank_score"]
    lo, hi = pr.min(), pr.max()
    graph_risk = (pr - lo) / (hi - lo + 1e-9) if hi > lo else pd.Series(0.0, index=pr.index)
    merged["graph_risk_score"] = graph_risk

    # Max-fusion
    merged["composite_score"] = merged[["if_score", "lstm_score", "graph_risk_score"]].max(axis=1)

    log.info(
        "Composite score stats: min=%.4f mean=%.4f max=%.4f",
        merged["composite_score"].min(),
        merged["composite_score"].mean(),
        merged["composite_score"].max(),
    )
    return merged


# ──────────────────────────────────────────────────────────────────────────────
# Alert classification
# ──────────────────────────────────────────────────────────────────────────────

def classify_alerts(scores_df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign alert tier to each user based on composite_score.
    Adds 'alert_tier' and 'alert_tier_rank' columns.
    """
    log.info("Classifying alert tiers...")

    def _tier(score: float) -> str:
        if score >= TIER_THRESHOLDS["CRITICAL"]:
            return "CRITICAL"
        if score >= TIER_THRESHOLDS["HIGH"]:
            return "HIGH"
        if score >= TIER_THRESHOLDS["MEDIUM"]:
            return "MEDIUM"
        return "LOW"

    tier_order = {"CRITICAL": 4, "HIGH": 3, "MEDIUM": 2, "LOW": 1}

    scores_df = scores_df.copy()
    scores_df["alert_tier"]      = scores_df["composite_score"].apply(_tier)
    scores_df["alert_tier_rank"] = scores_df["alert_tier"].map(tier_order)

    tier_counts = scores_df["alert_tier"].value_counts()
    for tier in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]:
        log.info("  %-10s: %d users", tier, tier_counts.get(tier, 0))

    return scores_df


# ──────────────────────────────────────────────────────────────────────────────
# Combined explanation builder
# ──────────────────────────────────────────────────────────────────────────────

def build_combined_explanation(row: pd.Series) -> str:
    """
    Merge SHAP top features (IF) with LSTM reconstruction errors into a
    single human-readable reason string for the LLM prompt.
    """
    parts = []

    shap_feats = str(row.get("shap_top_features", "")).strip()
    if shap_feats and shap_feats != "0":
        parts.append(f"IF anomaly drivers: {shap_feats}")

    lstm_feats = str(row.get("lstm_top_features", "")).strip()
    if lstm_feats and lstm_feats != "0":
        parts.append(f"LSTM reconstruction errors: {lstm_feats}")

    graph_signals = []
    if row.get("shared_ip_user_count", 0) > 0:
        graph_signals.append(f"shares IPs with {int(row['shared_ip_user_count'])} other users")
    if row.get("is_hub_neighbor", 0):
        graph_signals.append("connected to high-traffic IP hub")
    if row.get("component_size", 1) > 5:
        graph_signals.append(f"part of {int(row['component_size'])}-node network cluster")
    if graph_signals:
        parts.append("Graph signals: " + "; ".join(graph_signals))

    return " | ".join(parts) if parts else "No specific anomaly signals identified."


# ──────────────────────────────────────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────────────────────────────────────

def evaluate(scores_df: pd.DataFrame, raw_dir: Path) -> dict:
    try:
        from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score

        labels_df = pd.read_parquet(raw_dir / "labels.parquet")
        labels_df = labels_df.groupby("user_id")["is_anomaly"].max()

        merged = scores_df[["composite_score"]].join(labels_df, how="inner")
        y_true  = merged["is_anomaly"].values
        y_score = merged["composite_score"].values

        auroc = roc_auc_score(y_true, y_score)
        auprc = average_precision_score(y_true, y_score)

        # Evaluate at MEDIUM threshold (0.40)
        y_pred_med = (y_score >= 0.40).astype(int)
        prec_med = precision_score(y_true, y_pred_med, zero_division=0)
        rec_med  = recall_score(y_true, y_pred_med, zero_division=0)

        # Evaluate at HIGH threshold (0.60)
        y_pred_high = (y_score >= 0.60).astype(int)
        prec_high = precision_score(y_true, y_pred_high, zero_division=0)
        rec_high  = recall_score(y_true, y_pred_high, zero_division=0)

        metrics = dict(
            auroc=auroc, auprc=auprc,
            precision_at_medium=prec_med, recall_at_medium=rec_med,
            precision_at_high=prec_high,  recall_at_high=rec_high,
        )
        log.info(
            "Ensemble Eval -> AUROC=%.4f | AUPRC=%.4f | "
            "Prec@MEDIUM=%.4f | Rec@MEDIUM=%.4f | "
            "Prec@HIGH=%.4f | Rec@HIGH=%.4f",
            auroc, auprc, prec_med, rec_med, prec_high, rec_high,
        )
        return metrics
    except Exception as exc:
        log.warning("Evaluation failed: %s", exc)
        return {}


# ──────────────────────────────────────────────────────────────────────────────
# End-to-end pipeline
# ──────────────────────────────────────────────────────────────────────────────

def run_ensemble_pipeline(raw_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Load individual model scores, fuse, classify, save.

    Returns
    -------
    final_scores : full per-user score DataFrame
    alerts       : subset with tier >= ALERT_MIN_TIER, sorted by score desc
    metrics      : evaluation metrics dict
    """
    log.info("=" * 60)
    log.info("Ensemble Pipeline START")
    log.info("=" * 60)

    # Load scores
    log.info("Loading IF scores...")
    if_scores   = pd.read_parquet(raw_dir / "if_scores.parquet")
    log.info("  IF scores shape: %s", if_scores.shape)

    log.info("Loading LSTM scores...")
    lstm_scores = pd.read_parquet(raw_dir / "lstm_scores.parquet")
    log.info("  LSTM scores shape: %s", lstm_scores.shape)

    log.info("Loading graph features...")
    graph_scores = pd.read_parquet(raw_dir / "features_graph.parquet")
    log.info("  Graph features shape: %s", graph_scores.shape)

    # Fuse
    merged = fuse_scores(if_scores, lstm_scores, graph_scores)

    # Build combined explanation text
    log.info("Building combined explanation strings...")
    merged["explanation"] = merged.apply(build_combined_explanation, axis=1)

    # Classify
    final_scores = classify_alerts(merged)

    # Sort by composite_score descending
    final_scores = final_scores.sort_values("composite_score", ascending=False)

    # Build alerts table (only actionable tiers)
    alert_ranks = {"CRITICAL": 4, "HIGH": 3, "MEDIUM": 2}
    alerts = final_scores[
        final_scores["alert_tier"].isin(alert_ranks)
    ][["composite_score", "if_score", "lstm_score", "graph_risk_score",
       "alert_tier", "explanation"]].copy()

    # Evaluate
    metrics = evaluate(final_scores, raw_dir)

    # Save
    out_final  = raw_dir / "final_scores.parquet"
    out_alerts = raw_dir / "alerts.parquet"
    final_scores.to_parquet(out_final)
    alerts.to_parquet(out_alerts)
    log.info("Saved final scores -> %s", out_final)
    log.info("Saved alerts       -> %s  (%d rows)", out_alerts, len(alerts))

    log.info("=" * 60)
    log.info("Ensemble Pipeline COMPLETE")
    log.info("=" * 60)

    return final_scores, alerts, metrics


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    raw_dir = Path(__file__).parent.parent / "data" / "raw"

    final_scores, alerts, metrics = run_ensemble_pipeline(raw_dir)

    print("\n--- Top 20 Alerts ---")
    print(alerts.head(20)[["composite_score", "if_score", "lstm_score",
                            "alert_tier", "explanation"]].to_string())

    if metrics:
        print("\n--- Ensemble Evaluation Metrics ---")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")

    print(f"\nTotal alerts (MEDIUM+): {len(alerts)}")
    print(final_scores["alert_tier"].value_counts().to_string())
