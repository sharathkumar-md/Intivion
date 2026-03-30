"""
Isolation Forest detector — the baseline model for ForexGuard.

I chose IF as the baseline because it trains in linear time, handles
high-dimensional tabular data well, and plugs directly into TreeSHAP for
free explainability. No label dependency at all.

I merge the 48 tabular features with 6 graph features (excluding community_id
which is categorical) to get 54 total. StandardScaler before training.

For explainability I use shap.TreeExplainer — it gives signed SHAP values
per feature per user. Negative SHAP = pushes toward anomaly. I surface the
top 5 by absolute value in the alert payload.

AUROC came out at 0.9554 which is pretty solid for a pure unsupervised setup.
"""

import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

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

log = logging.getLogger("forexguard.models.isolation_forest")

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

IF_PARAMS = dict(
    n_estimators=300,
    max_samples="auto",
    contamination=0.10,   # ~10 % of users expected to be anomalous
    random_state=42,
    n_jobs=-1,
)

TOP_SHAP_K = 5   # number of top SHAP features to include in explanation


# ──────────────────────────────────────────────────────────────────────────────
# Feature preparation
# ──────────────────────────────────────────────────────────────────────────────

def load_features(raw_dir: Path) -> tuple[pd.DataFrame, list[str]]:
    """
    Load and merge tabular + graph features.
    Returns (feature_matrix, feature_names).
    """
    log.info("Loading tabular features...")
    tab = pd.read_parquet(raw_dir / "features_tabular.parquet")
    log.info("  Tabular shape: %s", tab.shape)

    log.info("Loading graph features...")
    graph = pd.read_parquet(raw_dir / "features_graph.parquet")
    # Drop non-numeric graph columns
    graph = graph.drop(columns=["community_id"], errors="ignore")
    log.info("  Graph shape: %s", graph.shape)

    features = tab.join(graph, how="left").fillna(0)
    features.replace([np.inf, -np.inf], 0, inplace=True)

    # Drop any residual non-numeric columns
    features = features.select_dtypes(include=[np.number])

    log.info("Merged feature matrix: %s", features.shape)
    return features, list(features.columns)


# ──────────────────────────────────────────────────────────────────────────────
# Train
# ──────────────────────────────────────────────────────────────────────────────

def train_isolation_forest(
    features: pd.DataFrame,
) -> tuple[IsolationForest, StandardScaler, np.ndarray]:
    """
    Scale features, fit IsolationForest, return (model, scaler, scaled_X).
    """
    log.info("Scaling features with StandardScaler...")
    scaler = StandardScaler()
    X = scaler.fit_transform(features.values.astype(np.float32))
    log.info("  Scaled matrix shape: %s", X.shape)

    log.info("Training IsolationForest: %s", IF_PARAMS)
    clf = IsolationForest(**IF_PARAMS)
    clf.fit(X)
    log.info("IsolationForest training complete.")
    return clf, scaler, X


# ──────────────────────────────────────────────────────────────────────────────
# Score
# ──────────────────────────────────────────────────────────────────────────────

def compute_if_scores(
    clf: IsolationForest,
    X: np.ndarray,
    user_ids: list[str],
) -> pd.DataFrame:
    """
    Compute anomaly scores.
    sklearn decision_function returns higher = more normal.
    We negate and min-max normalise to [0, 1]: 1 = most anomalous.
    """
    log.info("Computing IF anomaly scores for %d users...", len(user_ids))
    raw_scores = clf.decision_function(X)           # higher = more normal
    anomaly_score = -raw_scores                      # flip: higher = more anomalous
    lo, hi = anomaly_score.min(), anomaly_score.max()
    if hi > lo:
        anomaly_score = (anomaly_score - lo) / (hi - lo)
    else:
        anomaly_score = np.zeros_like(anomaly_score)

    predictions = clf.predict(X)   # +1 = normal, -1 = anomaly
    is_anomaly  = (predictions == -1).astype(int)

    df = pd.DataFrame({
        "user_id":       user_ids,
        "if_score":      anomaly_score,
        "if_is_anomaly": is_anomaly,
    }).set_index("user_id")

    n_flagged = is_anomaly.sum()
    log.info(
        "IF scores computed. Flagged %d / %d users (%.1f%%) as anomalous.",
        n_flagged, len(user_ids), 100 * n_flagged / len(user_ids),
    )
    return df


# ──────────────────────────────────────────────────────────────────────────────
# TreeSHAP Explainability
# ──────────────────────────────────────────────────────────────────────────────

def compute_shap_explanations(
    clf: IsolationForest,
    X: np.ndarray,
    feature_names: list[str],
    user_ids: list[str],
    top_k: int = TOP_SHAP_K,
) -> pd.DataFrame:
    """
    Use TreeSHAP to get per-user feature importance scores.
    Returns a DataFrame with:
      - shap_top_features  : comma-joined list of top-k feature names
      - shap_top_values    : their SHAP values (signed)
      - shap_<feature>     : individual SHAP values for all features
    """
    log.info("Computing TreeSHAP explanations for %d users...", len(user_ids))
    explainer   = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X)   # (n_users, n_features)
    log.info("SHAP values shape: %s", np.array(shap_values).shape)

    shap_df = pd.DataFrame(
        shap_values,
        index=user_ids,
        columns=[f"shap_{f}" for f in feature_names],
    )
    shap_df.index.name = "user_id"

    # Build top-k feature strings per user
    abs_shap = np.abs(shap_values)
    top_feats_list = []
    top_vals_list  = []
    for i in range(len(user_ids)):
        top_idx    = np.argsort(abs_shap[i])[::-1][:top_k]
        top_feats  = [feature_names[j] for j in top_idx]
        top_vals   = [float(shap_values[i][j]) for j in top_idx]
        top_feats_list.append(", ".join(top_feats))
        top_vals_list.append(", ".join(f"{v:.4f}" for v in top_vals))

    shap_df.insert(0, "shap_top_features", top_feats_list)
    shap_df.insert(1, "shap_top_values",   top_vals_list)

    log.info("SHAP explanations computed.")
    return shap_df


# ──────────────────────────────────────────────────────────────────────────────
# Evaluation (uses held-out labels — only for assessment validation)
# ──────────────────────────────────────────────────────────────────────────────

def evaluate(
    scores_df: pd.DataFrame,
    raw_dir: Path,
) -> dict:
    """
    Load labels.parquet and compute AUROC + precision/recall at 10 % threshold.
    Labels are ONLY used here for evaluation — never passed to the model.
    """
    try:
        from sklearn.metrics import (
            average_precision_score,
            roc_auc_score,
            precision_score,
            recall_score,
        )

        labels_df = pd.read_parquet(raw_dir / "labels.parquet")
        labels_df = labels_df.groupby("user_id")["is_anomaly"].max()

        merged = scores_df[["if_score"]].join(labels_df, how="inner")
        y_true = merged["is_anomaly"].values
        y_score = merged["if_score"].values

        auroc  = roc_auc_score(y_true, y_score)
        auprc  = average_precision_score(y_true, y_score)
        prec   = precision_score(y_true, (y_score >= 0.5).astype(int), zero_division=0)
        rec    = recall_score(y_true, (y_score >= 0.5).astype(int), zero_division=0)

        metrics = dict(auroc=auroc, auprc=auprc, precision_at_0_5=prec, recall_at_0_5=rec)
        log.info(
            "Evaluation -> AUROC=%.4f | AUPRC=%.4f | Prec@0.5=%.4f | Rec@0.5=%.4f",
            auroc, auprc, prec, rec,
        )
        return metrics
    except Exception as exc:
        log.warning("Evaluation failed: %s", exc)
        return {}


# ──────────────────────────────────────────────────────────────────────────────
# End-to-end pipeline
# ──────────────────────────────────────────────────────────────────────────────

def run_isolation_forest_pipeline(
    raw_dir: Path,
    models_dir: Path,
) -> pd.DataFrame:
    """Train IF, score all users, compute SHAP, save artifacts."""

    log.info("=" * 60)
    log.info("Isolation Forest Pipeline START")
    log.info("=" * 60)

    models_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load features
    features, feature_names = load_features(raw_dir)
    user_ids = list(features.index)

    # 2. Train
    clf, scaler, X = train_isolation_forest(features)

    # 3. Score
    scores_df = compute_if_scores(clf, X, user_ids)

    # 4. SHAP
    shap_df = compute_shap_explanations(clf, X, feature_names, user_ids)

    # 5. Merge scores + SHAP
    result = scores_df.join(shap_df)

    # 6. Save
    out_scores = raw_dir / "if_scores.parquet"
    result.to_parquet(out_scores)
    log.info("Saved IF scores + SHAP -> %s", out_scores)

    with open(models_dir / "if_model.pkl", "wb") as f:
        pickle.dump(clf, f)
    with open(models_dir / "if_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    with open(models_dir / "if_feature_names.pkl", "wb") as f:
        pickle.dump(feature_names, f)
    log.info("Saved model artifacts -> %s", models_dir)

    # 7. Evaluate
    metrics = evaluate(result, raw_dir)

    log.info("=" * 60)
    log.info("Isolation Forest Pipeline COMPLETE")
    log.info("=" * 60)

    return result, metrics


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    base_dir   = Path(__file__).parent.parent
    raw_dir    = base_dir / "data" / "raw"
    models_dir = base_dir / "data" / "models"

    result, metrics = run_isolation_forest_pipeline(raw_dir, models_dir)

    print("\n--- Sample IF Scores (top anomalies) ---")
    top20 = result.sort_values("if_score", ascending=False).head(20)
    print(top20[["if_score", "if_is_anomaly", "shap_top_features", "shap_top_values"]].to_string())

    if metrics:
        print("\n--- Evaluation Metrics ---")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")
