"""
MLflow experiment tracking for ForexGuard.

Logs both model runs (IF and LSTM) with their params, AUROC/AUPRC metrics,
and artifact paths. I'm using the file-store backend locally (forexguard/mlruns)
which you can view with `mlflow ui`. In production you'd point this at a
Postgres/S3 backend.

Run standalone after training:
  python forexguard/tracking/mlflow_setup.py
"""

import logging
import os
import pickle
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd

from forexguard.log_utils import setup_logger
log = setup_logger("forexguard.tracking.mlflow", "forexguard_tracking.log")

BASE_DIR   = Path(__file__).parent.parent
RAW_DIR    = BASE_DIR / "data" / "raw"
MODELS_DIR = BASE_DIR / "data" / "models"

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", f"file:///{BASE_DIR / 'mlruns'}")
EXPERIMENT_NAME     = "ForexGuard-AnomalyDetection"


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _get_labels(raw_dir: Path) -> pd.Series | None:
    try:
        labels_df = pd.read_parquet(raw_dir / "labels.parquet")
        return labels_df.groupby("user_id")["is_anomaly"].max()
    except Exception:
        return None


def _compute_metrics(scores: pd.Series, labels: pd.Series) -> dict:
    """Compute AUROC, AUPRC for a score series."""
    from sklearn.metrics import roc_auc_score, average_precision_score
    merged = pd.concat([scores, labels], axis=1).dropna()
    y_true  = merged.iloc[:, 1].values
    y_score = merged.iloc[:, 0].values
    return {
        "auroc": float(roc_auc_score(y_true, y_score)),
        "auprc": float(average_precision_score(y_true, y_score)),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Log Isolation Forest run
# ──────────────────────────────────────────────────────────────────────────────

def log_isolation_forest_run(raw_dir: Path = RAW_DIR, models_dir: Path = MODELS_DIR) -> str:
    """Log IF training run to MLflow. Returns run_id."""

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name="IsolationForest") as run:
        log.info("MLflow run ID: %s", run.info.run_id)

        # Parameters
        mlflow.log_params({
            "model":          "IsolationForest",
            "n_estimators":   300,
            "contamination":  0.10,
            "random_state":   42,
            "n_features":     54,
            "n_users":        5000,
        })

        # Metrics from scores
        labels = _get_labels(raw_dir)
        if_scores = pd.read_parquet(raw_dir / "if_scores.parquet")["if_score"]
        if labels is not None:
            metrics = _compute_metrics(if_scores, labels)
            mlflow.log_metrics(metrics)
            log.info("IF metrics: %s", metrics)

        # Ensemble metrics
        try:
            final_scores = pd.read_parquet(raw_dir / "final_scores.parquet")
            comp_scores  = final_scores["composite_score"]
            if labels is not None:
                ens_metrics = _compute_metrics(comp_scores, labels)
                mlflow.log_metrics({f"ensemble_{k}": v for k, v in ens_metrics.items()})
                log.info("Ensemble metrics: %s", ens_metrics)

            tier_counts = final_scores["alert_tier"].value_counts().to_dict()
            mlflow.log_metrics({f"alerts_{k.lower()}": v for k, v in tier_counts.items()})
        except Exception as exc:
            log.warning("Could not log ensemble metrics: %s", exc)

        # Artifacts
        mlflow.log_artifact(str(models_dir / "if_model.pkl"),  artifact_path="models")
        mlflow.log_artifact(str(models_dir / "if_scaler.pkl"), artifact_path="models")
        mlflow.log_artifact(str(raw_dir / "if_scores.parquet"), artifact_path="scores")

        log.info("IF run logged: %s", run.info.run_id)
        return run.info.run_id


# ──────────────────────────────────────────────────────────────────────────────
# Log LSTM run
# ──────────────────────────────────────────────────────────────────────────────

def log_lstm_run(raw_dir: Path = RAW_DIR, models_dir: Path = MODELS_DIR) -> str:
    """Log LSTM Autoencoder training run to MLflow. Returns run_id."""

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with open(models_dir / "lstm_config.pkl", "rb") as f:
        lstm_cfg = pickle.load(f)

    with mlflow.start_run(run_name="LSTM-Autoencoder") as run:
        log.info("MLflow run ID: %s", run.info.run_id)

        mlflow.log_params({
            "model":        "LSTMAutoencoder",
            "hidden_size":  lstm_cfg.get("hidden_size", 64),
            "num_layers":   lstm_cfg.get("num_layers", 2),
            "dropout":      lstm_cfg.get("dropout", 0.2),
            "seq_len":      lstm_cfg.get("seq_len", 20),
            "input_size":   lstm_cfg.get("input_size", 21),
            "epochs":       lstm_cfg.get("epochs", 40),
            "lr":           lstm_cfg.get("lr", 1e-3),
            "batch_size":   lstm_cfg.get("batch_size", 256),
        })

        labels = _get_labels(raw_dir)
        lstm_scores = pd.read_parquet(raw_dir / "lstm_scores.parquet")["lstm_score"]
        if labels is not None:
            metrics = _compute_metrics(lstm_scores, labels)
            mlflow.log_metrics(metrics)
            log.info("LSTM metrics: %s", metrics)

        mlflow.log_artifact(str(models_dir / "lstm_ae.pt"),      artifact_path="models")
        mlflow.log_artifact(str(models_dir / "lstm_config.pkl"), artifact_path="models")
        mlflow.log_artifact(str(raw_dir / "lstm_scores.parquet"), artifact_path="scores")

        log.info("LSTM run logged: %s", run.info.run_id)
        return run.info.run_id


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    log.info("Logging runs to: %s", MLFLOW_TRACKING_URI)

    if_run_id   = log_isolation_forest_run()
    lstm_run_id = log_lstm_run()

    log.info("Logged IF run   : %s", if_run_id)
    log.info("Logged LSTM run : %s", lstm_run_id)
    log.info("To view: mlflow ui --backend-store-uri %s", MLFLOW_TRACKING_URI)
