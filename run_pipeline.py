"""
Runs the full ForexGuard pipeline end-to-end.

Steps:
  1. generate synthetic data
  2. tabular features
  3. LSTM sequences
  4. graph features
  5. train Isolation Forest + SHAP
  6. train LSTM Autoencoder
  7. ensemble + alert classification
  8. LLM risk summaries
  9. log everything to MLflow

I added skip flags so I don't have to regenerate 54k events every time
I'm iterating on the model side.

  python run_pipeline.py                  # full run
  python run_pipeline.py --skip-data      # reuse existing events.parquet
  python run_pipeline.py --skip-features  # reuse pre-computed features
  python run_pipeline.py --skip-training  # reuse trained models (just re-score/log)
"""

import argparse
import logging
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
_sh = logging.root.handlers[0]
if hasattr(_sh, "stream"):
    _sh.stream.reconfigure(encoding="utf-8", errors="replace")
log = logging.getLogger("forexguard.pipeline")

BASE_DIR   = Path(__file__).parent / "forexguard"
RAW_DIR    = BASE_DIR / "data" / "raw"
MODELS_DIR = BASE_DIR / "data" / "models"


def _step(name: str):
    """Context manager / banner for pipeline steps."""
    class Step:
        def __enter__(self):
            log.info("=" * 60)
            log.info("STEP: %s", name)
            log.info("=" * 60)
            self.start = time.time()
        def __exit__(self, *_):
            elapsed = time.time() - self.start
            log.info("DONE: %s (%.1f s)", name, elapsed)
    return Step()


def run(args):
    # ── Step 1: Data generation ───────────────────────────────────────────────
    if not args.skip_data:
        with _step("1. Generate synthetic data"):
            from forexguard.data.generate import generate_dataset
            generate_dataset()
    else:
        log.info("Skipping data generation (--skip-data)")

    # ── Step 2–4: Feature engineering ────────────────────────────────────────
    if not args.skip_features:
        with _step("2. Build tabular features"):
            import pandas as pd
            from forexguard.features.tabular import build_tabular_features
            events_df = pd.read_parquet(RAW_DIR / "events.parquet")
            feats = build_tabular_features(events_df)
            feats.to_parquet(RAW_DIR / "features_tabular.parquet")

        with _step("3. Build LSTM sequences"):
            from forexguard.features.sequences import build_sequences
            import numpy as np, pickle
            seqs, uids, scaler = build_sequences(events_df)
            np.save(RAW_DIR / "sequences.npy", seqs)
            np.save(RAW_DIR / "sequence_user_ids.npy", np.array(uids))
            with open(RAW_DIR / "seq_scaler.pkl", "wb") as f:
                pickle.dump(scaler, f)

        with _step("4. Build graph features"):
            from forexguard.features.graph import build_graph_features
            import pickle
            graph_feats, G = build_graph_features(events_df)
            graph_feats.to_parquet(RAW_DIR / "features_graph.parquet")
            with open(RAW_DIR / "user_graph.gpickle", "wb") as f:
                pickle.dump(G, f)
    else:
        log.info("Skipping feature engineering (--skip-features)")

    # ── Step 5–7: Modelling ───────────────────────────────────────────────────
    if not args.skip_training:
        with _step("5. Train Isolation Forest + SHAP"):
            from forexguard.models.isolation_forest import run_isolation_forest_pipeline
            run_isolation_forest_pipeline(RAW_DIR, MODELS_DIR)

        with _step("6. Train LSTM Autoencoder"):
            from forexguard.models.lstm_autoencoder import run_lstm_pipeline
            run_lstm_pipeline(RAW_DIR, MODELS_DIR)

        with _step("7. Ensemble + alert classification"):
            from forexguard.models.ensemble import run_ensemble_pipeline
            run_ensemble_pipeline(RAW_DIR)
    else:
        log.info("Skipping model training (--skip-training)")

    # ── Step 8: LLM summaries ─────────────────────────────────────────────────
    with _step("8. Generate LLM risk summaries"):
        import pandas as pd
        from forexguard.llm.risk_summary import generate_batch_summaries
        alerts_df = pd.read_parquet(RAW_DIR / "alerts.parquet")
        summaries = generate_batch_summaries(alerts_df, max_users=20)
        summaries.to_parquet(RAW_DIR / "llm_summaries.parquet")

    # ── Step 9: MLflow ────────────────────────────────────────────────────────
    with _step("9. Log to MLflow"):
        from forexguard.tracking.mlflow_setup import log_isolation_forest_run, log_lstm_run
        log_isolation_forest_run(RAW_DIR, MODELS_DIR)
        log_lstm_run(RAW_DIR, MODELS_DIR)

    log.info("=" * 60)
    log.info("PIPELINE COMPLETE")
    log.info("=" * 60)
    log.info("Next steps:")
    log.info("  API server  : uvicorn forexguard.api.app:app --reload")
    log.info("  Dashboard   : streamlit run forexguard/dashboard/app.py")
    log.info("  MLflow UI   : mlflow ui --backend-store-uri forexguard/mlruns")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ForexGuard full pipeline runner")
    parser.add_argument("--skip-data",     action="store_true", help="Skip synthetic data generation")
    parser.add_argument("--skip-features", action="store_true", help="Skip feature engineering")
    parser.add_argument("--skip-training", action="store_true", help="Skip model training")
    args = parser.parse_args()
    run(args)
