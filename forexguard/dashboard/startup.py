# Runs the pipeline on first launch if pre-computed data doesn't exist.
# Runs inline (not subprocess) so errors surface in Streamlit logs.

import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_RAW_DIR = Path("/tmp/forexguard/raw")
_FINAL_SCORES = _RAW_DIR / "final_scores.parquet"


def ensure_data():
    if _FINAL_SCORES.exists():
        return

    import streamlit as st
    import pandas as pd
    import numpy as np
    import pickle

    st.info("First launch — generating data and training models (~5 min). Please wait...")

    # ── Step 1: Generate data ─────────────────────────────────────────────────
    with st.spinner("Generating synthetic dataset..."):
        from forexguard.data.generate import generate_dataset
        generate_dataset()

    events_df = pd.read_parquet(_RAW_DIR / "events.parquet")

    # ── Step 2: Tabular features ──────────────────────────────────────────────
    with st.spinner("Building tabular features..."):
        from forexguard.features.tabular import build_tabular_features
        feats = build_tabular_features(events_df)
        feats.to_parquet(_RAW_DIR / "features_tabular.parquet")

    # ── Step 3: Sequences ─────────────────────────────────────────────────────
    with st.spinner("Building LSTM sequences..."):
        from forexguard.features.sequences import build_sequences
        seqs, uids, scaler = build_sequences(events_df)
        np.save(_RAW_DIR / "sequences.npy", seqs)
        np.save(_RAW_DIR / "sequence_user_ids.npy", np.array(uids))
        with open(_RAW_DIR / "seq_scaler.pkl", "wb") as f:
            pickle.dump(scaler, f)

    # ── Step 4: Graph features ────────────────────────────────────────────────
    with st.spinner("Building graph features..."):
        from forexguard.features.graph import build_graph_features
        graph_feats, G = build_graph_features(events_df)
        graph_feats.to_parquet(_RAW_DIR / "features_graph.parquet")
        with open(_RAW_DIR / "user_graph.gpickle", "wb") as f:
            pickle.dump(G, f)

    # ── Step 5: Isolation Forest ──────────────────────────────────────────────
    with st.spinner("Training Isolation Forest..."):
        from forexguard.models.isolation_forest import run_isolation_forest_pipeline
        models_dir = _RAW_DIR.parent / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        run_isolation_forest_pipeline(_RAW_DIR, models_dir)

    # ── Step 6: LSTM Autoencoder ──────────────────────────────────────────────
    with st.spinner("Training LSTM Autoencoder..."):
        from forexguard.models.lstm_autoencoder import run_lstm_pipeline
        run_lstm_pipeline(_RAW_DIR, models_dir)

    # ── Step 7: Ensemble ──────────────────────────────────────────────────────
    with st.spinner("Running ensemble scoring..."):
        from forexguard.models.ensemble import run_ensemble_pipeline
        run_ensemble_pipeline(_RAW_DIR)

    # ── Step 8: LLM summaries (best-effort) ───────────────────────────────────
    with st.spinner("Generating risk summaries..."):
        try:
            from forexguard.llm.risk_summary import generate_batch_summaries
            alerts_df = pd.read_parquet(_RAW_DIR / "alerts.parquet")
            summaries = generate_batch_summaries(alerts_df, max_users=20)
            summaries.to_parquet(_RAW_DIR / "llm_summaries.parquet")
        except Exception:
            pass  # LLM summaries are optional

    st.success("Setup complete! Reloading...")
    st.rerun()
