"""
Builds per-user event sequences for the LSTM Autoencoder.

The LSTM needs sequences, not flat feature vectors, so I take each user's
last 20 events sorted by timestamp and encode them as a (20, 21) array.
Each timestep = 10 event-type one-hot + 2 sin/cos hour encoding + 9 numeric
features. Users with fewer than 20 events get zero-padded at the start.

The scaler is fit on the full dataset's numeric columns and reused at
inference time — I pickle it alongside the model weights.
"""

import logging

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from forexguard.log_utils import setup_logger
log = setup_logger("forexguard.features.sequences", "forexguard_features.log")

# Event types → integer index for one-hot-style encoding
EVENT_TYPE_MAP = {
    "login": 0, "logout": 1, "trade_open": 2, "trade_close": 3,
    "deposit": 4, "withdrawal": 5, "kyc_update": 6,
    "support_ticket": 7, "account_modify": 8, "doc_upload": 9,
}
N_EVENT_TYPES = len(EVENT_TYPE_MAP)
SEQ_LEN       = 20      # timesteps per sequence
SEQ_FEATURES  = [       # raw numeric features used in each timestep
    "trade_volume", "lot_size", "pnl", "deposit_amount",
    "withdrawal_amount", "session_duration", "pages_per_minute",
    "margin_used", "failed_logins",
]


def _encode_event_type(event_type: str) -> np.ndarray:
    """One-hot encode a single event type."""
    vec = np.zeros(N_EVENT_TYPES, dtype=np.float32)
    idx = EVENT_TYPE_MAP.get(event_type, 0)
    vec[idx] = 1.0
    return vec


def _hour_of_day_encoding(hour: int) -> np.ndarray:
    """Sine/cosine encoding of hour to preserve cyclical nature."""
    return np.array([
        np.sin(2 * np.pi * hour / 24),
        np.cos(2 * np.pi * hour / 24),
    ], dtype=np.float32)


def build_user_sequence(
    user_events: pd.DataFrame,
    scaler: StandardScaler,
    seq_len: int = SEQ_LEN,
) -> np.ndarray:
    """
    Build a (seq_len, n_features) array for a single user.
    Pads with zeros if user has fewer than seq_len events.
    Truncates to the last seq_len events if longer.

    Feature vector per timestep = [
        one_hot(event_type),    # 10 dims
        sin/cos(hour),           # 2 dims
        numeric_features,        # len(SEQ_FEATURES) dims
    ] → total = 10 + 2 + 9 = 21 dims
    """
    user_events = user_events.sort_values("timestamp").reset_index(drop=True)

    # Truncate to last seq_len events
    if len(user_events) > seq_len:
        user_events = user_events.iloc[-seq_len:]

    rows = []
    for _, row in user_events.iterrows():
        event_vec  = _encode_event_type(row["event_type"])
        hour_vec   = _hour_of_day_encoding(row["timestamp"].hour)
        num_vals   = np.array([row[f] for f in SEQ_FEATURES], dtype=np.float32)
        step_vec   = np.concatenate([event_vec, hour_vec, num_vals])
        rows.append(step_vec)

    seq = np.array(rows, dtype=np.float32)

    # Pad with zeros at the beginning if sequence is short
    if len(seq) < seq_len:
        pad = np.zeros((seq_len - len(seq), seq.shape[1]), dtype=np.float32)
        seq = np.vstack([pad, seq])

    # Scale the numeric portion only (columns 12 onward)
    numeric_start = N_EVENT_TYPES + 2
    seq[:, numeric_start:] = scaler.transform(seq[:, numeric_start:])
    return seq


def build_sequences(
    events_df: pd.DataFrame,
    seq_len: int = SEQ_LEN,
    fit_scaler: bool = True,
    scaler: StandardScaler | None = None,
) -> tuple[np.ndarray, list[str], StandardScaler]:
    """
    Build sequences for ALL users.

    Returns
    -------
    sequences  : (n_users, seq_len, n_seq_features) ndarray
    user_ids   : ordered list of user_ids corresponding to sequences[i]
    scaler     : fitted StandardScaler (reuse for inference)
    """
    log.info("=" * 60)
    log.info("Building LSTM sequences from %d events", len(events_df))
    log.info("SEQ_LEN=%d, SEQ_FEATURES=%d", seq_len, len(SEQ_FEATURES))

    events_df = events_df.copy()
    events_df["timestamp"] = pd.to_datetime(events_df["timestamp"], format="mixed")

    # Fit scaler on numeric portion across all events
    if fit_scaler:
        scaler = StandardScaler()
        scaler.fit(events_df[SEQ_FEATURES].fillna(0).values)
        log.info("Fitted StandardScaler on %d events", len(events_df))
    elif scaler is None:
        raise ValueError("Must provide a fitted scaler when fit_scaler=False")

    user_ids  = sorted(events_df["user_id"].unique())
    n_users   = len(user_ids)
    n_feats   = N_EVENT_TYPES + 2 + len(SEQ_FEATURES)

    sequences = np.zeros((n_users, seq_len, n_feats), dtype=np.float32)

    for i, uid in enumerate(user_ids):
        user_events = events_df[events_df["user_id"] == uid]
        sequences[i] = build_user_sequence(user_events, scaler, seq_len)
        if (i + 1) % 500 == 0:
            log.info("  Processed %d / %d users", i + 1, n_users)

    log.info("Sequences built. Shape: %s", sequences.shape)
    return sequences, user_ids, scaler


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import pickle

    raw_dir = Path(__file__).parent.parent / "data" / "raw"
    events_df = pd.read_parquet(raw_dir / "events.parquet")

    sequences, user_ids, scaler = build_sequences(events_df)

    out_seq   = raw_dir / "sequences.npy"
    out_uids  = raw_dir / "sequence_user_ids.npy"
    out_scaler = raw_dir / "seq_scaler.pkl"

    np.save(out_seq, sequences)
    np.save(out_uids, np.array(user_ids))
    with open(out_scaler, "wb") as f:
        pickle.dump(scaler, f)

    log.info("Saved sequences  -> %s", out_seq)
    log.info("Saved user_ids   -> %s", out_uids)
    log.info("Saved scaler     -> %s", out_scaler)
    log.info("Sequence shape: %s  (users x steps x features)", sequences.shape)

    # Sanity
    assert sequences.shape == (len(user_ids), SEQ_LEN, N_EVENT_TYPES + 2 + len(SEQ_FEATURES))
    assert not np.isnan(sequences).any(), "NaN in sequences!"
    log.info("Sanity checks passed [OK]")
