"""
LSTM Autoencoder — the advanced sequence model for ForexGuard.

The idea: train the autoencoder on all users (including anomalous ones,
since we have no labels). Normal users reconstruct their sequences well.
Anomalous users — whose behaviour doesn't fit the patterns the model learned —
reconstruct badly. High MSE = anomaly.

Architecture is a standard seq2seq LSTM:
  encoder reads the 20-step input -> produces a context vector
  decoder takes that context vector repeated T times -> reconstructs the sequence
  linear layer maps hidden_size back to input_size (21 features)

For explainability, after reconstruction I compute per-feature MSE averaged
across the 20 timesteps. Whatever feature has the highest reconstruction error
is what made the user look anomalous — e.g., if withdrawal_amount has 53x
higher error than normal, that's your signal.

I dropped VAE because of posterior collapse + tricky threshold tuning.
LSTM AE is more predictable and gets AUROC 0.9159 which is still strong.
"""

import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from forexguard.log_utils import setup_logger
log = setup_logger("forexguard.models.lstm_autoencoder", "forexguard_models.log")

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

LSTM_CFG = dict(
    hidden_size  = 64,
    num_layers   = 2,
    dropout      = 0.2,
    batch_size   = 256,
    lr           = 1e-3,
    epochs       = 40,
    patience     = 5,          # early-stopping patience
    device       = "cpu",      # flip to "cuda" if GPU available
)

TOP_FEAT_K = 5   # top-k features to surface in explanation


# ──────────────────────────────────────────────────────────────────────────────
# Model definition
# ──────────────────────────────────────────────────────────────────────────────

class LSTMAutoencoder(nn.Module):
    """
    Sequence-to-sequence LSTM autoencoder.
    Encoder reads the input sequence and produces a context vector.
    Decoder reconstructs the sequence from the context vector.
    """

    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers  = num_layers

        self.encoder = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.decoder = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.output_layer = nn.Linear(hidden_size, input_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len, input_size)
        returns: (batch, seq_len, input_size) reconstruction
        """
        batch_size, seq_len, _ = x.shape

        # Encode
        _, (hidden, cell) = self.encoder(x)

        # Repeat context vector as decoder input for each timestep
        decoder_input = hidden[-1].unsqueeze(1).repeat(1, seq_len, 1)
        # decoder_input: (batch, seq_len, hidden_size)

        # Decode
        decoder_out, _ = self.decoder(decoder_input, (hidden, cell))
        # decoder_out: (batch, seq_len, hidden_size)

        reconstruction = self.output_layer(decoder_out)
        # reconstruction: (batch, seq_len, input_size)

        return reconstruction


# ──────────────────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────────────────

def load_sequences(raw_dir: Path) -> tuple[np.ndarray, list[str]]:
    """Load pre-built LSTM sequences and user_ids."""
    log.info("Loading sequences from %s ...", raw_dir)
    sequences = np.load(raw_dir / "sequences.npy")          # (n_users, seq_len, n_feats)
    user_ids  = np.load(raw_dir / "sequence_user_ids.npy", allow_pickle=True).tolist()
    log.info("Sequences shape: %s | Users: %d", sequences.shape, len(user_ids))
    return sequences, user_ids


def make_dataloaders(
    sequences: np.ndarray,
    batch_size: int,
    val_frac: float = 0.1,
) -> tuple[DataLoader, DataLoader]:
    """Split sequences into train / val and wrap in DataLoaders."""
    n = len(sequences)
    n_val   = int(n * val_frac)
    n_train = n - n_val

    # Shuffle indices deterministically
    rng = np.random.default_rng(seed=42)
    idx = rng.permutation(n)
    train_idx = idx[:n_train]
    val_idx   = idx[n_train:]

    X_train = torch.tensor(sequences[train_idx], dtype=torch.float32)
    X_val   = torch.tensor(sequences[val_idx],   dtype=torch.float32)

    train_ds = TensorDataset(X_train)
    val_ds   = TensorDataset(X_val)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)

    log.info("Train batches: %d | Val batches: %d", len(train_dl), len(val_dl))
    return train_dl, val_dl


# ──────────────────────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────────────────────

def train_lstm_ae(
    model: LSTMAutoencoder,
    train_dl: DataLoader,
    val_dl: DataLoader,
    epochs: int,
    lr: float,
    patience: int,
    device: str,
) -> list[float]:
    """
    Train with Adam + MSELoss, early stopping on val loss.
    Returns list of (epoch, train_loss, val_loss) tuples.
    """
    model = model.to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    no_improve    = 0
    history       = []

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        train_loss = 0.0
        for (batch,) in train_dl:
            batch = batch.to(device)
            optimiser.zero_grad()
            recon = model(batch)
            loss  = criterion(recon, batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimiser.step()
            train_loss += loss.item() * len(batch)
        train_loss /= len(train_dl.dataset)

        # Validate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for (batch,) in val_dl:
                batch = batch.to(device)
                recon = model(batch)
                val_loss += criterion(recon, batch).item() * len(batch)
        val_loss /= len(val_dl.dataset)

        history.append((epoch, train_loss, val_loss))
        log.info(
            "Epoch %3d/%d | train_loss=%.6f | val_loss=%.6f",
            epoch, epochs, train_loss, val_loss,
        )

        # Early stopping
        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            no_improve    = 0
            # Save best weights
            best_weights = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            no_improve += 1
            if no_improve >= patience:
                log.info("Early stopping at epoch %d (no improvement for %d epochs)", epoch, patience)
                break

    # Restore best weights
    model.load_state_dict(best_weights)
    log.info("Best val_loss: %.6f", best_val_loss)
    return history


# ──────────────────────────────────────────────────────────────────────────────
# Scoring + Explainability
# ──────────────────────────────────────────────────────────────────────────────

def compute_lstm_scores(
    model: LSTMAutoencoder,
    sequences: np.ndarray,
    user_ids: list[str],
    device: str,
    batch_size: int = 512,
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    For each user compute:
      - lstm_score          : mean per-timestep MSE (anomaly score, normalised [0,1])
      - per_feature_mse     : mean MSE per feature across the sequence
    Returns (scores_df, per_feature_mse array).
    """
    log.info("Computing LSTM reconstruction errors for %d users...", len(sequences))
    model.eval()
    all_mse      = []    # scalar per user
    all_feat_mse = []   # (n_features,) per user

    X_tensor = torch.tensor(sequences, dtype=torch.float32)
    ds = DataLoader(TensorDataset(X_tensor), batch_size=batch_size, shuffle=False)

    with torch.no_grad():
        for (batch,) in ds:
            batch = batch.to(device)
            recon = model(batch)
            # Per-sample, per-timestep, per-feature squared error
            se = (recon - batch) ** 2                     # (B, T, F)
            # Mean over time -> per-sample, per-feature MSE
            feat_mse = se.mean(dim=1).cpu().numpy()       # (B, F)
            # Mean over features too -> scalar MSE per sample
            scalar_mse = feat_mse.mean(axis=1)            # (B,)
            all_mse.append(scalar_mse)
            all_feat_mse.append(feat_mse)

    all_mse      = np.concatenate(all_mse)
    all_feat_mse = np.vstack(all_feat_mse)

    # Normalise scalar scores to [0, 1]
    lo, hi = all_mse.min(), all_mse.max()
    norm_scores = (all_mse - lo) / (hi - lo + 1e-9)

    df = pd.DataFrame({
        "user_id":    user_ids,
        "lstm_score": norm_scores,
    }).set_index("user_id")

    log.info(
        "LSTM scores: min=%.4f mean=%.4f max=%.4f",
        norm_scores.min(), norm_scores.mean(), norm_scores.max(),
    )
    return df, all_feat_mse


def build_lstm_explanation(
    per_feature_mse: np.ndarray,
    user_ids: list[str],
    seq_feature_names: list[str],
    top_k: int = TOP_FEAT_K,
) -> pd.DataFrame:
    """
    From per-feature reconstruction MSE, build human-readable explanations.
    Returns DataFrame with lstm_top_features and lstm_top_errors columns.
    """
    log.info("Building LSTM feature explanations...")
    top_feats_list = []
    top_errs_list  = []

    for i in range(len(user_ids)):
        feat_err = per_feature_mse[i]
        top_idx  = np.argsort(feat_err)[::-1][:top_k]
        feats    = [seq_feature_names[j] for j in top_idx]
        errs     = [float(feat_err[j]) for j in top_idx]
        top_feats_list.append(", ".join(feats))
        top_errs_list.append(", ".join(f"{e:.4f}" for e in errs))

    df = pd.DataFrame({
        "user_id":          user_ids,
        "lstm_top_features": top_feats_list,
        "lstm_top_errors":   top_errs_list,
    }).set_index("user_id")

    # Also store per-feature MSE columns
    feat_cols = pd.DataFrame(
        per_feature_mse,
        index=user_ids,
        columns=[f"lstm_feat_mse_{f}" for f in seq_feature_names],
    )
    feat_cols.index.name = "user_id"
    return df.join(feat_cols)


# ──────────────────────────────────────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────────────────────────────────────

def evaluate(scores_df: pd.DataFrame, raw_dir: Path) -> dict:
    try:
        from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score

        labels_df = pd.read_parquet(raw_dir / "labels.parquet")
        labels_df = labels_df.groupby("user_id")["is_anomaly"].max()

        merged = scores_df[["lstm_score"]].join(labels_df, how="inner")
        y_true  = merged["is_anomaly"].values
        y_score = merged["lstm_score"].values

        auroc = roc_auc_score(y_true, y_score)
        auprc = average_precision_score(y_true, y_score)
        prec  = precision_score(y_true, (y_score >= 0.5).astype(int), zero_division=0)
        rec   = recall_score(y_true, (y_score >= 0.5).astype(int), zero_division=0)

        metrics = dict(auroc=auroc, auprc=auprc, precision_at_0_5=prec, recall_at_0_5=rec)
        log.info(
            "LSTM Evaluation -> AUROC=%.4f | AUPRC=%.4f | Prec@0.5=%.4f | Rec@0.5=%.4f",
            auroc, auprc, prec, rec,
        )
        return metrics
    except Exception as exc:
        log.warning("Evaluation failed: %s", exc)
        return {}


# ──────────────────────────────────────────────────────────────────────────────
# Sequence feature names (must match sequences.py SEQ_FEATURES order)
# ──────────────────────────────────────────────────────────────────────────────

EVENT_TYPES = [
    "login", "logout", "trade_open", "trade_close", "deposit",
    "withdrawal", "kyc_update", "support_ticket", "account_modify", "doc_upload",
]
SEQ_NUMERICS = [
    "trade_volume", "lot_size", "pnl", "deposit_amount",
    "withdrawal_amount", "session_duration", "pages_per_minute",
    "margin_used", "failed_logins",
]
SEQ_FEATURE_NAMES = (
    [f"evt_{e}" for e in EVENT_TYPES]
    + ["hour_sin", "hour_cos"]
    + SEQ_NUMERICS
)   # length 21


# ──────────────────────────────────────────────────────────────────────────────
# End-to-end pipeline
# ──────────────────────────────────────────────────────────────────────────────

def run_lstm_pipeline(raw_dir: Path, models_dir: Path) -> tuple[pd.DataFrame, dict]:
    log.info("=" * 60)
    log.info("LSTM Autoencoder Pipeline START")
    log.info("=" * 60)

    models_dir.mkdir(parents=True, exist_ok=True)
    device = LSTM_CFG["device"]
    if torch.cuda.is_available():
        device = "cuda"
        log.info("CUDA available — using GPU")

    # 1. Load sequences
    sequences, user_ids = load_sequences(raw_dir)
    n_users, seq_len, input_size = sequences.shape
    log.info("input_size=%d | seq_len=%d", input_size, seq_len)

    # 2. DataLoaders
    train_dl, val_dl = make_dataloaders(sequences, batch_size=LSTM_CFG["batch_size"])

    # 3. Build model
    model = LSTMAutoencoder(
        input_size=input_size,
        hidden_size=LSTM_CFG["hidden_size"],
        num_layers=LSTM_CFG["num_layers"],
        dropout=LSTM_CFG["dropout"],
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info("Model parameters: %d", n_params)

    # 4. Train
    history = train_lstm_ae(
        model, train_dl, val_dl,
        epochs=LSTM_CFG["epochs"],
        lr=LSTM_CFG["lr"],
        patience=LSTM_CFG["patience"],
        device=device,
    )

    # 5. Score all users
    scores_df, per_feature_mse = compute_lstm_scores(
        model, sequences, user_ids, device=device,
    )

    # 6. Explanations
    expl_df = build_lstm_explanation(per_feature_mse, user_ids, SEQ_FEATURE_NAMES)

    # 7. Merge
    result = scores_df.join(expl_df)

    # 8. Save
    out_scores = raw_dir / "lstm_scores.parquet"
    result.to_parquet(out_scores)
    log.info("Saved LSTM scores -> %s", out_scores)

    torch.save(model.state_dict(), models_dir / "lstm_ae.pt")
    config = {**LSTM_CFG, "input_size": input_size, "seq_len": seq_len,
              "seq_feature_names": SEQ_FEATURE_NAMES}
    with open(models_dir / "lstm_config.pkl", "wb") as f:
        pickle.dump(config, f)
    log.info("Saved model + config -> %s", models_dir)

    # 9. Evaluate
    metrics = evaluate(scores_df, raw_dir)

    log.info("=" * 60)
    log.info("LSTM Autoencoder Pipeline COMPLETE")
    log.info("=" * 60)

    return result, metrics


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    base_dir   = Path(__file__).parent.parent
    raw_dir    = base_dir / "data" / "raw"
    models_dir = base_dir / "data" / "models"

    result, metrics = run_lstm_pipeline(raw_dir, models_dir)

    print("\n--- Top LSTM anomalies ---")
    top20 = result.sort_values("lstm_score", ascending=False).head(20)
    print(top20[["lstm_score", "lstm_top_features", "lstm_top_errors"]].to_string())

    if metrics:
        print("\n--- Evaluation Metrics ---")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")
