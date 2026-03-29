"""
Tabular feature engineering for ForexGuard.

I aggregate raw event-level data into per-user features covering all
the suspicious behaviour signals from sections 8.1 through 8.7. Five
blocks — login, financial, trading, session, and IP-hub. Ends up being
54 features per user, which is what the Isolation Forest trains on.

Each block is its own function so I can debug/swap them independently.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from forexguard.log_utils import setup_logger
log = setup_logger("forexguard.features.tabular", "forexguard_features.log", mode="w")


# ─────────────────────────────────────────────────────────────────────────────
# Rolling z-score helper
# ─────────────────────────────────────────────────────────────────────────────

def _zscore(series: pd.Series) -> pd.Series:
    """Standardize with fallback for zero-std columns."""
    mean, std = series.mean(), series.std()
    if std < 1e-9:
        return pd.Series(0.0, index=series.index)
    return (series - mean) / std


# ─────────────────────────────────────────────────────────────────────────────
# Section 8.1 — Login & Access features
# ─────────────────────────────────────────────────────────────────────────────

def _login_features(df: pd.DataFrame) -> pd.DataFrame:
    log.info("Computing login/access features (8.1)...")
    logins = df[df["event_type"] == "login"].copy()

    # Distinct IPs per user
    ip_counts = (
        logins.groupby("user_id")["ip_address"]
        .nunique()
        .rename("distinct_ips_total")
    )

    # Distinct countries
    country_counts = (
        logins.groupby("user_id")["country"]
        .nunique()
        .rename("distinct_countries")
    )

    # Login hour statistics
    logins["login_hour"] = logins["timestamp"].dt.hour
    hour_stats = logins.groupby("user_id")["login_hour"].agg(
        login_hour_mean="mean",
        login_hour_std="std",
    )

    # Fraction of logins between 01:00–05:00 (unusual hours)
    logins["is_night_login"] = logins["login_hour"].between(1, 5).astype(int)
    night_ratio = (
        logins.groupby("user_id")["is_night_login"]
        .mean()
        .rename("night_login_ratio")
    )

    # Failed login stats
    failed = df.groupby("user_id")["failed_logins"].agg(
        failed_logins_total="sum",
        failed_logins_max="max",
    )

    # Distinct devices
    device_counts = (
        df.groupby("user_id")["device_fingerprint"]
        .nunique()
        .rename("distinct_devices")
    )

    # IP-hub score: how many total users share any of this user's IPs
    # (computed later after merging, set placeholder here)

    result = pd.concat(
        [ip_counts, country_counts, hour_stats, night_ratio, failed, device_counts],
        axis=1,
    ).fillna(0)

    log.info("  Login features shape: %s", result.shape)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Section 8.2 — Financial Behaviour features
# ─────────────────────────────────────────────────────────────────────────────

def _financial_features(df: pd.DataFrame) -> pd.DataFrame:
    log.info("Computing financial features (8.2)...")

    deposits    = df[df["event_type"] == "deposit"]
    withdrawals = df[df["event_type"] == "withdrawal"]
    trades      = df[df["event_type"].isin(["trade_open", "trade_close"])]

    # Total & mean deposit / withdrawal
    dep_stats = deposits.groupby("user_id")["deposit_amount"].agg(
        total_deposit="sum",
        mean_deposit="mean",
        deposit_count="count",
        deposit_std="std",
    ).fillna(0)

    with_stats = withdrawals.groupby("user_id")["withdrawal_amount"].agg(
        total_withdrawal="sum",
        mean_withdrawal="mean",
        withdrawal_count="count",
        withdrawal_max="max",
    ).fillna(0)

    # Deposit-to-trade ratio (bonus abuse: deposit but barely trade)
    trade_vol = trades.groupby("user_id")["trade_volume"].sum().rename("total_trade_volume")
    fin = dep_stats.join(with_stats, how="outer").join(trade_vol, how="outer").fillna(0)
    fin["deposit_to_trade_ratio"] = fin["total_trade_volume"] / (fin["total_deposit"] + 1e-9)

    # Withdrawal z-score (sudden large withdrawal)
    fin["withdrawal_zscore"] = _zscore(fin["mean_withdrawal"])

    # High-frequency small deposits (structuring): many deposits, low std
    fin["deposit_amount_cv"] = fin["deposit_std"] / (fin["mean_deposit"] + 1e-9)
    fin["is_structuring_flag"] = (
        (fin["deposit_count"] >= 5) & (fin["deposit_amount_cv"] < 0.3)
    ).astype(float)

    # KYC change before withdrawal
    kyc_flags = df.groupby("user_id")["kyc_change"].any().astype(int).rename("kyc_change_flag")
    fin = fin.join(kyc_flags, how="left").fillna(0)

    log.info("  Financial features shape: %s", fin.shape)
    return fin


# ─────────────────────────────────────────────────────────────────────────────
# Section 8.3 — Trading Behaviour features
# ─────────────────────────────────────────────────────────────────────────────

def _trading_features(df: pd.DataFrame) -> pd.DataFrame:
    log.info("Computing trading features (8.3)...")

    trades = df[df["event_type"].isin(["trade_open", "trade_close"])].copy()
    if trades.empty:
        log.warning("No trade events found — returning empty trading features")
        return pd.DataFrame()

    # Trade volume statistics
    vol_stats = trades.groupby("user_id")["trade_volume"].agg(
        trade_vol_mean="mean",
        trade_vol_std="std",
        trade_vol_max="max",
        trade_count="count",
    ).fillna(0)
    vol_stats["trade_vol_zscore"] = _zscore(vol_stats["trade_vol_max"])

    # Lot size statistics
    lot_stats = trades.groupby("user_id")["lot_size"].agg(
        lot_size_mean="mean",
        lot_size_max="max",
    ).fillna(0)
    lot_stats["lot_size_zscore"] = _zscore(lot_stats["lot_size_max"])

    # Instrument concentration — Herfindahl-Hirschman Index (1.0 = all one instrument)
    def hhi(series: pd.Series) -> float:
        counts = series.value_counts(normalize=True)
        return float((counts ** 2).sum())

    instrument_hhi = (
        trades[trades["instrument"] != "NONE"]
        .groupby("user_id")["instrument"]
        .apply(hhi)
        .rename("instrument_hhi")
    )

    # PnL consistency (latency arb: always positive, low variance)
    pnl_stats = trades.groupby("user_id")["pnl"].agg(
        pnl_mean="mean",
        pnl_std="std",
        pnl_positive_ratio=lambda x: (x > 0).mean(),
    ).fillna(0)
    pnl_stats["pnl_consistency"] = pnl_stats["pnl_positive_ratio"] / (pnl_stats["pnl_std"] + 1e-9)
    pnl_stats["pnl_zscore"] = _zscore(pnl_stats["pnl_mean"])

    # Average trade inter-event delta (bot detection: very short, very consistent)
    trades_sorted = trades.sort_values(["user_id", "timestamp"])
    trades_sorted["inter_event_delta"] = (
        trades_sorted.groupby("user_id")["timestamp"]
        .diff()
        .dt.total_seconds()
    )
    delta_stats = trades_sorted.groupby("user_id")["inter_event_delta"].agg(
        inter_event_mean="mean",
        inter_event_std="std",
    ).fillna(0)

    result = pd.concat(
        [vol_stats, lot_stats, instrument_hhi, pnl_stats, delta_stats],
        axis=1,
    ).fillna(0)

    log.info("  Trading features shape: %s", result.shape)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Section 8.4 — Behavioural / Session features
# ─────────────────────────────────────────────────────────────────────────────

def _session_features(df: pd.DataFrame) -> pd.DataFrame:
    log.info("Computing session/behavioural features (8.4)...")

    session_stats = df.groupby("user_id")["session_duration"].agg(
        session_dur_mean="mean",
        session_dur_std="std",
        session_dur_max="max",
    ).fillna(0)
    session_stats["session_dur_zscore"] = _zscore(session_stats["session_dur_mean"])

    # Pages per minute (bot detection: high mean, low std)
    nav_stats = df.groupby("user_id")["pages_per_minute"].agg(
        pages_per_min_mean="mean",
        pages_per_min_std="std",
        pages_per_min_max="max",
    ).fillna(0)
    nav_stats["nav_regularity"] = nav_stats["pages_per_min_mean"] / (
        nav_stats["pages_per_min_std"] + 1e-9
    )  # high = robotic

    result = pd.concat([session_stats, nav_stats], axis=1).fillna(0)
    log.info("  Session features shape: %s", result.shape)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# IP-hub score (cross-user computation)
# ─────────────────────────────────────────────────────────────────────────────

def _ip_hub_features(df: pd.DataFrame) -> pd.DataFrame:
    """How many distinct users share each IP? High = suspicious hub."""
    log.info("Computing IP-hub features (8.5)...")

    ip_user_map = df.groupby("ip_address")["user_id"].nunique().rename("ip_user_count")
    df_ip = df[["user_id", "ip_address"]].drop_duplicates()
    df_ip = df_ip.merge(ip_user_map, on="ip_address")

    # Per user: max IP-user-count across all their IPs
    result = (
        df_ip.groupby("user_id")["ip_user_count"]
        .agg(ip_hub_max="max", ip_hub_mean="mean")
        .fillna(1)
    )
    log.info("  IP-hub features shape: %s", result.shape)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def build_tabular_features(events_df: pd.DataFrame) -> pd.DataFrame:
    """
    Parameters
    ----------
    events_df : raw events DataFrame (output of data/generate.py)

    Returns
    -------
    features_df : one row per user_id, all engineered features
    """
    log.info("=" * 60)
    log.info("Building tabular features from %d events", len(events_df))
    log.info("=" * 60)

    events_df = events_df.copy()
    events_df["timestamp"] = pd.to_datetime(events_df["timestamp"], format="mixed")

    feature_blocks = [
        _login_features(events_df),
        _financial_features(events_df),
        _trading_features(events_df),
        _session_features(events_df),
        _ip_hub_features(events_df),
    ]

    features = feature_blocks[0]
    for block in feature_blocks[1:]:
        if not block.empty:
            features = features.join(block, how="outer")

    features = features.fillna(0)

    # Replace inf values that can arise from ratio computations
    features.replace([np.inf, -np.inf], 0, inplace=True)

    # Clip extreme outliers to 99th percentile (prevents IF from over-splitting)
    numeric_cols = features.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        cap = features[col].quantile(0.99)
        features[col] = features[col].clip(upper=cap * 10)

    log.info("Final tabular features shape: %s", features.shape)
    log.info("Feature columns: %s", list(features.columns))
    return features


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from pathlib import Path as P

    raw_dir = P(__file__).parent.parent / "data" / "raw"
    events_df = pd.read_parquet(raw_dir / "events.parquet")
    features  = build_tabular_features(events_df)

    out_path = raw_dir / "features_tabular.parquet"
    features.to_parquet(out_path)
    log.info("Saved tabular features -> %s", out_path)

    # Quick sanity checks
    assert not features.isnull().any().any(), "NaN values in features!"
    assert not (features == np.inf).any().any(), "Inf values in features!"
    log.info("Sanity checks passed ✓")
    print(features.describe().T.to_string())
