"""
Generates the synthetic dataset for ForexGuard.

I'm creating ~54k events across 5000 users (4500 normal, 500 anomalous).
The anomalous users have one of 12 different fraud patterns injected into
their behaviour — things like IP hopping, structuring deposits, latency arb,
collusion rings etc.

Importantly, I save the ground truth labels to a *separate* file (labels.parquet)
and never pass them to the models. This keeps the detection genuinely unsupervised
and lets me compute AUROC at the end as an honest evaluation.
"""

import json
import logging
import random
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from faker import Faker

# ── Logging ──────────────────────────────────────────────────────────────────
_stream_handler = logging.StreamHandler(sys.stdout)
_stream_handler.stream.reconfigure(encoding="utf-8", errors="replace")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        _stream_handler,
        logging.FileHandler(Path(__file__).parent / "generate.log", mode="w", encoding="utf-8"),
    ],
)
log = logging.getLogger("forexguard.datagen")

# ── Constants ─────────────────────────────────────────────────────────────────
SEED = 42
N_NORMAL_USERS = 4_500
N_ANOMALOUS_USERS = 500          # 10% anomaly rate
N_EVENTS_PER_USER_MEAN = 10      # → ~50k total events
START_DATE = datetime(2024, 1, 1)
END_DATE   = datetime(2024, 6, 30)

INSTRUMENTS = ["EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD", "USD/CAD",
               "NZD/USD", "EUR/GBP", "XAU/USD", "BTC/USD", "US30"]
EVENT_TYPES = ["login", "logout", "trade_open", "trade_close",
               "deposit", "withdrawal", "kyc_update", "support_ticket",
               "account_modify", "doc_upload"]
COUNTRIES   = ["US", "GB", "DE", "FR", "AU", "SG", "IN", "AE", "JP", "CA"]

OUT_DIR = Path(__file__).parent / "raw"
OUT_DIR.mkdir(parents=True, exist_ok=True)

fake = Faker()
Faker.seed(SEED)
rng  = np.random.default_rng(SEED)
random.seed(SEED)


# ─────────────────────────────────────────────────────────────────────────────
# Helper utilities
# ─────────────────────────────────────────────────────────────────────────────

def rand_ip(country: str | None = None) -> str:
    return fake.ipv4_private() if rng.random() < 0.3 else fake.ipv4_public()


def rand_device() -> str:
    return fake.md5()[:16]          # simulate device fingerprint


def rand_ts(start: datetime, end: datetime) -> datetime:
    delta = (end - start).total_seconds()
    return start + timedelta(seconds=float(rng.uniform(0, delta)))


def rand_ts_hour(start: datetime, end: datetime, hour: int) -> datetime:
    """Return a timestamp with a specific hour of day."""
    ts = rand_ts(start, end)
    return ts.replace(hour=hour, minute=int(rng.integers(0, 60)))


# ─────────────────────────────────────────────────────────────────────────────
# Normal user profile
# ─────────────────────────────────────────────────────────────────────────────

def generate_normal_user(user_id: str) -> dict:
    country = random.choice(COUNTRIES)
    usual_login_hour = int(rng.integers(7, 21))      # business hours
    n_devices = int(rng.integers(1, 3))
    devices = [rand_device() for _ in range(n_devices)]
    n_ips    = int(rng.integers(1, 4))
    ips      = [rand_ip() for _ in range(n_ips)]
    account_balance = float(rng.uniform(1_000, 100_000))
    return {
        "user_id": user_id,
        "country": country,
        "usual_login_hour": usual_login_hour,
        "devices": devices,
        "ips": ips,
        "account_balance": account_balance,
        "kyc_verified": True,
        "anomaly_type": "normal",
    }


def generate_events_for_normal_user(profile: dict) -> list[dict]:
    n_events = max(1, int(rng.poisson(N_EVENTS_PER_USER_MEAN)))
    events = []
    for _ in range(n_events):
        ts = rand_ts(START_DATE, END_DATE)
        # bias login hour to user's usual window ± 2h
        if rng.random() < 0.8:
            h = int(np.clip(
                rng.normal(profile["usual_login_hour"], 1.5), 0, 23
            ))
            ts = ts.replace(hour=h)

        event = {
            "user_id":          profile["user_id"],
            "timestamp":        ts.isoformat(),
            "event_type":       rng.choice(EVENT_TYPES, p=_normal_event_probs()),
            "ip_address":       random.choice(profile["ips"]),
            "device_fingerprint": random.choice(profile["devices"]),
            "country":          profile["country"],
            "session_duration": float(rng.exponential(1_800)),   # seconds
            "trade_volume":     float(rng.lognormal(3, 1)) if rng.random() < 0.4 else 0.0,
            "lot_size":         float(rng.uniform(0.01, 2.0)) if rng.random() < 0.4 else 0.0,
            "instrument":       random.choice(INSTRUMENTS) if rng.random() < 0.4 else None,
            "pnl":              float(rng.normal(50, 200)) if rng.random() < 0.4 else 0.0,
            "deposit_amount":   float(rng.exponential(500)) if rng.random() < 0.15 else 0.0,
            "withdrawal_amount": float(rng.exponential(300)) if rng.random() < 0.08 else 0.0,
            "margin_used":      float(rng.uniform(0.1, 0.6)),
            "pages_per_minute": float(rng.normal(3.0, 1.0)),
            "failed_logins":    int(rng.integers(0, 2)),
            "kyc_change":       False,
        }
        events.append(event)
    return events


def _normal_event_probs():
    p = np.array([0.25, 0.20, 0.15, 0.15, 0.08, 0.04, 0.03, 0.04, 0.04, 0.02])
    return p / p.sum()


# ─────────────────────────────────────────────────────────────────────────────
# Anomaly injectors — one function per section 8.x
# ─────────────────────────────────────────────────────────────────────────────

def inject_ip_hopping(profile: dict) -> list[dict]:
    """8.1 — Rapid IP / geography switching."""
    log.debug("Injecting ip_hopping for %s", profile["user_id"])
    events = generate_events_for_normal_user(profile)
    # Replace IPs with many distinct ones, one per event
    for e in events:
        e["ip_address"] = rand_ip()
    # Add extra rapid logins from foreign IPs
    base_ts = rand_ts(START_DATE, END_DATE)
    for i in range(int(rng.integers(5, 12))):
        ts = base_ts + timedelta(minutes=i * 3)
        e = _base_event(profile, ts)
        e["event_type"] = "login"
        e["ip_address"]  = rand_ip()
        e["country"]     = random.choice(COUNTRIES)
        events.append(e)
    return events


def inject_3am_logins(profile: dict) -> list[dict]:
    """8.1 — Login from unusual time (3 AM)."""
    log.debug("Injecting 3am_logins for %s", profile["user_id"])
    events = generate_events_for_normal_user(profile)
    for _ in range(int(rng.integers(3, 8))):
        ts = rand_ts_hour(START_DATE, END_DATE, hour=int(rng.integers(1, 5)))
        e = _base_event(profile, ts)
        e["event_type"]    = "login"
        e["failed_logins"] = int(rng.integers(0, 2))
        events.append(e)
    return events


def inject_credential_stuffing(profile: dict) -> list[dict]:
    """8.1 — Multiple failed logins then success."""
    log.debug("Injecting credential_stuffing for %s", profile["user_id"])
    events = generate_events_for_normal_user(profile)
    base_ts = rand_ts(START_DATE, END_DATE)
    for i in range(int(rng.integers(5, 15))):
        ts = base_ts + timedelta(seconds=i * 10)
        e = _base_event(profile, ts)
        e["event_type"]    = "login"
        e["failed_logins"] = int(rng.integers(3, 10))
        events.append(e)
    return events


def inject_ip_sharing(profile: dict, shared_ip: str, n_users: int = 5) -> list[dict]:
    """8.5 — Multiple users from same IP (IP hub)."""
    log.debug("Injecting ip_sharing for %s (hub %s)", profile["user_id"], shared_ip)
    events = generate_events_for_normal_user(profile)
    for e in events[:max(1, len(events) // 2)]:
        e["ip_address"] = shared_ip
    return events


def inject_deposit_withdraw_abuse(profile: dict) -> list[dict]:
    """8.2 — Deposit → minimal trading → withdrawal (bonus abuse)."""
    log.debug("Injecting deposit_withdraw_abuse for %s", profile["user_id"])
    events = []
    base_ts = rand_ts(START_DATE, END_DATE - timedelta(days=10))

    # 1. Large deposit
    e = _base_event(profile, base_ts)
    e["event_type"]     = "deposit"
    e["deposit_amount"] = float(rng.uniform(5_000, 20_000))
    events.append(e)

    # 2. Tiny trade (minimal activity)
    e = _base_event(profile, base_ts + timedelta(hours=2))
    e["event_type"]  = "trade_open"
    e["trade_volume"] = float(rng.uniform(0.01, 0.05))
    e["lot_size"]    = 0.01
    events.append(e)

    # 3. Large withdrawal shortly after
    e = _base_event(profile, base_ts + timedelta(days=int(rng.integers(1, 5))))
    e["event_type"]        = "withdrawal"
    e["withdrawal_amount"] = float(rng.uniform(4_000, 18_000))
    events.append(e)

    events += generate_events_for_normal_user(profile)[:3]
    return events


def inject_structuring(profile: dict) -> list[dict]:
    """8.2 — High-frequency small deposits (structuring / smurfing)."""
    log.debug("Injecting structuring for %s", profile["user_id"])
    events = generate_events_for_normal_user(profile)
    base_ts = rand_ts(START_DATE, END_DATE - timedelta(days=3))
    for i in range(int(rng.integers(8, 20))):
        ts = base_ts + timedelta(hours=i * 4)
        e = _base_event(profile, ts)
        e["event_type"]     = "deposit"
        e["deposit_amount"] = float(rng.uniform(800, 2_000))  # just below typical AML threshold
        events.append(e)
    return events


def inject_sudden_withdrawal(profile: dict) -> list[dict]:
    """8.2 — Sudden large withdrawal after dormancy."""
    log.debug("Injecting sudden_withdrawal for %s", profile["user_id"])
    # Very few events, then a large withdrawal at the end
    events = []
    dormant_ts = rand_ts(START_DATE, START_DATE + timedelta(days=30))
    e = _base_event(profile, dormant_ts)
    e["event_type"] = "login"
    events.append(e)

    withdrawal_ts = dormant_ts + timedelta(days=int(rng.integers(30, 90)))
    e = _base_event(profile, withdrawal_ts)
    e["event_type"]        = "withdrawal"
    e["withdrawal_amount"] = float(rng.uniform(10_000, 50_000))
    e["kyc_change"]        = True     # 8.7 — KYC change before withdrawal
    events.append(e)
    return events


def inject_volume_spike(profile: dict) -> list[dict]:
    """8.3 — Sudden spike in trade volume (10× baseline)."""
    log.debug("Injecting volume_spike for %s", profile["user_id"])
    events = generate_events_for_normal_user(profile)
    base_ts = rand_ts(START_DATE, END_DATE - timedelta(days=5))
    for i in range(int(rng.integers(5, 15))):
        ts = base_ts + timedelta(hours=i)
        e = _base_event(profile, ts)
        e["event_type"]  = "trade_open"
        e["trade_volume"] = float(rng.uniform(50_000, 200_000))  # 10× normal
        e["lot_size"]    = float(rng.uniform(5.0, 20.0))
        events.append(e)
    return events


def inject_instrument_concentration(profile: dict) -> list[dict]:
    """8.3 — Single-instrument concentration (Herfindahl ≈ 1.0)."""
    log.debug("Injecting instrument_concentration for %s", profile["user_id"])
    events = generate_events_for_normal_user(profile)
    single = "XAU/USD"
    for e in events:
        if e["event_type"] in ("trade_open", "trade_close"):
            e["instrument"] = single
    # Add extra concentrated trades
    base_ts = rand_ts(START_DATE, END_DATE - timedelta(days=5))
    for i in range(int(rng.integers(8, 20))):
        ts = base_ts + timedelta(hours=i)
        e = _base_event(profile, ts)
        e["event_type"]  = "trade_open"
        e["instrument"]  = single
        e["trade_volume"] = float(rng.uniform(1_000, 5_000))
        events.append(e)
    return events


def inject_latency_arb(profile: dict) -> list[dict]:
    """8.3 — Consistent profit in very short bursts (latency arb)."""
    log.debug("Injecting latency_arb for %s", profile["user_id"])
    events = generate_events_for_normal_user(profile)
    base_ts = rand_ts(START_DATE, END_DATE - timedelta(days=5))
    for i in range(int(rng.integers(10, 30))):
        ts = base_ts + timedelta(seconds=i * 2)   # trades every 2 seconds
        e = _base_event(profile, ts)
        e["event_type"]  = "trade_open"
        e["pnl"]         = float(rng.uniform(50, 300))   # always profitable
        e["session_duration"] = float(rng.uniform(1, 5)) # very short session
        events.append(e)
    return events


def inject_bot_behavior(profile: dict) -> list[dict]:
    """8.4 — Robotic inter-event timing (bot-like navigation)."""
    log.debug("Injecting bot_behavior for %s", profile["user_id"])
    events = generate_events_for_normal_user(profile)
    base_ts = rand_ts(START_DATE, END_DATE - timedelta(days=2))
    for i in range(int(rng.integers(20, 50))):
        ts = base_ts + timedelta(milliseconds=i * 500)    # exactly 500ms intervals
        e = _base_event(profile, ts)
        e["pages_per_minute"] = float(rng.normal(25.0, 0.5))   # very high, very consistent
        e["session_duration"] = 0.5 + rng.normal(0, 0.01)      # nearly constant
        events.append(e)
    return events


def inject_collusion_ring(profiles: list[dict], shared_ip: str) -> list[list[dict]]:
    """8.5 — Synchronized trading patterns across accounts (collusion ring)."""
    log.info("Injecting collusion_ring with %d users, shared IP=%s", len(profiles), shared_ip)
    base_ts = rand_ts(START_DATE, END_DATE - timedelta(days=10))
    all_events = []
    for profile in profiles:
        events = generate_events_for_normal_user(profile)
        # Mirror trades: all users trade same instrument at nearly same time
        for i in range(int(rng.integers(5, 15))):
            ts = base_ts + timedelta(minutes=i) + timedelta(seconds=float(rng.uniform(-5, 5)))
            e = _base_event(profile, ts)
            e["event_type"]  = "trade_open"
            e["ip_address"]  = shared_ip
            e["instrument"]  = "EUR/USD"
            e["lot_size"]    = 1.0
            e["trade_volume"] = 10_000.0
            events.append(e)
        all_events.append(events)
    return all_events


def inject_news_trading(profile: dict) -> list[dict]:
    """8.6 — Trading aligned with major news events."""
    log.debug("Injecting news_trading for %s", profile["user_id"])
    events = generate_events_for_normal_user(profile)
    # Non-farm payrolls: first Friday of each month
    news_dates = [datetime(2024, m, 5) for m in range(1, 7)]
    for nd in news_dates:
        for _ in range(int(rng.integers(3, 8))):
            ts = nd + timedelta(hours=13, minutes=int(rng.integers(-5, 30)))
            e = _base_event(profile, ts)
            e["event_type"]  = "trade_open"
            e["trade_volume"] = float(rng.uniform(5_000, 20_000))
            e["pnl"]         = float(rng.uniform(200, 1_000))
            events.append(e)
    return events


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _base_event(profile: dict, ts: datetime) -> dict:
    return {
        "user_id":            profile["user_id"],
        "timestamp":          ts.isoformat(),
        "event_type":         "login",
        "ip_address":         random.choice(profile["ips"]),
        "device_fingerprint": random.choice(profile["devices"]),
        "country":            profile["country"],
        "session_duration":   float(rng.exponential(1_800)),
        "trade_volume":       0.0,
        "lot_size":           0.0,
        "instrument":         None,
        "pnl":                0.0,
        "deposit_amount":     0.0,
        "withdrawal_amount":  0.0,
        "margin_used":        float(rng.uniform(0.1, 0.6)),
        "pages_per_minute":   float(rng.normal(3.0, 1.0)),
        "failed_logins":      0,
        "kyc_change":         False,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main generation pipeline
# ─────────────────────────────────────────────────────────────────────────────

def build_dataset() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns
    -------
    events_df : all ~50k events (no labels — fed to models)
    labels_df : per-user ground truth (anomaly_type, is_anomaly)
    """
    log.info("=" * 60)
    log.info("ForexGuard Synthetic Data Generator")
    log.info("=" * 60)
    log.info("Normal users  : %d", N_NORMAL_USERS)
    log.info("Anomalous users: %d", N_ANOMALOUS_USERS)

    all_events: list[dict] = []
    label_records: list[dict] = []

    # ── 1. Normal users ──────────────────────────────────────────
    log.info("Generating normal users...")
    for i in range(N_NORMAL_USERS):
        uid = f"U{i:05d}"
        profile = generate_normal_user(uid)
        events  = generate_events_for_normal_user(profile)
        all_events.extend(events)
        label_records.append({
            "user_id":      uid,
            "anomaly_type": "normal",
            "is_anomaly":   0,
        })
    log.info("Normal users done. Events so far: %d", len(all_events))

    # ── 2. Anomalous users — split across 8 pattern types ────────
    log.info("Injecting anomalous patterns...")
    anon_uid_counter = N_NORMAL_USERS

    anomaly_budget = {
        "ip_hopping":           60,
        "3am_logins":           55,
        "credential_stuffing":  45,
        "deposit_withdraw":     60,
        "structuring":          45,
        "sudden_withdrawal":    45,
        "volume_spike":         50,
        "instrument_conc":      40,
        "latency_arb":          30,
        "bot_behavior":         30,
        "news_trading":         20,
        # collusion rings handled separately below (need groups of 5)
        "collusion_ring":       20,   # users — must be divisible by ring_size=5
    }
    # Sanity check
    assert sum(anomaly_budget.values()) == N_ANOMALOUS_USERS, \
        f"Budget sums to {sum(anomaly_budget.values())}, expected {N_ANOMALOUS_USERS}"

    def _make_uid() -> str:
        nonlocal anon_uid_counter
        uid = f"A{anon_uid_counter:05d}"
        anon_uid_counter += 1
        return uid

    def _add_anomaly(events: list[dict], uid: str, atype: str):
        all_events.extend(events)
        label_records.append({"user_id": uid, "anomaly_type": atype, "is_anomaly": 1})

    # Simple single-user anomalies
    single_patterns = {
        "ip_hopping":          inject_ip_hopping,
        "3am_logins":          inject_3am_logins,
        "credential_stuffing": inject_credential_stuffing,
        "deposit_withdraw":    inject_deposit_withdraw_abuse,
        "structuring":         inject_structuring,
        "sudden_withdrawal":   inject_sudden_withdrawal,
        "volume_spike":        inject_volume_spike,
        "instrument_conc":     inject_instrument_concentration,
        "latency_arb":         inject_latency_arb,
        "bot_behavior":        inject_bot_behavior,
        "news_trading":        inject_news_trading,
    }

    for atype, injector in single_patterns.items():
        n = anomaly_budget[atype]
        log.info("  -> %s : %d users", atype, n)
        for _ in range(n):
            uid     = _make_uid()
            profile = generate_normal_user(uid)
            profile["anomaly_type"] = atype
            events  = injector(profile)
            _add_anomaly(events, uid, atype)

    # Collusion rings — groups of 5 users sharing same IP
    n_ring_users = anomaly_budget["collusion_ring"]
    ring_size    = 5
    n_rings      = n_ring_users // ring_size
    log.info("  -> collusion_ring : %d rings x %d users", n_rings, ring_size)
    for _ in range(n_rings):
        shared_ip   = rand_ip()
        ring_uids   = [_make_uid() for _ in range(ring_size)]
        ring_profiles = [generate_normal_user(uid) for uid in ring_uids]
        all_ring_events = inject_collusion_ring(ring_profiles, shared_ip)
        for uid, events in zip(ring_uids, all_ring_events):
            _add_anomaly(events, uid, "collusion_ring")

    log.info("Total events generated: %d", len(all_events))
    log.info("Total users: %d", len(label_records))

    events_df = pd.DataFrame(all_events)
    labels_df = pd.DataFrame(label_records)

    # ── 3. Post-processing ───────────────────────────────────────
    events_df["timestamp"] = pd.to_datetime(events_df["timestamp"], format="mixed")
    events_df = events_df.sort_values("timestamp").reset_index(drop=True)

    # Fill nulls
    events_df["instrument"]    = events_df["instrument"].fillna("NONE")
    events_df["trade_volume"]  = events_df["trade_volume"].clip(lower=0)
    events_df["lot_size"]      = events_df["lot_size"].clip(lower=0)
    events_df["pnl"]           = events_df["pnl"].fillna(0.0)
    events_df["deposit_amount"] = events_df["deposit_amount"].clip(lower=0)
    events_df["withdrawal_amount"] = events_df["withdrawal_amount"].clip(lower=0)
    events_df["pages_per_minute"] = events_df["pages_per_minute"].clip(lower=0)
    events_df["session_duration"] = events_df["session_duration"].clip(lower=0)

    log.info("Events shape : %s", events_df.shape)
    log.info("Labels shape : %s", labels_df.shape)
    log.info("Anomaly distribution:\n%s",
             labels_df["anomaly_type"].value_counts().to_string())

    return events_df, labels_df


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    log.info("Starting data generation pipeline...")
    events_df, labels_df = build_dataset()

    events_path = OUT_DIR / "events.parquet"
    labels_path = OUT_DIR / "labels.parquet"
    meta_path   = OUT_DIR / "meta.json"

    events_df.to_parquet(events_path, index=False)
    labels_df.to_parquet(labels_path, index=False)

    meta = {
        "generated_at":    datetime.utcnow().isoformat(),
        "n_events":        len(events_df),
        "n_users":         len(labels_df),
        "n_anomalous":     int(labels_df["is_anomaly"].sum()),
        "anomaly_rate":    float(labels_df["is_anomaly"].mean()),
        "date_range":      [START_DATE.isoformat(), END_DATE.isoformat()],
        "columns":         list(events_df.columns),
        "anomaly_types":   labels_df["anomaly_type"].value_counts().to_dict(),
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    log.info("Saved events  → %s", events_path)
    log.info("Saved labels  → %s (DO NOT pass to models)", labels_path)
    log.info("Saved meta    → %s", meta_path)
    log.info("Data generation complete.")

    # Quick sanity checks
    assert len(events_df) > 40_000, f"Too few events: {len(events_df)}"
    assert events_df["user_id"].nunique() == len(labels_df), "User count mismatch"
    assert labels_df["is_anomaly"].sum() == N_ANOMALOUS_USERS, "Anomaly count mismatch"
    log.info("All sanity checks passed ✓")


if __name__ == "__main__":
    main()
