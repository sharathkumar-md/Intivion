# ForexGuard

**Live demo: [forexguard.streamlit.app](https://forexguard.streamlit.app)**

A real-time anomaly detection engine I built for detecting suspicious trader behaviour in a forex brokerage environment. The idea is to catch things like collusion rings, bonus abuse, latency arb, bot traders etc. — without any labels, purely unsupervised.

---

## Why this approach

I went with Isolation Forest + LSTM Autoencoder rather than a single model because they catch different things. IF is great at finding weird outliers in aggregated tabular features (like someone who deposits a lot but barely trades), while the LSTM picks up on unusual *sequences* of events — like a sudden change in behaviour pattern over someone's last 20 actions. I fuse them with a max() so if either model screams anomaly, the user gets flagged.

For the graph stuff — I added NetworkX because Section 8.5 (collusion rings) literally can't be solved with per-user models alone. You need to see the structure of *who shares IPs with who*, and Louvain community detection + PageRank does that nicely.

I dropped VAE early on. Training instability (posterior collapse) plus tricky threshold selection made it not worth it for a 2-day build. The LSTM AE gets the same job done with way more predictable training.

---

## Results

| Model | AUROC | AUPRC |
|---|---|---|
| Isolation Forest | 0.9554 | 0.8276 |
| LSTM Autoencoder | 0.9159 | 0.7334 |
| Ensemble (max-fusion) | **0.9531** | **0.8349** |

At the HIGH alert tier, precision is exactly 1.0 — zero false positives. That matters a lot for compliance teams who can't afford to be chasing ghosts.

---

## Architecture

```
Event stream (async queue, Kafka-compatible)
        |
   Feature Engineering
   /         |          \
Tabular    Sequences    Graph
(54 feats) (LSTM input) (NetworkX)
   |         |          |
   IF+SHAP  LSTM AE    Louvain/PageRank
        \     |     /
         max-fusion
              |
        Alert tiers (CRITICAL / HIGH / MEDIUM / LOW)
              |
     FastAPI + Streamlit + Claude/Gemini LLM summaries
```

---

## Setup

```bash
pip install -r requirements.txt
```

Run the full pipeline from scratch:
```bash
python run_pipeline.py
```

Or if you already have the data and just want to retrain models:
```bash
python run_pipeline.py --skip-data --skip-features
```

Individual stages:
```bash
python forexguard/data/generate.py          # generates ~54k events with ground truth labels
python forexguard/features/tabular.py       # 54 per-user aggregated features
python forexguard/features/sequences.py     # (5000, 20, 21) tensor for LSTM
python forexguard/features/graph.py         # bipartite user-IP-device graph
python forexguard/models/isolation_forest.py
python forexguard/models/lstm_autoencoder.py
python forexguard/models/ensemble.py
```

---

## Running the services

```bash
# API on port 8000
uvicorn forexguard.api.app:app --reload

# Dashboard on port 8501
streamlit run forexguard/dashboard/app.py

# MLflow UI on port 5000
mlflow ui --backend-store-uri forexguard/mlruns
```

Or just Docker Compose everything:
```bash
docker-compose up --build
```

For LLM risk summaries, set one of these (Claude is primary, Gemini is fallback):
```bash
export ANTHROPIC_API_KEY=...
export GEMINI_API_KEY=...    # used if Claude key isn't set
```

---

## API

Five endpoints:

- `POST /score` — IF anomaly score + top SHAP features for a user
- `POST /score_seq` — LSTM reconstruction error score from a sequence of events
- `GET /graph_risk/{user_id}` — graph risk features (IP sharing, community size etc.)
- `GET /alerts?tier=HIGH&limit=50` — paginated alert queue
- `GET /alerts/{user_id}` — single user's alert + full explanation

Swagger at `http://localhost:8000/docs`.

---

## How explainability works

**Isolation Forest** — I use TreeSHAP (`shap.TreeExplainer`) directly on the fitted IF model. Each user gets a top-5 list of feature names with their SHAP values. Negative SHAP = pushes toward anomaly. So if someone has `withdrawal_max` as their top negative SHAP feature, it's telling you their withdrawal behaviour is what looks weird.

**LSTM Autoencoder** — After reconstruction, I compute the per-feature MSE across the 20-step window. Whichever features the model struggles to reconstruct are the anomalous ones. If `withdrawal_amount` has 53x higher reconstruction error than normal, that's your signal.

**Graph** — I surface `shared_ip_user_count`, `community_size`, and `is_hub_neighbor` directly in the alert payload. A collusion ring shows up as a cluster of users with high `shared_ip_user_count` all in the same Louvain community.

---

## Feature coverage (what signals each feature targets)

| Signal (from brief) | Feature |
|---|---|
| Multiple IPs / IP hopping (8.1) | `distinct_ips_total`, `ip_hub_max` |
| 3 AM / unusual-hour logins (8.1) | `night_login_ratio`, `login_hour_std` |
| Device fingerprint switching (8.1) | `distinct_devices` |
| Deposit → barely trade → withdraw (8.2) | `deposit_to_trade_ratio` |
| Structuring — many small deposits (8.2) | `is_structuring_flag`, `deposit_amount_cv` |
| Sudden large withdrawal (8.2) | `withdrawal_zscore`, `withdrawal_max` |
| KYC change before withdrawal (8.2, 8.7) | `kyc_change_flag` |
| Volume spike 10x baseline (8.3) | `trade_vol_zscore` |
| Single instrument concentration (8.3) | `instrument_hhi` (Herfindahl index) |
| Latency arb — always profitable (8.3) | `pnl_consistency` |
| Bot-like inter-trade timing (8.3, 8.4) | `inter_event_mean/std`, `nav_regularity` |
| Session duration anomalies (8.4) | `session_dur_zscore` |
| IP hub / collusion rings (8.5) | `shared_ip_user_count`, Louvain `community_size` |
| News-event aligned trading (8.6) | injected anomaly pattern + LSTM temporal signal |
| Failed logins → success (8.7) | `failed_logins_total`, `failed_logins_max` |

---

## Assumptions and trade-offs I made

**Contamination = 10%** — the synthetic dataset has exactly 10% anomalous users, so I set IF's contamination to match. In production you'd tune this based on your historical false-positive rate from compliance reviews.

**Max-fusion over weighted average** — I went with `max(IF, LSTM, graph)` instead of a weighted average because (a) I don't have labelled production data to calibrate weights, and (b) max-fusion is conservative — it flags anyone that *either* model is confident about, which is the right call for compliance. The trade-off is slightly lower recall vs. a tuned weighted ensemble.

**Kafka stub, not live Kafka** — The `EventQueue` in `streaming/simulator.py` has the exact same `put`/`get` interface as aiokafka. Swapping to real Kafka is literally two lines of code. I kept the stub to avoid infra complexity during the assessment.

**LSTM zero-padding** — Users with fewer than 20 events get zero-padded at the start of the sequence. This dilutes the reconstruction signal for sparse users. A min-event filter (e.g., only score users with ≥5 events) would improve the LSTM's AUROC. I left it out to keep all 5000 users scoreable.

**Louvain seed=42** — Community detection is non-deterministic by default. Fixed seed for reproducible graph features across pipeline runs.

**VAE dropped** — Posterior collapse made training unstable. LSTM AE is simpler, more predictable, and gets a better AUROC anyway.

---

## Limitations

- Synthetic data can't fully replicate real brokerage complexity (correlated instruments, real latency distributions, actual fraud ring coordination)
- No sliding-window retraining — scores are static until you rerun `run_pipeline.py`
- LSTM AUROC (0.9159) is lower than IF because zero-padding hurts sparse users

---

## Structure

```
forexguard/
├── data/generate.py              # synthetic dataset generator
├── features/
│   ├── tabular.py                # aggregated per-user features
│   ├── sequences.py              # LSTM sequence builder
│   └── graph.py                  # NetworkX graph + community detection
├── models/
│   ├── isolation_forest.py       # IF + TreeSHAP
│   ├── lstm_autoencoder.py       # LSTM AE + reconstruction error explainability
│   └── ensemble.py               # score fusion + alert classification
├── streaming/simulator.py        # async Kafka-compatible event queue
├── api/app.py                    # FastAPI
├── llm/risk_summary.py           # Claude / Gemini risk narrative generator
├── tracking/mlflow_setup.py      # MLflow logging
└── dashboard/app.py              # Streamlit dashboard
run_pipeline.py                   # runs everything end-to-end
```
