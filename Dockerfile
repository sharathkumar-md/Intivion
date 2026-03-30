# ForexGuard — Multi-stage Dockerfile
# Stage 1: base with all dependencies
# Stage 2: api — FastAPI server
# Stage 3: dashboard — Streamlit app

FROM python:3.11-slim AS base

WORKDIR /app

# System deps for networkx, scipy, etc.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY forexguard/ forexguard/


# ─────────────────────────────────────────────────────────────────────────────
# API stage
# ─────────────────────────────────────────────────────────────────────────────
FROM base AS api

EXPOSE 8000

CMD ["uvicorn", "forexguard.api.app:app", \
     "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]


# ─────────────────────────────────────────────────────────────────────────────
# Dashboard stage
# ─────────────────────────────────────────────────────────────────────────────
FROM base AS dashboard

EXPOSE 8501

# Streamlit config: disable usage stats, set server settings
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

CMD ["streamlit", "run", "forexguard/dashboard/app.py", \
     "--server.address", "0.0.0.0"]
