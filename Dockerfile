# ForexGuard — Multi-stage Dockerfile
# base  : deps + run full pipeline (bakes data + models into the image)
# api   : FastAPI server
# dashboard : Streamlit app

FROM python:3.11-slim AS base

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY forexguard/ forexguard/
COPY run_pipeline.py .

# Run the full pipeline at build time so data + models are baked into the image.
# This means the container starts instantly with no warm-up needed.
RUN python run_pipeline.py


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

ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

CMD ["streamlit", "run", "forexguard/dashboard/app.py", \
     "--server.address", "0.0.0.0"]
