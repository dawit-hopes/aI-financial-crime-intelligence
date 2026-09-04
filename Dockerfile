# Inference image for the ML library. HTTP lives in the backend repo.
# Matches that production image: Python 3.12, CPU XGBoost, no SHAP.

# ---------- build stage -------------------------------------------------
FROM python:3.12-slim-bookworm AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /build

COPY pyproject.toml ./
RUN pip install --prefix=/install --no-cache-dir \
        "pydantic>=2.9" \
        "numpy>=2.0" \
        "networkx>=3.6.1" \
        "xgboost-cpu==3.4.1" \
        "scikit-learn>=1.9.0" \
        "joblib>=1.4" \
    && find /install -type d \( -name "__pycache__" -o -name "tests" -o -name "test" \) \
        -prune -exec rm -rf {} + \
    && find /install -type f -name "*.pyc" -delete

# ---------- runtime stage -----------------------------------------------
FROM python:3.12-slim-bookworm AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 --shell /usr/sbin/nologin appuser

COPY --from=builder /install /usr/local

WORKDIR /app
COPY --chown=appuser:appuser src ./src
COPY --chown=appuser:appuser scripts/__init__.py scripts/smoke_inference.py ./scripts/

USER appuser

HEALTHCHECK --interval=30s --timeout=30s --start-period=20s --retries=2 \
    CMD python -m scripts.smoke_inference

CMD ["python", "-m", "scripts.smoke_inference"]
