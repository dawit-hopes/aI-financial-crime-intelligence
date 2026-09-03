# ---------- build stage -------------------------------------------------
FROM python:3.12-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /build
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
# Install into a self-contained prefix we can copy into the runtime image, so
# no compiler or headers ship to production.
RUN pip install --prefix=/install --no-cache-dir \
        "fastapi>=0.115" "uvicorn[standard]>=0.32" "pydantic>=2.9" "pydantic-settings>=2.6" \
        "sqlalchemy[asyncio]>=2.0.36" "alembic>=1.14" "asyncpg>=0.30" "aiosqlite>=0.20" \
        "redis>=5.2" "httpx>=0.28" "pyjwt[crypto]>=2.10" "argon2-cffi>=23.1" \
        "python-multipart>=0.0.12" "email-validator>=2.2" \
        "pandas>=3.0.5" "scipy>=1.17.1" "xgboost>=3.2.0" "shap>=0.51.0"

# ---------- runtime stage -----------------------------------------------
FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# Non-root from the start. The app never needs to write to its own filesystem.
RUN useradd --create-home --uid 10001 --shell /usr/sbin/nologin appuser

COPY --from=builder /install /usr/local
WORKDIR /app
COPY --chown=appuser:appuser app ./app
COPY --chown=appuser:appuser scripts ./scripts
COPY --chown=appuser:appuser alembic ./alembic
COPY --chown=appuser:appuser alembic.ini ./

USER appuser
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/healthz').status==200 else 1)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2", "--no-server-header"]