FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_SYSTEM_PYTHON=1 \
    UV_NATIVE_TLS=1 \
    UV_CONCURRENT_DOWNLOADS=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && update-ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.6.17 /uv /uvx /bin/

COPY backend/pyproject.toml /tmp/pyproject.toml
RUN uv pip compile /tmp/pyproject.toml \
        --extra-index-url https://download.pytorch.org/whl/cu121 \
        -o /tmp/requirements.lock \
    && uv pip install --system --no-cache \
        --extra-index-url https://download.pytorch.org/whl/cu121 \
        -r /tmp/requirements.lock

COPY backend /app

EXPOSE 8000
