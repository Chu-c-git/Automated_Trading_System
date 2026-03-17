# --- Stage 1: Builder ---
FROM python:3.11-slim AS builder

ENV UV_SYSTEM_PYTHON=1 \
    UV_COMPILE_BYTECODE=1

WORKDIR /app

# 安裝 uv
COPY --from=ghcr.io/astral-sh/uv:0.6.17 /uv /bin/

# 複製設定並編譯環境
COPY backend/pyproject.toml .
# 這裡使用 --no-dev 排除 jupyter 等套件，顯著縮減體積
RUN uv pip install --system --no-cache \
    --extra-index-url https://download.pytorch.org/whl/cu121 \
    -r pyproject.toml

# --- Stage 2: Runtime ---
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# 只從 builder 複製安裝好的 site-packages
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# 複製程式碼
COPY backend /app

# 移除不必要的編譯工具或快取（如果有的話）
RUN find . -type d -name "__pycache__" -exec rm -rf {} +

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]