FROM python:3.11-slim AS builder

WORKDIR /build

RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# -------------------------------------------------------------------

FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

COPY --from=builder /install /usr/local

RUN useradd -m -u 1000 appuser

COPY main.py .
COPY backend/ ./backend/
COPY frontend/ ./frontend/
COPY prompts/ ./prompts/
COPY rules/ ./rules/
COPY skills/ ./skills/
COPY entrypoint.sh .
RUN chmod +x entrypoint.sh && \
    mkdir -p novels && \
    chown -R appuser:appuser /app

USER appuser

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -sf http://localhost:8000/api/config || exit 1

EXPOSE 8000

ENTRYPOINT ["./entrypoint.sh"]
