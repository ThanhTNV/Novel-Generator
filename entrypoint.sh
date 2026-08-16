#!/usr/bin/env bash
set -e

mkdir -p /app/data/memory /app/data/chapters

echo "Novel Generator"
echo "  Provider : ${DEFAULT_LLM_PROVIDER:-claude}"
echo "  Model    : ${DEFAULT_MODEL:-claude-sonnet-4-20250514}"
echo "  Embedding: ${EMBEDDING_PROVIDER:-auto}${EMBEDDING_MODEL:+ (${EMBEDDING_MODEL})}"
echo "  Listen   : ${HOST:-0.0.0.0}:${PORT:-8000}"

exec python main.py
