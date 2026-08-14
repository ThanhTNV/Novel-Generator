[![CI](https://github.com/ThanhTNV/Novel-Generator/actions/workflows/ci.yml/badge.svg)](https://github.com/ThanhTNV/Novel-Generator/actions/workflows/ci.yml)
# Novel Generator

AI-powered novel writing agent with Zero-Mem structured memory (arXiv:2607.29377), modular skills, always-on rules, and a web-based chapter editor.

Generate, revise, and finalize novel chapters using Claude, OpenAI, Groq, or local models via Ollama. Context from your world bible and previous chapters is retrieved by a Zero-Mem engine — verbatim narrative traces in SQLite, a character/location entity graph with Personalized PageRank, and BM25 + dense dual-view retrieval — and injected into every prompt as contiguous, correctly-attributed passages. No LLM call is spent on memory operations.

## Features

- **Zero-Mem memory** — replaces the old ChromaDB RAG pipeline, which measurably lost context on this Vietnamese corpus: its sentence splitter required `[A-Z]` after punctuation (never matches Đ/Ư/Ổ...), collapsing the whole character bible into two ~400-word blobs; revised chapters left stale chunks behind forever; and every generation fired a hard-coded English plot query. Zero-Mem stores paragraphs verbatim with heading provenance, supersedes on re-ingest, grounds queries in an entity graph (accent-tolerant: "Van Tam" finds "Văn Tâm"), and returns contiguous passages in narrative order.
- **Multi-provider LLM** — Claude, OpenAI, Groq, Ollama with streaming support.
- **Skills & rules** — modular markdown files that shape every generation.
- **Web editor** — generate, preview, revise, and finalize chapters in the browser.
- **Feedback loop** — edit drafts, request revisions, then save and auto-index.
- **Docker-ready** — single command to build and run with persistent volumes.

## Project structure

```
backend/
  config.py          Settings from environment variables
  rag_pipeline.py    Zero-Mem facade (legacy API kept for compatibility)
  zero_mem/          memory engine: segmentation, entity graph, PPR,
                     BM25 + dense retrieval, SQLite trace store
  api_client.py      Unified LLM client (Claude / OpenAI / Groq / Ollama)
  agent.py           Prompt assembly with skill and rule injection
  server.py          FastAPI REST API with SSE streaming
frontend/
  templates/         HTML
  static/            CSS + JS
skills/              Step-by-step generation workflows
rules/               Always-on constraints (tone, style, genre)
prompts/             System and chapter prompt templates
context/             Novel world files (characters, locations, plot)
chapters/            Finalized chapter output
data/zero_mem.db     Zero-Mem trace store (SQLite)
```

## Quick start

```bash
# 1. Install
pip install -r requirements.txt

# 2. Configure
cp .env.example .env
# edit .env — set at least one API key

# 3. Run
python main.py
```

Open **http://localhost:8000**, go to the **Vector DB** tab, click **Ingest Context Directory**, then start generating chapters.

## Docker

```bash
cp .env.docker.example .env.docker
# edit .env.docker — set at least one API key

docker compose up -d
```

Optionally include a local Ollama instance:

```bash
docker compose --profile with-ollama up -d
```

Persistent data lives in `data/memory/` and `data/chapters/`.

See [DOCS.md](DOCS.md) for volume details, API reference, production notes, and customization.

## LLM providers

| Provider | Env variable | Example models |
|----------|-------------|----------------|
| Claude | `ANTHROPIC_API_KEY` | `claude-sonnet-4-20250514` |
| OpenAI | `OPENAI_API_KEY` | `gpt-4o`, `gpt-4o-mini` |
| Groq | `GROQ_API_KEY` | `llama-3.3-70b-versatile` |
| Ollama | `OLLAMA_BASE_URL` | Any local model |

Set `DEFAULT_LLM_PROVIDER` and `DEFAULT_MODEL` in `.env`.

## License

MIT
