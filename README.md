[![CI](https://github.com/ThanhTNV/Novel-Generator/actions/workflows/ci.yml/badge.svg)](https://github.com/ThanhTNV/Novel-Generator/actions/workflows/ci.yml)
# Novel Generator

AI-powered novel writing agent with Zero-Mem structured memory (arXiv:2607.29377), modular skills, always-on rules, and a web-based chapter editor.

Generate, revise, and finalize novel chapters using Claude, OpenAI, Groq, or local models via Ollama. Context from your world bible and previous chapters is retrieved by a Zero-Mem engine — verbatim narrative traces in SQLite, a character/location entity graph with Personalized PageRank, and BM25 + dense dual-view retrieval — and injected into every prompt as contiguous, correctly-attributed passages. No LLM call is spent on memory operations.

## Features

- **No local ML** — embeddings (`gemini-embedding-001`) and entity/relation extraction (`gemini-3.1-flash-lite`) both run in the cloud over plain HTTP. Set one `GEMINI_API_KEY` and the whole memory layer works; without it the engine uses a built-in hash function and still retrieves well via BM25 + the entity graph. Nothing is ever downloaded.
- **Zero-Mem memory** — replaces the old ChromaDB RAG pipeline, which measurably lost context on this Vietnamese corpus: its sentence splitter required `[A-Z]` after punctuation (never matches Đ/Ư/Ổ...), collapsing the whole character bible into two ~400-word blobs; revised chapters left stale chunks behind forever; and every generation fired a hard-coded English plot query. Zero-Mem stores paragraphs verbatim with heading provenance, supersedes on re-ingest, grounds queries in an entity graph (accent-tolerant: "Van Tam" finds "Văn Tâm"), and returns contiguous passages in narrative order.
- **Multi-provider LLM** — Claude, OpenAI, Groq, Ollama with streaming support.
- **Verified history, not remembered history** — the writer can call a `search_history` tool that answers only from a corpus you vetted. A record without a citation is refused when the corpus loads, retrieval generates nothing, and no match returns an explicit "no record" rather than a guess. Finished drafts are checked for anachronisms and contradictions, each flag citing its source. Alternate history is first-class: declare where your novel departs and the checker stops flagging your own premise.
- **One folder per novel** — each book keeps its own world bible, chapters and memory store, so writing chapter 12 of one can never retrieve a character from another. Switch between them in the sidebar.
- **Skills & rules** — modular markdown files that shape every generation, with per-novel overrides when one book needs a different tone.
- **Web editor** — compose, preview, revise and finalize chapters in the browser, in light or dark.
- **Feedback loop** — edit drafts, request revisions, then save and auto-index.
- **Docker-ready** — single command to build and run with persistent volumes.

## Project structure

```
backend/
  config.py          Settings from environment variables
  novels.py          Per-novel workspaces: paths, registry, first-run seeding
  history/           Vetted historical corpus: loader, closed-corpus search,
                     anachronism + contradiction checker, LLM tool surface
  rag_pipeline.py    Zero-Mem facade — one engine per novel
  zero_mem/          memory engine: segmentation, entity graph, PPR,
                     BM25 + dense retrieval, SQLite trace store
  api_client.py      Unified LLM client (Claude / OpenAI / Groq / Ollama)
  agent.py           Prompt assembly with skill and rule injection
  server.py          FastAPI REST API with SSE streaming
frontend/
  templates/         HTML
  static/            CSS + JS
skills/              Project-default generation workflows
rules/               Project-default constraints (tone, style, genre)
prompts/             System and chapter prompt templates
history/*.yaml       Vetted historical record, shared by every novel

novels/<id>/         one self-contained novel
  novel.json         title, description, created_at
  context/           world bible — the entity gazetteer is built from this
  chapters/          finalized chapters
  rules/ skills/     optional per-novel overrides
  history/           this novel's added records and divergence points
  memory/            that novel's own Zero-Mem store (SQLite)
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

Open **http://localhost:8000**. A default novel is created on first run, seeded from any existing `context/` and `chapters/`
(copied, not moved). Open the **World** tab to write its bible and index it, then **Compose** to write a chapter.
Use the switcher at the top of the sidebar — or `⌘/Ctrl + K` — to add another novel.

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
