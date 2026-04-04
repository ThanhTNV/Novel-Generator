# Novel Generator

AI-powered novel writing agent with RAG, modular skills, always-on rules, and a web-based chapter editor.

Generate, revise, and finalize novel chapters using Claude, OpenAI, Groq, or local models via Ollama. Context from your world bible is automatically retrieved from a ChromaDB vector store and injected into every prompt.

## Features

- **RAG pipeline** — chunking, embedding, and semantic retrieval via ChromaDB.
- **Multi-provider LLM** — Claude, OpenAI, Groq, Ollama with streaming support.
- **Skills & rules** — modular markdown files that shape every generation.
- **Web editor** — generate, preview, revise, and finalize chapters in the browser.
- **Feedback loop** — edit drafts, request revisions, then save and auto-index.
- **Docker-ready** — single command to build and run with persistent volumes.

## Project structure

```
backend/
  config.py          Settings from environment variables
  rag_pipeline.py    ChromaDB embedding, chunking, retrieval
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
vectorstore/         ChromaDB persistent storage
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

Persistent data lives in `data/vectorstore/` and `data/chapters/`.

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
