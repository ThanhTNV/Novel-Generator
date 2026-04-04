# Documentation

## Docker Deployment

### Prerequisites

- Docker Engine 20.10+
- Docker Compose v2

### Start with Docker

```bash
cp .env.docker.example .env.docker   # create config
# edit .env.docker — add at least one LLM API key

docker compose up -d                  # build & run
docker compose logs -f                # watch logs
```

The app is available at **http://localhost:8000**.

### Start with local LLM (Ollama)

```bash
docker compose --profile with-ollama up -d
docker compose exec ollama ollama pull llama3.1
```

Set `DEFAULT_LLM_PROVIDER=ollama` and `DEFAULT_MODEL=llama3.1` in `.env.docker`, then `docker compose restart`.

### Persistent volumes

| Volume | Container path | Purpose |
|--------|---------------|---------|
| `vectorstore_data` | `/app/data/vectorstore` | ChromaDB embeddings |
| `chapters_data` | `/app/data/chapters` | Finalized chapters |
| `ollama_data` | `/root/.ollama` | Downloaded models (optional) |

Configuration directories (`context/`, `rules/`, `skills/`, `prompts/`) are bind-mounted read-only.

### Common operations

```bash
docker compose ps                     # status
docker compose restart                # restart
docker compose down                   # stop
docker compose down -v                # stop and delete volumes
docker compose build --no-cache       # rebuild image
```

### Production notes

Before exposing to the internet:

1. Put a reverse proxy (nginx / Caddy) in front for TLS termination.
2. Restrict CORS in `backend/server.py` from `["*"]` to your domain.
3. Add authentication (API key header, OAuth, etc.).
4. Rotate API keys periodically; never commit `.env` files.
5. Adjust resource limits in `docker-compose.yml` to match your hardware.

---

## API Reference

All endpoints are served from the FastAPI backend at `http://localhost:8000`.

### Generation

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/generate` | Generate a chapter draft |
| `POST` | `/api/revise` | Revise a draft with feedback |

Both accept `"stream": true` for SSE streaming.

**Generate request body:**

```json
{
  "chapter_instructions": "Elara crosses the Smuggler's Cut...",
  "story_summary": "After fleeing Verenthia...",
  "characters": ["Elara", "Theron"],
  "locations": ["Ironspine Mountains"],
  "target_words": 2000,
  "temperature": 0.7,
  "provider": null,
  "model": null,
  "stream": false
}
```

**Revise request body:**

```json
{
  "draft": "...",
  "feedback": "Make the dialogue sharper and add more tension.",
  "stream": false
}
```

### Vector store

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/ingest/text` | Ingest raw text |
| `POST` | `/api/ingest/file` | Ingest a file by path |
| `POST` | `/api/ingest/context` | Ingest all `context/` files |
| `POST` | `/api/search` | Semantic search |
| `GET` | `/api/vectordb/stats` | Chunk count |
| `DELETE` | `/api/vectordb/clear` | Wipe collection |

### Chapters

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/chapters/save` | Save & auto-index a chapter |
| `GET` | `/api/chapters` | List saved chapters |
| `GET` | `/api/chapters/{file}` | Read a chapter |

### Configuration

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/config` | Current runtime config |
| `GET` | `/api/rules` | Active rules |
| `GET` | `/api/skills` | Active skills |

---

## Customization

### Rules (`rules/*.md`)

Always-on constraints injected into every prompt. The agent concatenates all `.md` files in `rules/` and includes them in the system prompt.

Defaults: `tone.md`, `style.md`, `genre.md`.

### Skills (`skills/*.md`)

Step-by-step workflows the agent follows during retrieval and generation. Each skill describes what context to fetch and how to verify consistency.

Defaults: `character-consistency.md`, `plot-thread-check.md`, `world-building.md`.

### Prompts (`prompts/*.md`)

- `base-prompt.md` — system-level instructions.
- `chapter-prompt.md` — per-generation template with `{{ variable }}` placeholders filled at runtime.

### Context (`context/*.md`)

Your novel's world bible. Ingest these into ChromaDB via the Vector DB tab or `/api/ingest/context`. The RAG pipeline retrieves relevant chunks during generation.

---

## Security

- **No secrets in code.** All API keys are loaded from environment variables via `backend/config.py`.
- **`.env` is git-ignored.** Copy `.env.example` (local) or `.env.docker.example` (Docker) and fill in your keys.
- **Non-root container.** The Dockerfile creates and runs as `appuser` (UID 1000).
- **Read-only mounts.** Config directories are mounted `:ro` in Docker.
- **Resource limits.** `docker-compose.yml` caps CPU and memory.

If you ever accidentally expose a key, revoke it immediately in your provider dashboard and rotate.
