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
| `novels_data` | `/app/novels` | Every novel: world bible, chapters, memory store |
| `ollama_data` | `/root/.ollama` | Downloaded models (optional) |

Project-level defaults (`rules/`, `skills/`, `prompts/`) are bind-mounted read-only. Everything an author writes lives under `novels_data`, so one volume backs up the whole library.

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
| `POST` | `/api/ingest/context` | Re-index this novel's whole world bible |
| `POST` | `/api/search` | Semantic search |
| `GET` | `/api/vectordb/stats` | Segment / entity / source counts |
| `GET` | `/api/memory/entity/{name}` | Entity profile: mentions, passages, relation triples, related entities |
| `GET` | `/api/memory/sources` | Ingested documents (reference vs chapter) |
| `DELETE` | `/api/vectordb/clear` | Wipe this novel's memory (files stay on disk) |

**Extractor choice.** `ZERO_MEM_EXTRACTOR` selects who extracts entities/relations: `gemini` (default when a key is set), `ollama` (reuse your Ollama model, no Gemini key needed), or `local` (no SLM). Measured on a 93-segment world bible: Gemini 3.1 Flash-Lite found 29 relations in 14.1s; Ollama gemma4:31b-cloud found 8 in 19.6s and consumes the same free quota chapter writing uses. Ollama's hosted models treat the JSON-schema `format` parameter as a hint rather than a constraint, so the Ollama path uses a shape-tolerant parser.

**Output length.** `max_tokens` is optional on `/api/generate` and `/api/revise`; when omitted the server derives it from `target_words` (or the draft length for revisions). Vietnamese prose measures ~2.26 tokens/word, so the previous fixed 4096 cap truncated a default 2000-word chapter mid-sentence. Tune with `OUTPUT_TOKENS_PER_WORD` / `MAX_OUTPUT_TOKENS`.

### Chapters

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/chapters/save` | Save & auto-index a chapter |
| `GET` | `/api/chapters` | List saved chapters |
| `GET` | `/api/chapters/{file}` | Read a chapter |
| `DELETE` | `/api/chapters/{file}` | Delete it and drop it from memory |

### Configuration

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/config` | Current runtime config |
| `GET` | `/api/rules` | Active rules |
| `GET` | `/api/skills` | Active skills |

---

## Customization

### Rules (`rules/*.md`)

Always-on constraints injected into every prompt. The agent concatenates all `.md` files in `rules/` into the system prompt.

Defaults: `tone.md`, `style.md`, `genre.md`. A novel that needs different ones drops a same-named file into `novels/<id>/rules/`, which replaces the project version for that novel only — see [Novels](#novels).

### Skills (`skills/*.md`)

Step-by-step workflows the agent follows during retrieval and generation. Each skill describes what context to fetch and how to verify consistency.

Defaults: `character-consistency.md`, `plot-thread-check.md`, `world-building.md`. Overridable per novel the same way as rules.

### Prompts (`prompts/*.md`)

- `base-prompt.md` — system-level instructions.
- `chapter-prompt.md` — per-generation template with `{{ variable }}` placeholders filled at runtime.

### Context (`novels/<id>/context/*.md`)

One novel's world bible. Edit it in the **World** tab, or drop files into the directory and hit *Re-index everything*. The Zero-Mem engine harvests its entity gazetteer (characters, locations, artefacts) from these files' headings, so keep one `## Heading` per character/place. Retrieval returns contiguous passages under the correct heading during generation.

---

## Novels

Each novel is a self-contained workspace. Retrieval for one can never reach another: the engine, the trace store, the entity graph and the gazetteer are all per-novel.

```
novels/
  do-luc-ky-su/
    novel.json          title, description, created_at
    context/*.md        world bible — the gazetteer is built from this
    chapters/*.md       saved chapters
    rules/*.md          optional; overrides the project rule of the same name
    skills/*.md         optional
    memory/zero_mem.db  isolated traces, entity graph, embeddings
```

**Scoping.** Every story-bearing endpoint takes a novel id — as a `novel` field in POST bodies, or `?novel=<id>` on GET/DELETE. Omitting it means the default workspace, which is what keeps pre-multi-novel URLs working. There is no ambient "current novel" on the server, so two browser tabs can work on two books at once.

**Rules and skills** keep project-level defaults, because "write in third person past tense" is usually a house style rather than a per-book decision. A file of the same name under `novels/<id>/rules/` *replaces* the shared one for that novel — two contradictory tone rules in one system prompt is worse than either alone. `/api/rules` reports each entry's `scope` so you can see what a novel overrides.

**Upgrading.** On first run the default workspace is created and seeded from the old single-novel layout — `context/`, `data/chapters/` and `data/zero_mem.db` are **copied, not moved**, so nothing is relocated out from under you.

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/novels` | List novels with chapter / context counts |
| `POST` | `/api/novels` | Create one (`{title, description}`) |
| `PATCH` | `/api/novels/{id}` | Rename or re-describe |
| `DELETE` | `/api/novels/{id}` | Delete the workspace and everything in it |
| `GET` | `/api/context` | List this novel's world-bible files |
| `GET` | `/api/context/{file}` | Read one |
| `PUT` | `/api/context/{file}` | Write one and index it immediately |
| `DELETE` | `/api/context/{file}` | Delete it and drop it from memory |

---

## Security

- **No secrets in code.** All API keys are loaded from environment variables via `backend/config.py`.
- **`.env` is git-ignored.** Copy `.env.example` (local) or `.env.docker.example` (Docker) and fill in your keys.
- **Non-root container.** The Dockerfile creates and runs as `appuser` (UID 1000).
- **Read-only mounts.** Config directories are mounted `:ro` in Docker.
- **Resource limits.** `docker-compose.yml` caps CPU and memory.

If you ever accidentally expose a key, revoke it immediately in your provider dashboard and rotate.
