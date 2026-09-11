# Novel Generator MCP server

A small [MCP](https://modelcontextprotocol.io) server that lets an MCP host
(Claude Code, Claude Desktop, any other client) read a Novel Generator
workspace: the story state, chapters, world bible, memory, the vetted history
corpus, and the writing rules. It runs as a separate stdio process and talks
to the app over HTTP. **It is read-only by design** — see below.

All story content is Vietnamese; every tool description says so, and the two
bundled prompts ask the model to cite history record ids and never to invent
history the tools did not return.

## Why a separate Python 3.12 venv

The app runs on Python 3.9. The `mcp` SDK requires 3.10+. The two cannot share
an interpreter, so this directory has its own venv (`mcp_server/.venv`) and
its own `requirements.txt`. Nothing under `backend/` is imported from here.

## Why it proxies over HTTP instead of importing the backend

Importing `backend` would look simpler, but opening the memory engine
*writes*: it embeds any vectors that are missing and runs schema/WAL
migrations on open. MCP hosts launch servers with a minimal environment that
does not include the app's `.env`, so a direct import would open the store
with the default embedder, discover that every vector "is missing" under that
embedder, and re-embed the whole corpus with the wrong model — silently
corrupting the memory the app relies on. Going through the running app means
the app's own configuration, and only the app's, ever touches the store.

## Why it is read-only

There is no tool that generates, revises, chats, saves a chapter, ingests,
clears the vector store, reloads the history corpus, or issues any
`PUT`/`PATCH`/`DELETE`. An agent that could save or clear from inside a chat
turn is a liability the author never asked for; those actions happen in the
app's UI, where the author sees them. The test suite records every request
each tool makes and fails if any of them is a write or hits one of those
paths, so the promise is enforced, not just stated.

## Setup (Windows)

```
py -3.12 -m venv mcp_server\.venv
mcp_server\.venv\Scripts\python.exe -m pip install -r mcp_server\requirements.txt
```

This installs `mcp>=1.10`: the tools return their dicts as structured tool
output, which the SDK added in 1.10.0, so an older `mcp` will not do.

The app must be running for any tool to return data:

```
python main.py
```

(from the repo root, with the app's 3.9 Python). The server reads
`NOVELGEN_URL` (default `http://127.0.0.1:8000`) to find it. When the app is
not running every tool returns
`{"error": "Novel Generator app is not running at <url>. Start it with: python main.py"}`.

### Register with Claude Code

```
claude mcp add novel-generator -- D:/RiSet/Novel-Generator/mcp_server/.venv/Scripts/python.exe D:/RiSet/Novel-Generator/mcp_server/server.py
```

### Register with Claude Desktop

Add to `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "novel-generator": {
      "command": "D:/RiSet/Novel-Generator/mcp_server/.venv/Scripts/python.exe",
      "args": ["D:/RiSet/Novel-Generator/mcp_server/server.py"],
      "env": { "NOVELGEN_URL": "http://127.0.0.1:8000" }
    }
  }
}
```

## Prompts

| Prompt | Arguments | What it makes the model do |
|---|---|---|
| `suggest_next_chapter` | `novel` | Call `get_story_state` first, then `search_history` with `before=latest_scene_date`, then propose 2–3 concrete next-chapter directions that fit the next arc beat and the declared divergences, citing history record ids and never inventing history the tools did not return. |
| `review_chapter` | `novel`, `filename` | Read the chapter with `read_chapter` (all parts), run `check_history` with the chapter's `meta.scene_date`, call `entity_profile` for each named character, and only then comment on prose, pacing and arc fit. Deterministic findings come first, with record ids. |

## Tools

All read-only. `novel` is the slug from `list_novels`.

| Tool | Proxies | Returns |
|---|---|---|
| `list_novels()` | `GET /api/novels` | every workspace with slug, title, counts |
| `get_story_state(novel)` | `GET /api/novels/{novel}/state` | novel, chapters, latest chapter tail, latest scene date, outline, divergences, memory stats, rule/skill names |
| `list_chapters(novel)` | `GET /api/chapters` | filename, number, title, words, scene_date, arc, summary |
| `read_chapter(novel, filename, part=1)` | `GET /api/chapters/{filename}` | the body in parts of ≤1500 words, split on paragraph boundaries (a paragraph longer than that is cut on its line breaks, then on words): `{filename, meta, part, parts, words_in_part, text}` |
| `read_world(novel, filename=None)` | `GET /api/context` / `GET /api/context/{filename}` | the world-bible file list, or one file |
| `search_memory(novel, query, top_k=8)` | `POST /api/search` | semantic hits over chapters and world files |
| `entity_profile(novel, name)` | `GET /api/memory/entity/{name}` | what memory knows about one character/place/item |
| `get_writer_context(novel, query, characters=[], locations=[])` | `POST /api/memory/context` | exactly the context block the app's writer would be handed |
| `search_history(novel, query, before=None, limit=6)` | `POST /api/history/search` | vetted history records with ids; `before` limits to what had happened by that date |
| `check_history(novel, text, scene_date=None)` | `POST /api/history/check` | anachronisms and locked-history conflicts, with record ids |
| `get_rules_and_skills(novel)` | `GET /api/rules` + `GET /api/skills` | `{rules: [...], skills: [...]}` |

Resources: `novel://{slug}/chapters/{filename}` (a chapter body) and
`novel://{slug}/world/{filename}` (a world file), both `text/markdown`.

Errors are plain dicts, never exceptions: `{"error": <detail>, "status": <code>}`
for HTTP errors, and the "not running" message above when the app is
unreachable.

## Tests and smoke check

Unit tests (no app needed — HTTP is mocked). They need `pytest`, which the
runtime install above deliberately leaves out, so install the dev
requirements first:

```
mcp_server\.venv\Scripts\python.exe -m pip install -r mcp_server\requirements-dev.txt
mcp_server\.venv\Scripts\python.exe -m pytest mcp_server\tests -q
```

Live end-to-end check (app must be running): spawns the server over stdio,
lists its tools, calls `list_novels` and `get_story_state`:

```
mcp_server\.venv\Scripts\python.exe mcp_server\smoke.py
```
