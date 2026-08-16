"""
FastAPI server: REST API + SSE streaming for the Novel Generator.

Every story-bearing route is scoped to one novel. The scope arrives as a
``novel`` field (POST bodies) or query parameter (GET/DELETE); omitting it
means the default workspace, which is what keeps pre-multi-novel URLs working.
There is no ambient "current novel" on the server — two browser tabs can work
on two books at once without one silently reassigning the other's scope.
"""

import json
import re
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from backend import history, novels
from backend.agent import NovelAgent
from backend.config import ROOT_DIR, settings
from backend.rag_pipeline import (
    clear_collection,
    get_collection_stats,
    get_engine,
    ingest_directory,
    ingest_file,
    ingest_text,
    release_engine,
    retrieve,
)

app = FastAPI(title="Novel Generator", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

TEMPLATES_DIR = ROOT_DIR / "frontend" / "templates"
STATIC_DIR = ROOT_DIR / "frontend" / "static"


class RevalidatingStatics(StaticFiles):
    """
    Serve static files with `Cache-Control: no-cache`.

    Starlette sends ETag and Last-Modified but no Cache-Control, which leaves
    the browser to guess — and browsers guess "reuse without asking". That
    produced the worst possible skew here: index.html comes from a route below
    and is always fresh, while app.js and style.css came from cache, so a new
    page ran old scripts against markup whose ids had changed and threw on the
    first getElementById that returned null.

    `no-cache` means revalidate, not "don't store": the ETag still turns almost
    every request into a 304, so this costs one round trip, not a re-download.
    """

    def file_response(self, *args, **kwargs):
        response = super().file_response(*args, **kwargs)
        response.headers["Cache-Control"] = "no-cache, must-revalidate"
        return response


if STATIC_DIR.exists():
    app.mount("/static", RevalidatingStatics(directory=str(STATIC_DIR)), name="static")


def _asset_stamp(name: str) -> str:
    """Short fingerprint of a static file, from its size and mtime."""
    try:
        st = (STATIC_DIR / name).stat()
        return "%x-%x" % (int(st.st_mtime), st.st_size)
    except OSError:
        return "0"


# ---------------------------------------------------------------------------
# Novel scoping
# ---------------------------------------------------------------------------

def _novel(slug: Optional[str]) -> novels.Novel:
    """Resolve a novel id to its workspace, or fail with a useful status."""
    try:
        return novels.resolve_or_default(slug, logger=print)
    except novels.NovelNotFound:
        raise HTTPException(404, "No novel with id '%s'." % slug)
    except novels.InvalidNovelName as exc:
        raise HTTPException(400, str(exc))


def is_safe_child_name(filename: str) -> bool:
    """
    Is this a plain filename, with no path meaning on any operating system?

    Pure string logic, deliberately: ``Path`` semantics are not portable and
    this guard must not be either. Starlette percent-decodes path parameters,
    so ``..%5C..%5Cx`` arrives as ``..\\..\\x`` — on Windows those backslashes
    are separators and the join escapes the directory, while on Linux they are
    ordinary filename characters, so nothing escapes but the server cheerfully
    creates a file named ``..\\..\\x``. One input, two behaviours, neither
    wanted, and a containment check alone cannot see the second one. Rejecting
    the string means Windows and Linux answer identically — which is also what
    lets a test run on either platform prove the behaviour on both.
    """
    name = (filename or "").strip()
    if not name or len(name) > 200:
        return False
    if "/" in name or "\\" in name or "\x00" in name:
        return False
    if name.startswith("."):          # covers "." and ".." as well as dotfiles
        return False
    # Reserved on Windows; harmless on Linux, but a shared corpus should not
    # contain a file one of its platforms cannot check out.
    stem = name.split(".")[0].upper()
    if stem in ("CON", "PRN", "AUX", "NUL") or re.match(r"^(COM|LPT)[1-9]$", stem):
        return False
    return True


def _safe_child(directory: Path, filename: str, what: str) -> Path:
    """Resolve ``filename`` inside ``directory``, refusing to escape it."""
    if not is_safe_child_name(filename):
        raise HTTPException(400, "Invalid %s name." % what)

    base = directory.resolve()
    candidate = (base / filename.strip()).resolve()
    # Kept behind the string check as defence in depth: symlinks and odd mount
    # layouts can still resolve a legal-looking name outside the directory.
    if candidate != base and base not in candidate.parents:
        raise HTTPException(400, "Invalid %s name." % what)
    return candidate


# Titles and filenames reach us as free text and are pasted into paths, so a
# separator or dot-segment in one would write outside the workspace.
_SLUG_STRIP_RE = re.compile(r"[^\w\-]+", re.UNICODE)


def _slugify(title: str, limit: int = 40) -> str:
    slug = _SLUG_STRIP_RE.sub("-", title.lower())
    slug = re.sub(r"-{2,}", "-", slug).strip("-")
    return slug[:limit].strip("-") or "untitled"


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class NovelScoped(BaseModel):
    novel: Optional[str] = None


class GenerateRequest(NovelScoped):
    chapter_instructions: str
    story_summary: str = ""
    characters: List[str] = Field(default_factory=list)
    locations: List[str] = Field(default_factory=list)
    # The scene's in-story date (YYYY, YYYY-MM, YYYY-MM-DD). Drives both halves
    # of the guardrail: the model is only served records up to this date, and
    # the finished draft is checked for anachronisms against it.
    scene_date: Optional[str] = None
    target_words: int = settings.default_target_words
    temperature: float = 0.7
    # None => derived from target_words (Vietnamese needs ~2.4 tok/word, so a
    # fixed cap truncated chapters). Pass a number only to override.
    max_tokens: Optional[int] = None
    provider: Optional[str] = None
    model: Optional[str] = None
    stream: bool = False


class ReviseRequest(NovelScoped):
    draft: str
    feedback: str
    temperature: float = 0.5
    max_tokens: Optional[int] = None
    provider: Optional[str] = None
    model: Optional[str] = None
    stream: bool = False


class IngestTextRequest(NovelScoped):
    text: str
    source: str = "manual"
    chapter: str = ""
    characters: List[str] = Field(default_factory=list)
    locations: List[str] = Field(default_factory=list)


class IngestFileRequest(NovelScoped):
    filepath: str
    chapter: str = ""
    characters: List[str] = Field(default_factory=list)
    locations: List[str] = Field(default_factory=list)


class SearchRequest(NovelScoped):
    query: str
    top_k: int = 8


class ChatRequest(NovelScoped):
    message: str
    history: List[dict] = Field(default_factory=list)
    top_k: int = 5
    provider: Optional[str] = None
    model: Optional[str] = None
    stream: bool = True


class SaveChapterRequest(NovelScoped):
    chapter_number: int
    title: str = ""
    content: str


class CreateNovelRequest(BaseModel):
    title: str
    description: str = ""


class UpdateNovelRequest(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None


class ContextFileRequest(NovelScoped):
    content: str


# ---------------------------------------------------------------------------
# Frontend
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = TEMPLATES_DIR / "index.html"
    if not html_path.exists():
        return HTMLResponse("<h1>Novel Generator</h1><p>Frontend not found.</p>")

    html = html_path.read_text(encoding="utf-8")
    # Stamp the asset URLs so markup and scripts can never come from different
    # versions. Editing either file changes its URL, which no cache can answer
    # from a stale copy.
    for name in ("style.css", "app.js"):
        html = html.replace("/static/%s" % name,
                            "/static/%s?v=%s" % (name, _asset_stamp(name)))
    return HTMLResponse(html, headers={"Cache-Control": "no-store"})


# ---------------------------------------------------------------------------
# Novels
# ---------------------------------------------------------------------------

@app.get("/api/novels")
async def list_novels():
    novels.ensure_default(logger=print)
    return {"novels": [n.to_dict() for n in novels.list_novels()],
            "default": settings.default_novel}


@app.post("/api/novels")
async def create_novel(req: CreateNovelRequest):
    try:
        novel = novels.create(req.title, req.description)
    except novels.InvalidNovelName as exc:
        raise HTTPException(400, str(exc))
    return novel.to_dict()


@app.patch("/api/novels/{slug}")
async def update_novel(slug: str, req: UpdateNovelRequest):
    novel = _novel(slug)
    novel.update(title=req.title, description=req.description)
    return novel.to_dict()


@app.delete("/api/novels/{slug}")
async def delete_novel(slug: str):
    if slug == settings.default_novel:
        raise HTTPException(400, "The default workspace cannot be deleted.")
    novel = _novel(slug)
    # Close the SQLite handle first: on Windows the directory cannot be removed
    # while the memory store is still open.
    release_engine(novel.slug)
    try:
        novels.delete(novel.slug)
    except OSError as exc:
        raise HTTPException(500, "Could not delete '%s': %s" % (slug, exc))
    return {"status": "deleted", "slug": slug}


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def _sse(payload: dict) -> str:
    return "data: %s\n\n" % json.dumps(payload)


def _guarded_stream(source):
    """
    Wrap a token stream so a mid-stream failure reaches the client.

    An exception inside a StreamingResponse otherwise just drops the
    connection, and the UI shows a half-written chapter with no explanation.
    """
    async def gen():
        try:
            async for token in source:
                yield _sse({"token": token})
        except Exception as exc:
            yield _sse({"error": str(exc)})
        yield "data: [DONE]\n\n"
    return gen()


@app.post("/api/generate")
async def generate_chapter(req: GenerateRequest):
    novel = _novel(req.novel)
    agent = NovelAgent(provider=req.provider, model=req.model, novel=novel)

    kwargs = dict(
        chapter_instructions=req.chapter_instructions,
        story_summary=req.story_summary,
        characters=req.characters,
        locations=req.locations,
        target_words=req.target_words,
        temperature=req.temperature,
        max_tokens=req.max_tokens,
        scene_date=req.scene_date,
    )

    if req.stream:
        async def event_stream():
            # Research lands before the first token, the check after the last,
            # so the client can show sources up front and conflicts at the end
            # without the token stream carrying anything but prose.
            sidecar = {}
            try:
                stream = agent.generate_chapter_stream(
                    on_research=lambda r: sidecar.setdefault("research", r),
                    on_check=lambda c: sidecar.setdefault("check", c),
                    **kwargs
                )
                first = True
                async for token in stream:
                    if first and "research" in sidecar:
                        yield _sse({"history": {
                            "records": sidecar["research"]["records"],
                            "calls": sidecar["research"]["calls"],
                        }})
                        first = False
                    yield _sse({"token": token})
                if "check" in sidecar:
                    yield _sse({"history_check": sidecar["check"]})
            except Exception as exc:
                yield _sse({"error": str(exc)})
            yield "data: [DONE]\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    result = await agent.generate_chapter(**kwargs)
    result["novel"] = novel.slug
    return result


@app.post("/api/revise")
async def revise_chapter(req: ReviseRequest):
    novel = _novel(req.novel)
    agent = NovelAgent(provider=req.provider, model=req.model, novel=novel)

    kwargs = dict(
        draft=req.draft,
        feedback=req.feedback,
        temperature=req.temperature,
        max_tokens=req.max_tokens,
    )

    if req.stream:
        return StreamingResponse(
            _guarded_stream(agent.revise_chapter_stream(**kwargs)),
            media_type="text/event-stream",
        )
    return {"content": await agent.revise_chapter(**kwargs), "novel": novel.slug}


# ---------------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------------

@app.post("/api/chat")
async def chat_endpoint(req: ChatRequest):
    novel = _novel(req.novel)
    agent = NovelAgent(provider=req.provider, model=req.model, novel=novel)

    if req.stream:
        async def event_stream():
            try:
                stream, sources = await agent.chat_stream(
                    message=req.message, history=req.history, top_k=req.top_k,
                )
                yield _sse({"sources": sources})
                async for token in stream:
                    yield _sse({"token": token})
            except Exception as exc:
                yield _sse({"error": str(exc)})
            yield "data: [DONE]\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    result = await agent.chat(message=req.message, history=req.history, top_k=req.top_k)
    result["novel"] = novel.slug
    return result


# ---------------------------------------------------------------------------
# Memory (ingest / search / stats)
# ---------------------------------------------------------------------------

@app.post("/api/ingest/text")
async def ingest_text_endpoint(req: IngestTextRequest):
    novel = _novel(req.novel)
    count = ingest_text(
        text=req.text, source=req.source, chapter=req.chapter,
        characters=req.characters, locations=req.locations, novel=novel,
    )
    return {"chunks_stored": count, "novel": novel.slug}


@app.post("/api/ingest/file")
async def ingest_file_endpoint(req: IngestFileRequest):
    novel = _novel(req.novel)
    path = Path(req.filepath)
    if not path.exists():
        raise HTTPException(404, "File not found: %s" % req.filepath)
    count = ingest_file(
        filepath=path, chapter=req.chapter, characters=req.characters,
        locations=req.locations, novel=novel,
    )
    return {"chunks_stored": count, "novel": novel.slug}


@app.post("/api/ingest/context")
async def ingest_context_directory(req: Optional[NovelScoped] = None):
    """Ingest every .md file from this novel's own context directory."""
    novel = _novel(req.novel if req else None)
    count = ingest_directory(str(novel.context_dir), novel=novel)
    return {"chunks_stored": count, "novel": novel.slug}


@app.post("/api/search")
async def search_memory(req: SearchRequest):
    novel = _novel(req.novel)
    return {"results": retrieve(query=req.query, top_k=req.top_k, novel=novel),
            "novel": novel.slug}


@app.get("/api/vectordb/stats")
async def vectordb_stats(novel: Optional[str] = Query(None)):
    workspace = _novel(novel)
    stats = get_collection_stats(novel=workspace)
    stats["novel"] = workspace.slug
    return stats


@app.delete("/api/vectordb/clear")
async def vectordb_clear(novel: Optional[str] = Query(None)):
    workspace = _novel(novel)
    clear_collection(novel=workspace)
    return {"status": "cleared", "novel": workspace.slug}


@app.get("/api/memory/entity/{name}")
async def memory_entity(name: str, novel: Optional[str] = Query(None)):
    """What this novel's memory knows about one character/place/item."""
    return get_engine(_novel(novel)).entity_profile(name)


@app.get("/api/memory/sources")
async def memory_sources(novel: Optional[str] = Query(None)):
    """Every ingested document with segment counts, reference vs chapter."""
    return {"sources": get_engine(_novel(novel)).store.sources()}


# ---------------------------------------------------------------------------
# Context files — the world bible, per novel
# ---------------------------------------------------------------------------

@app.get("/api/context")
async def list_context_files(novel: Optional[str] = Query(None)):
    workspace = _novel(novel)
    files = []
    for f in sorted(workspace.context_dir.glob("*.md")):
        text = f.read_text(encoding="utf-8")
        files.append({"filename": f.name, "size": len(text),
                      "preview": text.strip()[:160]})
    return {"files": files, "novel": workspace.slug}


@app.get("/api/context/{filename}")
async def read_context_file(filename: str, novel: Optional[str] = Query(None)):
    workspace = _novel(novel)
    path = _safe_child(workspace.context_dir, filename, "context file")
    if not path.is_file():
        raise HTTPException(404, "Context file not found.")
    return {"filename": path.name, "content": path.read_text(encoding="utf-8")}


@app.put("/api/context/{filename}")
async def write_context_file(filename: str, req: ContextFileRequest):
    workspace = _novel(req.novel)
    if not filename.lower().endswith(".md"):
        filename += ".md"
    path = _safe_child(workspace.context_dir, filename, "context file")
    workspace.context_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(req.content, encoding="utf-8")
    # Re-ingest just this file so the change is retrievable immediately, and
    # refresh the gazetteer in case the edit introduced a new character.
    engine = get_engine(workspace)
    engine.refresh_gazetteer()
    count = engine.ingest_file(str(path))
    return {"filename": path.name, "segments": count, "novel": workspace.slug}


@app.delete("/api/context/{filename}")
async def delete_context_file(filename: str, novel: Optional[str] = Query(None)):
    workspace = _novel(novel)
    path = _safe_child(workspace.context_dir, filename, "context file")
    if not path.is_file():
        raise HTTPException(404, "Context file not found.")
    path.unlink()
    engine = get_engine(workspace)
    engine.delete_source(path.name)
    engine.refresh_gazetteer()
    return {"status": "deleted", "filename": path.name}


# ---------------------------------------------------------------------------
# History — the vetted corpus
# ---------------------------------------------------------------------------

class HistorySearchRequest(NovelScoped):
    query: str
    before: Optional[str] = None
    limit: int = 6


class HistoryCheckRequest(NovelScoped):
    text: str
    scene_date: Optional[str] = None


def _history(novel) -> history.HistoryIndex:
    try:
        return history.get_index(novel)
    except history.HistoryError as exc:
        # A malformed corpus is an author error with a fixable message, not a
        # server fault — and never a reason to serve unvetted claims instead.
        raise HTTPException(422, "Sử liệu không hợp lệ: %s" % exc)


@app.post("/api/history/search")
async def history_search(req: HistorySearchRequest):
    """The same closed-corpus search the model's tool calls."""
    workspace = _novel(req.novel)
    index = _history(workspace)
    try:
        hits = index.search(req.query, limit=max(1, min(req.limit, 20)), before=req.before)
    except history.HistoryError as exc:
        raise HTTPException(400, str(exc))
    return {"results": hits, "corpus_size": len(index), "novel": workspace.slug}


@app.get("/api/history")
async def history_list(novel: Optional[str] = Query(None),
                       start: Optional[str] = Query(None),
                       end: Optional[str] = Query(None)):
    """Every record in chronological order, for review."""
    workspace = _novel(novel)
    index = _history(workspace)
    try:
        records = index.timeline(start, end, limit=1000)
    except history.HistoryError as exc:
        raise HTTPException(400, str(exc))
    undated = [index._hit(r, 0.0) for r in index.records if r.start is None]
    return {"records": records + undated, "corpus_size": len(index),
            "novel": workspace.slug}


@app.post("/api/history/check")
async def history_check(req: HistoryCheckRequest):
    """Check a draft for anachronisms and locked-history conflicts."""
    workspace = _novel(req.novel)
    result = _history(workspace).check(req.text, req.scene_date)
    result["novel"] = workspace.slug
    return result


@app.post("/api/history/reload")
async def history_reload(req: Optional[NovelScoped] = None):
    """Re-read the corpus from disk after editing the YAML."""
    workspace = _novel(req.novel if req else None)
    try:
        index = history.reload_index(workspace)
    except history.HistoryError as exc:
        raise HTTPException(422, "Sử liệu không hợp lệ: %s" % exc)
    return {"corpus_size": len(index), "novel": workspace.slug}


# ---------------------------------------------------------------------------
# Chapters
# ---------------------------------------------------------------------------

@app.post("/api/chapters/save")
async def save_chapter(req: SaveChapterRequest):
    novel = _novel(req.novel)
    novel.chapters_dir.mkdir(parents=True, exist_ok=True)

    filename = "chapter-%03d-%s.md" % (req.chapter_number, _slugify(req.title))
    filepath = novel.chapters_dir / filename

    header = "# Chapter %d" % req.chapter_number
    if req.title:
        header += ": %s" % req.title
    filepath.write_text("%s\n\n%s" % (header, req.content), encoding="utf-8")

    count = ingest_text(
        text=req.content, source=filename,
        chapter=str(req.chapter_number), novel=novel,
    )
    return {"filepath": str(filepath), "filename": filename,
            "chunks_ingested": count, "novel": novel.slug}


@app.get("/api/chapters")
async def list_chapters(novel: Optional[str] = Query(None)):
    workspace = _novel(novel)
    if not workspace.chapters_dir.exists():
        return {"chapters": [], "novel": workspace.slug}

    chapters = []
    for f in sorted(workspace.chapters_dir.glob("chapter-*.md")):
        text = f.read_text(encoding="utf-8")
        first_line = text.split("\n")[0] if text else f.stem
        words = len([w for w in text.split() if w])
        chapters.append({
            "filename": f.name,
            "title": first_line.lstrip("# "),
            "size": len(text),
            "words": words,
        })
    return {"chapters": chapters, "novel": workspace.slug}


@app.get("/api/chapters/{filename}")
async def get_chapter(filename: str, novel: Optional[str] = Query(None)):
    workspace = _novel(novel)
    filepath = _safe_child(workspace.chapters_dir, filename, "chapter")
    if not filepath.is_file():
        raise HTTPException(404, "Chapter not found")
    return {"filename": filename, "content": filepath.read_text(encoding="utf-8")}


@app.delete("/api/chapters/{filename}")
async def delete_chapter(filename: str, novel: Optional[str] = Query(None)):
    workspace = _novel(novel)
    filepath = _safe_child(workspace.chapters_dir, filename, "chapter")
    if not filepath.is_file():
        raise HTTPException(404, "Chapter not found")
    filepath.unlink()
    # Drop it from memory too, or the deleted chapter stays retrievable.
    get_engine(workspace).delete_source(filepath.name)
    return {"status": "deleted", "filename": filepath.name}


# ---------------------------------------------------------------------------
# Config / rules / skills
# ---------------------------------------------------------------------------

@app.get("/api/config")
async def get_config():
    return {
        "default_provider": settings.default_llm_provider,
        "default_model": settings.default_model,
        "embedding_provider": settings.embedding_provider,
        "embedding_model": settings.embedding_model,
        "target_words": settings.default_target_words,
        "default_novel": settings.default_novel,
        "has_anthropic_key": bool(settings.anthropic_api_key),
        "has_openai_key": bool(settings.openai_api_key),
        "has_groq_key": bool(settings.groq_api_key),
        "has_gemini_key": bool(settings.gemini_api_key or settings.google_api_key),
    }


def _markdown_entries(project_dir, novel_dir) -> List[dict]:
    """
    Project-level files, overridden by same-named files in the novel.

    Mirrors what agent._collect_markdown feeds the model, and reports which
    scope each entry came from so the UI can show what this novel overrides.
    """
    by_name = {}
    for directory, scope in ((project_dir, "project"), (novel_dir, "novel")):
        if directory is None:
            continue
        path = Path(directory)
        if not path.is_dir():
            continue
        for f in sorted(path.glob("*.md")):
            by_name[f.stem] = {"name": f.stem,
                               "content": f.read_text(encoding="utf-8"),
                               "scope": scope}
    return [by_name[k] for k in sorted(by_name)]


@app.get("/api/rules")
async def get_rules(novel: Optional[str] = Query(None)):
    workspace = _novel(novel)
    return {"rules": _markdown_entries(settings.rules_dir, workspace.rules_dir),
            "novel": workspace.slug}


@app.get("/api/skills")
async def get_skills(novel: Optional[str] = Query(None)):
    workspace = _novel(novel)
    return {"skills": _markdown_entries(settings.skills_dir, workspace.skills_dir),
            "novel": workspace.slug}
