"""
FastAPI server: REST API + SSE streaming for the Novel Generator.
"""

import json
import re
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from backend.config import settings, ROOT_DIR
from backend.agent import NovelAgent
from backend.rag_pipeline import (
    ingest_text,
    ingest_file,
    ingest_directory,
    retrieve,
    get_collection_stats,
    clear_collection,
    get_engine,
)


app = FastAPI(title="Novel Generator", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

TEMPLATES_DIR = ROOT_DIR / "frontend" / "templates"
STATIC_DIR = ROOT_DIR / "frontend" / "static"

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class GenerateRequest(BaseModel):
    chapter_instructions: str
    story_summary: str = ""
    characters: List[str] = Field(default_factory=list)
    locations: List[str] = Field(default_factory=list)
    target_words: int = settings.default_target_words
    temperature: float = 0.7
    # None => derived from target_words (Vietnamese needs ~2.4 tok/word, so a
    # fixed cap truncated chapters). Pass a number only to override.
    max_tokens: Optional[int] = None
    provider: Optional[str] = None
    model: Optional[str] = None
    stream: bool = False


class ReviseRequest(BaseModel):
    draft: str
    feedback: str
    temperature: float = 0.5
    max_tokens: Optional[int] = None
    provider: Optional[str] = None
    model: Optional[str] = None
    stream: bool = False


class IngestTextRequest(BaseModel):
    text: str
    source: str = "manual"
    chapter: str = ""
    characters: List[str] = Field(default_factory=list)
    locations: List[str] = Field(default_factory=list)


class IngestFileRequest(BaseModel):
    filepath: str
    chapter: str = ""
    characters: List[str] = Field(default_factory=list)
    locations: List[str] = Field(default_factory=list)


class SearchRequest(BaseModel):
    query: str
    top_k: int = 8


class ChatRequest(BaseModel):
    message: str
    history: List[dict] = Field(default_factory=list)
    top_k: int = 5
    provider: Optional[str] = None
    model: Optional[str] = None
    stream: bool = True


class SaveChapterRequest(BaseModel):
    chapter_number: int
    title: str = ""
    content: str


# ---------------------------------------------------------------------------
# Frontend routes
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = TEMPLATES_DIR / "index.html"
    if html_path.exists():
        return HTMLResponse(html_path.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>Novel Generator</h1><p>Frontend not found.</p>")


# ---------------------------------------------------------------------------
# Generation endpoints
# ---------------------------------------------------------------------------

@app.post("/api/generate")
async def generate_chapter(req: GenerateRequest):
    agent = NovelAgent(provider=req.provider, model=req.model)

    if req.stream:
        async def event_stream():
            async for token in agent.generate_chapter_stream(
                chapter_instructions=req.chapter_instructions,
                story_summary=req.story_summary,
                characters=req.characters,
                locations=req.locations,
                target_words=req.target_words,
                temperature=req.temperature,
                max_tokens=req.max_tokens,
            ):
                yield f"data: {json.dumps({'token': token})}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    text = await agent.generate_chapter(
        chapter_instructions=req.chapter_instructions,
        story_summary=req.story_summary,
        characters=req.characters,
        locations=req.locations,
        target_words=req.target_words,
        temperature=req.temperature,
        max_tokens=req.max_tokens,
    )
    return {"content": text}


@app.post("/api/revise")
async def revise_chapter(req: ReviseRequest):
    agent = NovelAgent(provider=req.provider, model=req.model)

    if req.stream:
        async def event_stream():
            async for token in agent.revise_chapter_stream(
                draft=req.draft,
                feedback=req.feedback,
                temperature=req.temperature,
                max_tokens=req.max_tokens,
            ):
                yield f"data: {json.dumps({'token': token})}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    text = await agent.revise_chapter(
        draft=req.draft,
        feedback=req.feedback,
        temperature=req.temperature,
        max_tokens=req.max_tokens,
    )
    return {"content": text}


# ---------------------------------------------------------------------------
# Chat (RAG Q&A)
# ---------------------------------------------------------------------------

@app.post("/api/chat")
async def chat_endpoint(req: ChatRequest):
    agent = NovelAgent(provider=req.provider, model=req.model)

    if req.stream:
        async def event_stream():
            stream, sources = await agent.chat_stream(
                message=req.message,
                history=req.history,
                top_k=req.top_k,
            )
            yield f"data: {json.dumps({'sources': sources})}\n\n"
            async for token in stream:
                yield f"data: {json.dumps({'token': token})}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    result = await agent.chat(
        message=req.message,
        history=req.history,
        top_k=req.top_k,
    )
    return result


# ---------------------------------------------------------------------------
# RAG / VectorDB endpoints
# ---------------------------------------------------------------------------

@app.post("/api/ingest/text")
async def ingest_text_endpoint(req: IngestTextRequest):
    count = ingest_text(
        text=req.text,
        source=req.source,
        chapter=req.chapter,
        characters=req.characters,
        locations=req.locations,
    )
    return {"chunks_stored": count}


@app.post("/api/ingest/file")
async def ingest_file_endpoint(req: IngestFileRequest):
    path = Path(req.filepath)
    if not path.exists():
        raise HTTPException(404, f"File not found: {req.filepath}")
    count = ingest_file(
        filepath=path,
        chapter=req.chapter,
        characters=req.characters,
        locations=req.locations,
    )
    return {"chunks_stored": count}


@app.post("/api/ingest/context")
async def ingest_context_directory():
    """Ingest all files from the context/ directory."""
    count = ingest_directory(settings.context_dir)
    return {"chunks_stored": count}


@app.post("/api/search")
async def search_vectordb(req: SearchRequest):
    hits = retrieve(query=req.query, top_k=req.top_k)
    return {"results": hits}


@app.get("/api/vectordb/stats")
async def vectordb_stats():
    return get_collection_stats()


@app.delete("/api/vectordb/clear")
async def vectordb_clear():
    clear_collection()
    return {"status": "cleared"}


@app.get("/api/memory/entity/{name}")
async def memory_entity(name: str):
    """What the story memory knows about one character/place/item."""
    return get_engine().entity_profile(name)


@app.get("/api/memory/sources")
async def memory_sources():
    """Every ingested document with segment counts, reference vs chapter."""
    return {"sources": get_engine().store.sources()}


# ---------------------------------------------------------------------------
# Chapter management
# ---------------------------------------------------------------------------

# Titles reach us as free text and are pasted straight into a filename, so a
# separator or a dot-segment in one would write outside chapters_dir. Keep
# letters (any script — titles here are Vietnamese), digits, and dashes.
_SLUG_STRIP_RE = re.compile(r"[^\w\-]+", re.UNICODE)


def _slugify(title: str, limit: int = 40) -> str:
    slug = _SLUG_STRIP_RE.sub("-", title.lower())
    slug = re.sub(r"-{2,}", "-", slug).strip("-")
    return slug[:limit].strip("-") or "untitled"


@app.post("/api/chapters/save")
async def save_chapter(req: SaveChapterRequest):
    chapters_dir = Path(settings.chapters_dir)
    chapters_dir.mkdir(parents=True, exist_ok=True)

    filename = f"chapter-{req.chapter_number:03d}-{_slugify(req.title)}.md"
    filepath = chapters_dir / filename

    header = f"# Chapter {req.chapter_number}"
    if req.title:
        header += f": {req.title}"

    filepath.write_text(f"{header}\n\n{req.content}", encoding="utf-8")

    count = ingest_text(
        text=req.content,
        source=filename,
        chapter=str(req.chapter_number),
    )

    return {"filepath": str(filepath), "chunks_ingested": count}


@app.get("/api/chapters")
async def list_chapters():
    chapters_dir = Path(settings.chapters_dir)
    if not chapters_dir.exists():
        return {"chapters": []}

    chapters = []
    for f in sorted(chapters_dir.glob("chapter-*.md")):
        text = f.read_text(encoding="utf-8")
        first_line = text.split("\n")[0] if text else f.stem
        chapters.append({
            "filename": f.name,
            "title": first_line.lstrip("# "),
            "size": len(text),
        })
    return {"chapters": chapters}


def _resolve_chapter(filename: str) -> Path:
    """
    Resolve a chapter filename inside chapters_dir, refusing to escape it.

    Without this, ``GET /api/chapters/..%5C..%5Csecrets.env`` read arbitrary
    files: Starlette percent-decodes the path parameter, and on Windows
    ``Path`` treats the decoded backslash as a separator, so the join walked
    straight out of the directory. The server binds 0.0.0.0 by default, which
    makes that reachable from the network.
    """
    base = Path(settings.chapters_dir).resolve()
    candidate = (base / filename).resolve()
    if candidate != base and base not in candidate.parents:
        raise HTTPException(400, "Invalid chapter filename")
    return candidate


@app.get("/api/chapters/{filename}")
async def get_chapter(filename: str):
    filepath = _resolve_chapter(filename)
    if not filepath.is_file():
        raise HTTPException(404, "Chapter not found")
    return {"filename": filename, "content": filepath.read_text(encoding="utf-8")}


# ---------------------------------------------------------------------------
# Config / status
# ---------------------------------------------------------------------------

@app.get("/api/config")
async def get_config():
    return {
        "default_provider": settings.default_llm_provider,
        "default_model": settings.default_model,
        "embedding_model": settings.embedding_model,
        "target_words": settings.default_target_words,
        "has_anthropic_key": bool(settings.anthropic_api_key),
        "has_openai_key": bool(settings.openai_api_key),
        "has_groq_key": bool(settings.groq_api_key),
    }


@app.get("/api/rules")
async def get_rules():
    rules_dir = Path(settings.rules_dir)
    if not rules_dir.exists():
        return {"rules": []}
    rules = []
    for f in sorted(rules_dir.glob("*.md")):
        rules.append({"name": f.stem, "content": f.read_text(encoding="utf-8")})
    return {"rules": rules}


@app.get("/api/skills")
async def get_skills():
    skills_dir = Path(settings.skills_dir)
    if not skills_dir.exists():
        return {"skills": []}
    skills = []
    for f in sorted(skills_dir.glob("*.md")):
        skills.append({"name": f.stem, "content": f.read_text(encoding="utf-8")})
    return {"skills": skills}
