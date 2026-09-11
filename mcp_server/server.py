# -*- coding: utf-8 -*-
"""
Novel Generator MCP server — a read-only stdio proxy over the running app.

Everything here goes over HTTP to the FastAPI app (``python main.py``); nothing
imports ``backend``. That is deliberate: opening the memory engine writes (it
embeds missing vectors and runs schema/WAL migrations on open), and MCP hosts
launch servers without the app's ``.env``, so importing the backend from here
would re-embed the corpus with the wrong embedder. The proxy also lets the
server live in its own Python 3.12 venv while the app stays on 3.9.

There are no write tools, by design. An agent that can save chapters or clear
the vector store from inside a chat turn is a liability the author never
asked for; the app's UI is the only place those actions happen. See README.md.
"""

import functools
import os
import re
from typing import Any, Callable, Optional
from urllib.parse import quote

import httpx
from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations

DEFAULT_URL = "http://127.0.0.1:8000"
TIMEOUT_SECONDS = 60.0
# Long enough that a chapter rarely needs more than three parts, short enough
# that one part never blows a host's tool-result budget.
MAX_PART_WORDS = 1500

# Every tool is a pure read. Hosts use the hint to skip confirmation prompts,
# and it is a public promise the test suite enforces (no PUT/PATCH/DELETE, no
# generate/save/ingest paths).
READ_ONLY = ToolAnnotations(readOnlyHint=True, destructiveHint=False,
                            idempotentHint=True, openWorldHint=False)

mcp = FastMCP(
    "novel-generator",
    instructions=(
        "Read-only access to a Novel Generator workspace. All story content, "
        "world files and history records are in Vietnamese. When a tool "
        "returns history records, cite their ids in your answer and never "
        "invent history the tools did not return. Start with get_story_state."
    ),
)


# ---------------------------------------------------------------------------
# HTTP — one lazily created client so tests can swap in a MockTransport
# ---------------------------------------------------------------------------

_client: Optional[httpx.Client] = None
_transport: Optional[httpx.BaseTransport] = None


def base_url() -> str:
    return os.environ.get("NOVELGEN_URL", DEFAULT_URL).rstrip("/")


def set_transport(transport: Optional[httpx.BaseTransport]) -> None:
    """
    Replace the transport the next request uses (tests pass an
    ``httpx.MockTransport``; ``None`` restores the real network).

    The existing client is discarded rather than mutated because ``base_url``
    is fixed at construction: a test that changes NOVELGEN_URL and then swaps
    the transport must not keep talking to the old origin.
    """
    global _client, _transport
    _transport = transport
    if _client is not None:
        _client.close()
    _client = None


def client() -> httpx.Client:
    global _client
    if _client is None:
        _client = httpx.Client(base_url=base_url(), timeout=TIMEOUT_SECONDS,
                               transport=_transport)
    return _client


def _not_running() -> dict:
    return {"error": "Novel Generator app is not running at %s. "
                     "Start it with: python main.py" % base_url()}


def _detail(resp: httpx.Response) -> str:
    """The app's ``{"detail": ...}`` message when there is one, else status text."""
    try:
        body = resp.json()
    except ValueError:
        body = None
    detail = body.get("detail") if isinstance(body, dict) else None
    if isinstance(detail, str) and detail:
        return detail
    if isinstance(detail, list) and detail:
        # FastAPI validation errors: a list of {loc, msg, type}. Flatten so the
        # agent sees "body.query: field required" instead of a JSON blob.
        bits = []
        for item in detail:
            if isinstance(item, dict):
                loc = ".".join(str(x) for x in item.get("loc", []))
                bits.append("%s: %s" % (loc, item.get("msg", "")) if loc else str(item.get("msg", item)))
            else:
                bits.append(str(item))
        return "; ".join(bits)
    return resp.reason_phrase or "HTTP %d" % resp.status_code


def _request(method: str, path: str, params: Optional[dict] = None,
             json: Optional[dict] = None) -> dict:
    """
    One HTTP call mapped to the contract's error dicts. Nothing raises: a
    tool that throws becomes an opaque "tool error" in the host, while a dict
    with ``error`` is something the agent can read and act on.
    """
    try:
        resp = client().request(method, path, params=params, json=json)
    except (httpx.ConnectError, httpx.TimeoutException):
        return _not_running()
    except httpx.HTTPError as exc:
        return {"error": "%s talking to %s: %s" % (type(exc).__name__, base_url(), exc)}
    if resp.status_code >= 400:
        return {"error": _detail(resp), "status": resp.status_code}
    try:
        data = resp.json()
    except ValueError:
        return {"error": "Non-JSON response from %s" % path, "status": resp.status_code}
    if not isinstance(data, dict):
        # The app only ever returns objects; anything else is a proxy in
        # front of it (a login page, a 200 HTML error) and must not be
        # mistaken for story data.
        return {"error": "Unexpected response shape from %s" % path, "status": resp.status_code}
    return data


def _get(path: str, params: Optional[dict] = None) -> dict:
    return _request("GET", path, params=params)


def _post(path: str, json: Optional[dict] = None) -> dict:
    return _request("POST", path, json=json)


def _failed(data: dict) -> bool:
    return "error" in data


def _guarded(fn: Callable[..., dict]) -> Callable[..., dict]:
    """
    Last line of defence for "never let an exception escape a tool": anything
    ``_request`` did not already map (a KeyError on an unexpected payload, a
    bad argument) becomes an error dict instead of a protocol-level failure.
    ``functools.wraps`` keeps ``__wrapped__`` so FastMCP still reads the real
    signature and docstring for the tool schema.
    """
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001 - the whole point
            return {"error": "%s: %s" % (type(exc).__name__, exc)}
    return wrapper


# ---------------------------------------------------------------------------
# Chapter splitting — pure logic, unit-tested on its own
# ---------------------------------------------------------------------------

_PARAGRAPH_BREAK = re.compile(r"\n[ \t\r]*\n")


def _pieces(para: str, max_words: int) -> list:
    """
    One oversized paragraph as pieces that each fit the cap: on its own line
    breaks where it has them, and on word boundaries only for a line that is
    itself too long. The manual Write editor saves Enter as a single "\\n",
    so a whole chapter typed that way is one "paragraph" to the blank-line
    split — the case this exists for.
    """
    out = []
    for line in para.split("\n"):
        words = line.split()
        if len(words) <= max_words:
            if words:
                out.append(line.strip())
            continue
        for i in range(0, len(words), max_words):
            out.append(" ".join(words[i:i + max_words]))
    return out


def split_parts(body: str, max_words: int = MAX_PART_WORDS) -> list:
    """
    Cut a chapter body into parts of at most ``max_words`` words, breaking on
    blank lines. A paragraph is never split across parts unless it alone
    exceeds the cap — then the cap wins (see ``_pieces``): the cap is what
    keeps one part inside a host's tool-result budget, and a 3500-word
    "paragraph" handed back whole was exactly the blow-up it exists to stop.

    An empty body is one empty part, so ``part=1`` is always valid.
    """
    paragraphs = [p.strip() for p in _PARAGRAPH_BREAK.split(body.strip())]
    paragraphs = [p for p in paragraphs if p]
    if not paragraphs:
        return [""]
    # (text, word count, what joins it to the unit before it). Pieces cut
    # from one paragraph rejoin with the single "\n" the author wrote there,
    # not a blank line that would read as a paragraph break the file lacks.
    units = []
    for para in paragraphs:
        n = len(para.split())
        if n <= max_words:
            units.append((para, n, "\n\n"))
            continue
        for i, piece in enumerate(_pieces(para, max_words)):
            units.append((piece, len(piece.split()), "\n\n" if i == 0 else "\n"))
    parts, current, count = [], "", 0
    for text, n, joiner in units:
        if current and count + n > max_words:
            parts.append(current)
            current, count = "", 0
        current = current + joiner + text if current else text
        count += n
    parts.append(current)
    return parts


# ---------------------------------------------------------------------------
# Tools — all read-only
# ---------------------------------------------------------------------------

@mcp.tool(annotations=READ_ONLY)
@_guarded
def list_novels() -> dict[str, Any]:
    """
    List every novel workspace with its slug, title and content counts.
    Titles and descriptions are Vietnamese. Use the slug as the ``novel``
    argument of every other tool.
    """
    return _get("/api/novels")


@mcp.tool(annotations=READ_ONLY)
@_guarded
def get_story_state(novel: str) -> dict[str, Any]:
    """
    The composite snapshot to call first: novel info, the chapter list with
    scene dates and summaries, the latest chapter's tail, the most recent
    scene date, the outline, declared history divergences (with record ids),
    memory stats, and the rule/skill names. All prose is Vietnamese. Cite the
    divergence record ids when you reason about them.
    """
    return _get("/api/novels/%s/state" % quote(novel, safe=""))


@mcp.tool(annotations=READ_ONLY)
@_guarded
def list_chapters(novel: str) -> dict[str, Any]:
    """
    Saved chapters in order: filename, number, Vietnamese title, word count,
    scene_date, arc and summary. Read one with read_chapter(novel, filename).
    """
    return _get("/api/chapters", params={"novel": novel})


@mcp.tool(annotations=READ_ONLY)
@_guarded
def read_chapter(novel: str, filename: str, part: int = 1) -> dict[str, Any]:
    """
    Read one chapter's Vietnamese body in parts of at most 1500 words, split
    on paragraph boundaries (a paragraph longer than that is cut on its line
    breaks, then on words). Returns {filename, meta, part, parts,
    words_in_part, text}; call again with part=2..parts to read the rest.
    ``meta`` is the chapter's front matter (scene_date, arc, summary) and is
    empty for chapters saved before front matter existed.
    """
    data = _get("/api/chapters/%s" % quote(filename, safe=""), params={"novel": novel})
    if _failed(data):
        return data
    parts = split_parts(data.get("content") or "")
    if part < 1 or part > len(parts):
        return {"error": "part %d is out of range: %s has %d part(s)"
                         % (part, filename, len(parts))}
    text = parts[part - 1]
    return {
        "filename": data.get("filename", filename),
        "meta": data.get("meta") or {},
        "part": part,
        "parts": len(parts),
        "words_in_part": len(text.split()),
        "text": text,
    }


@mcp.tool(annotations=READ_ONLY)
@_guarded
def read_world(novel: str, filename: Optional[str] = None) -> dict[str, Any]:
    """
    The world bible (context files, Vietnamese markdown). Without ``filename``
    lists the files with a preview; with one returns that file's content.
    """
    if filename is None:
        return _get("/api/context", params={"novel": novel})
    return _get("/api/context/%s" % quote(filename, safe=""), params={"novel": novel})


@mcp.tool(annotations=READ_ONLY)
@_guarded
def search_memory(novel: str, query: str, top_k: int = 8) -> dict[str, Any]:
    """
    Semantic search over everything ingested (chapters and world files),
    returning the best Vietnamese passages with their sources. Phrase the
    query in Vietnamese for the best matches.
    """
    return _post("/api/search", json={"novel": novel, "query": query, "top_k": top_k})


@mcp.tool(annotations=READ_ONLY)
@_guarded
def entity_profile(novel: str, name: str) -> dict[str, Any]:
    """
    What the novel's memory knows about one character, place or item: its
    mentions, relations and the Vietnamese passages they come from. ``name``
    is matched as written in the text (diacritics included).
    """
    return _get("/api/memory/entity/%s" % quote(name, safe=""), params={"novel": novel})


@mcp.tool(annotations=READ_ONLY)
@_guarded
def get_writer_context(novel: str, query: str, characters: Optional[list[str]] = None,
                       locations: Optional[list[str]] = None) -> dict[str, Any]:
    """
    Exactly the Vietnamese context block the app's writer would be handed for
    this query, composed with the same retrieval query the generator uses.
    Use it to see what the model would and would not know before proposing a
    scene. Returns {context, used, route, profile, tokens, novel}.
    """
    return _post("/api/memory/context", json={
        "novel": novel, "query": query,
        "characters": characters or [], "locations": locations or [],
    })


@mcp.tool(annotations=READ_ONLY)
@_guarded
def search_history(novel: str, query: str, before: Optional[str] = None,
                   limit: int = 6) -> dict[str, Any]:
    """
    Search the vetted history corpus (Vietnamese records with ids and dates).
    ``before`` (YYYY, YYYY-MM or YYYY-MM-DD) restricts results to what had
    already happened by that date. Cite the returned record ids; do not
    invent records the corpus did not return.
    """
    return _post("/api/history/search", json={
        "novel": novel, "query": query, "before": before, "limit": limit,
    })


@mcp.tool(annotations=READ_ONLY)
@_guarded
def check_history(novel: str, text: str, scene_date: Optional[str] = None) -> dict[str, Any]:
    """
    Check a Vietnamese draft for anachronisms and conflicts with locked
    history, optionally against the scene's date. Findings reference history
    record ids; report them with those ids.
    """
    return _post("/api/history/check", json={
        "novel": novel, "text": text, "scene_date": scene_date,
    })


@mcp.tool(annotations=READ_ONLY)
@_guarded
def get_rules_and_skills(novel: str) -> dict[str, Any]:
    """
    The writing rules and skills (Vietnamese markdown) the generator follows,
    project-level ones overridden by the novel's own, as
    {rules: [{name, scope, content}], skills: [...]}.
    """
    rules = _get("/api/rules", params={"novel": novel})
    if _failed(rules):
        return rules
    skills = _get("/api/skills", params={"novel": novel})
    if _failed(skills):
        return skills
    return {"rules": rules.get("rules", []), "skills": skills.get("skills", []),
            "novel": rules.get("novel", novel)}


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

@mcp.prompt()
def suggest_next_chapter(novel: str) -> str:
    """
    Propose 2-3 concrete directions for the next chapter of a Vietnamese
    novel, grounded in the story state and the vetted history (cite record
    ids).
    """
    return (
        "Bạn là biên tập viên phát triển cốt truyện cho tiểu thuyết '%(novel)s' "
        "(nội dung tiếng Việt). Làm đúng theo thứ tự sau:\n"
        "\n"
        "1. Gọi get_story_state(novel=\"%(novel)s\") trước. Đọc kỹ: chương mới "
        "nhất (latest.tail), latest_scene_date, dàn ý (outline), các divergences "
        "đã khai báo (có id bản ghi), và danh sách chapters với summary của "
        "từng chương.\n"
        "2. Gọi search_history(novel=\"%(novel)s\", query=..., "
        "before=<latest_scene_date>) với một hoặc vài truy vấn về những sự kiện "
        "lịch sử có thể xảy ra ngay sau mốc thời gian hiện tại của truyện. Chỉ "
        "dùng before=latest_scene_date để không kéo về sự kiện chưa xảy ra. Nếu "
        "cần biết bối cảnh nhân vật, gọi thêm entity_profile.\n"
        "3. Đề xuất 2-3 hướng đi cụ thể cho chương tiếp theo. Mỗi hướng phải: "
        "khớp với nhịp tiếp theo của arc trong dàn ý; tôn trọng các divergences "
        "đã khai báo (không được 'sửa' lại lịch sử mà tác giả đã cố ý đổi); nêu "
        "rõ cảnh mở, xung đột, và cảnh kết; kèm scene_date đề xuất.\n"
        "\n"
        "Quy tắc bắt buộc: mọi chi tiết lịch sử phải trích dẫn id bản ghi mà "
        "search_history hoặc get_story_state trả về (ví dụ [id: ...]). Tuyệt đối "
        "không bịa sự kiện, nhân vật lịch sử hay ngày tháng mà công cụ không trả "
        "về; nếu corpus không có, hãy nói rõ là không có. Trả lời bằng tiếng Việt."
    ) % {"novel": novel}


@mcp.prompt()
def review_chapter(novel: str, filename: str) -> str:
    """
    Review one chapter of a Vietnamese novel: deterministic history and
    continuity checks first (with record ids), then prose, pacing and arc fit.
    """
    return (
        "Bạn là biên tập viên duyệt chương '%(filename)s' của tiểu thuyết "
        "'%(novel)s' (nội dung tiếng Việt). Làm đúng theo thứ tự sau, và báo "
        "cáo các phát hiện có tính xác định (bước 2-3) TRƯỚC phần nhận xét văn "
        "phong:\n"
        "\n"
        "1. Gọi read_chapter(novel=\"%(novel)s\", filename=\"%(filename)s\", "
        "part=1) rồi tiếp tục với part=2..parts cho đến khi đọc hết. Ghi lại "
        "meta.scene_date của chương.\n"
        "2. Gọi check_history(novel=\"%(novel)s\", text=<toàn bộ nội dung "
        "chương>, scene_date=<meta.scene_date>) . Liệt kê từng cảnh báo lỗi thời "
        "(anachronism) hoặc mâu thuẫn với lịch sử đã khóa, kèm id bản ghi mà "
        "công cụ trả về.\n"
        "3. Với mỗi nhân vật có tên trong chương, gọi entity_profile("
        "novel=\"%(novel)s\", name=<tên>) và đối chiếu: hành động, quan hệ, nơi "
        "chốn trong chương có khớp với những gì trí nhớ của truyện đã ghi không. "
        "Nêu rõ mâu thuẫn nếu có. Nếu cần, gọi search_history để xác minh một "
        "chi tiết cụ thể.\n"
        "4. Chỉ sau đó mới nhận xét về văn xuôi, nhịp truyện, và mức độ khớp với "
        "arc trong dàn ý (get_story_state nếu cần xem outline).\n"
        "\n"
        "Định dạng: (a) Phát hiện xác định — từng mục kèm [id: ...] của bản ghi "
        "lịch sử; (b) Mâu thuẫn nhân vật; (c) Văn phong và nhịp; (d) Đề xuất sửa "
        "cụ thể. Không bịa chi tiết lịch sử mà công cụ không trả về. Trả lời "
        "bằng tiếng Việt."
    ) % {"novel": novel, "filename": filename}


# ---------------------------------------------------------------------------
# Resources — the same chapter/world reads, addressable by URI
# ---------------------------------------------------------------------------

@mcp.resource("novel://{slug}/chapters/{filename}", mime_type="text/markdown",
              description="Body of one chapter (Vietnamese markdown, front matter stripped).")
def chapter_resource(slug: str, filename: str) -> str:
    data = _get("/api/chapters/%s" % quote(filename, safe=""), params={"novel": slug})
    if _failed(data):
        # Resources have no error-dict convention; a raised error is what the
        # protocol turns into a ResourceError the host can show.
        raise ValueError(data["error"])
    return data.get("content") or ""


@mcp.resource("novel://{slug}/world/{filename}", mime_type="text/markdown",
              description="One world-bible file (Vietnamese markdown).")
def world_resource(slug: str, filename: str) -> str:
    data = _get("/api/context/%s" % quote(filename, safe=""), params={"novel": slug})
    if _failed(data):
        raise ValueError(data["error"])
    return data.get("content") or ""


if __name__ == "__main__":
    mcp.run()  # stdio
