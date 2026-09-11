# -*- coding: utf-8 -*-
"""
Unit tests for the MCP server with the app replaced by an httpx.MockTransport.

Run with the 3.12 venv, never the app's 3.9:

    mcp_server/.venv/Scripts/python.exe -m pytest mcp_server/tests -q

Nothing here starts the app or touches novels/. Every request the server
would send is recorded, so the suite can prove the wire shape of each tool
and — the property that matters most — that no tool can ever write.
"""

import asyncio
import json
import sys
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import server  # noqa: E402

NOVEL = "tay-son"
FORBIDDEN_PATHS = (
    "/api/generate", "/api/revise", "/api/chat", "/api/chapters/save",
    "/api/ingest/", "/api/vectordb/clear", "/api/history/reload",
)


class Recorder:
    """
    MockTransport handler: logs (method, path, params, json) for every
    request and answers from a (method, path) -> payload table, defaulting
    to an empty 200 object so a tool under test never trips on the reply.
    """

    def __init__(self, responses=None):
        self.log = []
        self.responses = responses or {}

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content) if request.content else None
        self.log.append((request.method, request.url.path,
                         dict(request.url.params), body))
        reply = self.responses.get((request.method, request.url.path), {"ok": True})
        if isinstance(reply, httpx.Response):
            return reply
        return httpx.Response(200, json=reply)

    @property
    def last(self):
        return self.log[-1]


@pytest.fixture
def recorder():
    rec = Recorder()
    server.set_transport(httpx.MockTransport(rec))
    yield rec
    server.set_transport(None)


def _install(handler):
    server.set_transport(httpx.MockTransport(handler))


def _tool_names():
    return {t.name for t in asyncio.run(server.mcp.list_tools())}


def _paragraphs(count, words_each, tag="p"):
    return ["%s%d " % (tag, i) * words_each for i in range(count)]


# ---------------------------------------------------------------------------
# Wire shape — each tool hits the right method + path + params/json
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("call, method, path, params, body", [
    (lambda: server.list_novels(),
     "GET", "/api/novels", {}, None),
    (lambda: server.get_story_state(NOVEL),
     "GET", "/api/novels/%s/state" % NOVEL, {}, None),
    (lambda: server.list_chapters(NOVEL),
     "GET", "/api/chapters", {"novel": NOVEL}, None),
    (lambda: server.read_chapter(NOVEL, "chapter-001-mo-dau.md"),
     "GET", "/api/chapters/chapter-001-mo-dau.md", {"novel": NOVEL}, None),
    (lambda: server.read_world(NOVEL),
     "GET", "/api/context", {"novel": NOVEL}, None),
    (lambda: server.read_world(NOVEL, "characters.md"),
     "GET", "/api/context/characters.md", {"novel": NOVEL}, None),
    (lambda: server.search_memory(NOVEL, "Nguyễn Huệ", top_k=3),
     "POST", "/api/search", {}, {"novel": NOVEL, "query": "Nguyễn Huệ", "top_k": 3}),
    (lambda: server.get_writer_context(NOVEL, "trận Ngọc Hồi", ["Nguyễn Huệ"], ["Thăng Long"]),
     "POST", "/api/memory/context", {},
     {"novel": NOVEL, "query": "trận Ngọc Hồi", "characters": ["Nguyễn Huệ"],
      "locations": ["Thăng Long"]}),
    (lambda: server.search_history(NOVEL, "Đống Đa", before="1789-01", limit=4),
     "POST", "/api/history/search", {},
     {"novel": NOVEL, "query": "Đống Đa", "before": "1789-01", "limit": 4}),
    (lambda: server.check_history(NOVEL, "Quân Thanh tràn vào.", scene_date="1788-12"),
     "POST", "/api/history/check", {},
     {"novel": NOVEL, "text": "Quân Thanh tràn vào.", "scene_date": "1788-12"}),
])
def test_tool_sends_the_contracted_request(recorder, call, method, path, params, body):
    """
    Each tool is a thin proxy, so the whole contract is the request it sends:
    a wrong path 404s, a query param sent as JSON scopes to the default novel,
    a missing ``novel`` in a POST body reads another novel's memory.
    """
    call()
    assert recorder.log == [(method, path, params, body)]


def test_search_history_and_check_history_send_explicit_nulls(recorder):
    """
    ``before`` and ``scene_date`` default to None and must still be present
    in the body: the app's pydantic models accept null, and an agent reading
    the recorded request should see that no date filter was applied.
    """
    server.search_history(NOVEL, "vua Quang Trung")
    assert recorder.last[3] == {"novel": NOVEL, "query": "vua Quang Trung",
                                "before": None, "limit": 6}
    server.check_history(NOVEL, "đoạn văn")
    assert recorder.last[3] == {"novel": NOVEL, "text": "đoạn văn", "scene_date": None}


def test_writer_context_defaults_lists_to_empty_not_null(recorder):
    """
    The endpoint declares ``characters: List[str] = []``; sending null for
    the omitted lists would be a 422, and the tool would look broken for the
    most common call (query only).
    """
    server.get_writer_context(NOVEL, "câu hỏi")
    assert recorder.last[3] == {"novel": NOVEL, "query": "câu hỏi",
                                "characters": [], "locations": []}


def test_entity_profile_url_encodes_the_vietnamese_name(recorder):
    """
    Names carry diacritics and spaces ("Nguyễn Huệ"); the path must be
    percent-encoded so the app decodes the same string the agent typed, and
    a ``/`` in a name must not become a path segment.
    """
    server.entity_profile(NOVEL, "Nguyễn Huệ/Quang Trung")
    method, path, params, _ = recorder.last
    assert method == "GET"
    assert path == "/api/memory/entity/Nguyễn Huệ/Quang Trung"
    assert params == {"novel": NOVEL}


def test_entity_profile_raw_path_is_a_single_segment():
    """
    Companion to the test above: ``URL.path`` decodes, so it cannot show that
    the slash was escaped. Check the bytes that actually go on the wire.
    """
    seen = {}

    def handler(request):
        seen["raw"] = request.url.raw_path.decode()
        return httpx.Response(200, json={})

    _install(handler)
    try:
        server.entity_profile(NOVEL, "Nguyễn Huệ/Quang Trung")
    finally:
        server.set_transport(None)
    path = seen["raw"].split("?")[0]
    assert path.startswith("/api/memory/entity/")
    assert "/" not in path[len("/api/memory/entity/"):]
    assert "%20" in path and "%2F" in path


def test_get_rules_and_skills_merges_two_gets(recorder):
    """
    Two endpoints, one tool: an agent should not have to know that rules and
    skills live apart, and the merged shape is what the README documents.
    """
    recorder.responses = {
        ("GET", "/api/rules"): {"rules": [{"name": "style", "scope": "project", "content": "x"}],
                                "novel": NOVEL},
        ("GET", "/api/skills"): {"skills": [{"name": "dialogue", "scope": "novel", "content": "y"}],
                                 "novel": NOVEL},
    }
    out = server.get_rules_and_skills(NOVEL)
    assert [(m, p, q) for m, p, q, _ in recorder.log] == [
        ("GET", "/api/rules", {"novel": NOVEL}),
        ("GET", "/api/skills", {"novel": NOVEL}),
    ]
    assert out["rules"][0]["name"] == "style"
    assert out["skills"][0]["name"] == "dialogue"
    assert out["novel"] == NOVEL


def test_get_rules_and_skills_surfaces_the_first_failure(recorder):
    """A 404 on rules must not be hidden behind a half-merged result."""
    recorder.responses = {("GET", "/api/rules"): httpx.Response(404, json={"detail": "Novel not found"})}
    assert server.get_rules_and_skills("nope") == {"error": "Novel not found", "status": 404}
    assert len(recorder.log) == 1, "skills must not be fetched once rules failed"


# ---------------------------------------------------------------------------
# read_chapter — part splitting
# ---------------------------------------------------------------------------

def _chapter_transport(body, meta=None, filename="chapter-003-loi-moi.md"):
    payload = {"filename": filename, "content": body}
    if meta is not None:
        payload["meta"] = meta
    return Recorder({("GET", "/api/chapters/%s" % filename): payload})


def test_read_chapter_splits_4000_words_into_3_parts_on_paragraph_boundaries():
    """
    20 paragraphs of 200 words = 4000 words: 7 + 7 + 6 paragraphs, three
    parts, none over 1500 words, every paragraph intact and in order. A
    split mid-paragraph would hand the reader half a scene and make the
    reviewer prompt flag a "dangling" sentence that is whole in the file.
    """
    paragraphs = [p.strip() for p in _paragraphs(20, 200)]
    meta = {"chapter": 3, "title": "Lời mời từ bóng tối", "scene_date": "1789-02"}
    _install(_chapter_transport("\n\n".join(paragraphs), meta))
    try:
        parts = [server.read_chapter(NOVEL, "chapter-003-loi-moi.md", part=i) for i in (1, 2, 3)]
    finally:
        server.set_transport(None)

    assert [p["parts"] for p in parts] == [3, 3, 3]
    assert [p["part"] for p in parts] == [1, 2, 3]
    assert [p["words_in_part"] for p in parts] == [1400, 1400, 1200]
    assert all(p["words_in_part"] <= 1500 for p in parts)
    rejoined = [para for p in parts for para in p["text"].split("\n\n")]
    assert rejoined == paragraphs
    assert parts[0]["meta"] == meta
    assert parts[0]["filename"] == "chapter-003-loi-moi.md"


def test_read_chapter_part_out_of_range_is_an_error_dict():
    """
    An agent iterating part=1..N will overshoot by one when it loses count;
    it must get a readable error, not an exception, and not part 1 again.
    """
    _install(_chapter_transport("\n\n".join(_paragraphs(20, 200))))
    try:
        too_far = server.read_chapter(NOVEL, "chapter-003-loi-moi.md", part=4)
        zero = server.read_chapter(NOVEL, "chapter-003-loi-moi.md", part=0)
    finally:
        server.set_transport(None)
    assert set(too_far) == {"error"} and "3 part(s)" in too_far["error"]
    assert set(zero) == {"error"}


def test_read_chapter_legacy_file_has_empty_meta():
    """
    Chapters saved before front matter existed come back without ``meta``;
    the tool must report ``{}`` so the review prompt's "use meta.scene_date"
    step degrades to "no date" instead of a KeyError.
    """
    _install(_chapter_transport("Một đoạn.\n\nHai đoạn."))
    try:
        out = server.read_chapter(NOVEL, "chapter-003-loi-moi.md")
    finally:
        server.set_transport(None)
    assert out["meta"] == {}
    assert out["parts"] == 1
    assert out["text"] == "Một đoạn.\n\nHai đoạn."


def test_split_parts_cuts_an_oversized_paragraph_at_the_cap():
    """
    A 1600-word paragraph cannot fit any part. Handing it back whole (as this
    once did) broke the "at most 1500 words" promise the tool-result budget
    rests on, so the cap wins: the paragraph is cut on word boundaries, no
    part exceeds the cap, and not a word is lost or reordered.
    """
    small_a, big, small_b = "a " * 100, "b " * 1600, "c " * 100
    whole = "\n\n".join([small_a, big, small_b])
    parts = server.split_parts(whole)
    assert [len(p.split()) for p in parts] == [100, 1500, 200]
    assert " ".join(parts).split() == whole.split()


def test_split_parts_cuts_a_single_newline_chapter_on_its_own_line_breaks():
    """
    The manual Write editor saves Enter as a single "\\n", so a 3500-word
    chapter typed that way is one "paragraph" to the blank-line split and
    read_chapter answered parts=1, words_in_part=3500. It must be cut on the
    line breaks the author did write — never mid-line while a line fits —
    and keep them as "\\n", not turn them into blank lines.
    """
    lines = [("l%d " % i * 500).strip() for i in range(7)]
    body = "# Chapter 1: T\n\n" + "\n".join(lines)
    parts = server.split_parts(body)
    assert [len(p.split()) for p in parts] == [1004, 1500, 1000]
    for part in parts:
        for line in part.split("\n"):
            assert line in lines or line in ("", "# Chapter 1: T"), line[:40]
    assert "\n".join(parts).split() == body.split()

    _install(_chapter_transport(body))
    try:
        out = server.read_chapter(NOVEL, "chapter-003-loi-moi.md")
    finally:
        server.set_transport(None)
    assert out["parts"] == 3 and out["words_in_part"] <= 1500


def test_split_parts_treats_windows_and_padded_blank_lines_as_breaks():
    """Files saved on Windows carry ``\\r\\n``; a blank line with spaces is still blank."""
    parts = server.split_parts("x " * 10 + "\r\n  \r\n" + "y " * 10, max_words=12)
    assert len(parts) == 2


def test_split_parts_of_an_empty_body_is_one_empty_part():
    """``part=1`` must always be answerable, even for a chapter with no body yet."""
    assert server.split_parts("") == [""]
    assert server.split_parts("\n\n  \n") == [""]


# ---------------------------------------------------------------------------
# Error mapping
# ---------------------------------------------------------------------------

def test_unreachable_app_returns_the_exact_not_running_message(monkeypatch):
    """
    The one error every new user hits: they registered the server but did
    not start the app. The message must name the URL the server actually
    tried, and the fix, so they do not go looking for a bug in the host.
    """
    monkeypatch.setenv("NOVELGEN_URL", "http://127.0.0.1:9")

    def refuse(request):
        raise httpx.ConnectError("connection refused", request=request)

    _install(refuse)
    try:
        assert server.list_novels() == {
            "error": "Novel Generator app is not running at http://127.0.0.1:9. "
                     "Start it with: python main.py"}
    finally:
        server.set_transport(None)


def test_timeout_maps_to_the_same_not_running_message(monkeypatch):
    """A first embedding run can stall the app; that reads as 'not running', not a traceback."""
    monkeypatch.setenv("NOVELGEN_URL", "http://127.0.0.1:9")

    def stall(request):
        raise httpx.ReadTimeout("timed out", request=request)

    _install(stall)
    try:
        out = server.get_story_state(NOVEL)
    finally:
        server.set_transport(None)
    assert out["error"].startswith("Novel Generator app is not running at http://127.0.0.1:9")


def test_404_with_json_detail_becomes_error_and_status(recorder):
    """The app's ``{"detail": ...}`` is the message the author wrote; keep it verbatim."""
    recorder.responses = {("GET", "/api/chapters/nope.md"): httpx.Response(404, json={"detail": "Chapter not found"})}
    assert server.read_chapter(NOVEL, "nope.md") == {"error": "Chapter not found", "status": 404}


def test_error_without_json_body_falls_back_to_status_text(recorder):
    """A proxy or crash page in front of the app is HTML; the status text still tells the story."""
    recorder.responses = {("GET", "/api/novels"): httpx.Response(502, text="<html>bad gateway</html>")}
    assert server.list_novels() == {"error": "Bad Gateway", "status": 502}


def test_422_validation_list_is_flattened_to_a_readable_message(recorder):
    """FastAPI's list-of-dicts detail is unreadable as JSON in a tool result."""
    recorder.responses = {("POST", "/api/search"): httpx.Response(422, json={"detail": [
        {"loc": ["body", "query"], "msg": "field required", "type": "value_error.missing"}]})}
    out = server.search_memory(NOVEL, "")
    assert out == {"error": "body.query: field required", "status": 422}


def test_non_object_200_body_is_an_error_not_story_data(recorder):
    """A captive portal returning 200 with a list or a string must not be returned as a novel."""
    recorder.responses = {("GET", "/api/novels"): httpx.Response(200, json=["not", "an", "object"])}
    out = server.list_novels()
    assert "error" in out and out["status"] == 200


def test_unexpected_exception_inside_a_tool_becomes_an_error_dict(recorder, monkeypatch):
    """
    'Never let an exception escape a tool': a bug in the splitter must reach
    the agent as a message it can report, not as a protocol-level failure.
    """
    def boom(body, max_words=1500):
        raise RuntimeError("splitter bug")

    monkeypatch.setattr(server, "split_parts", boom)
    recorder.responses = {("GET", "/api/chapters/x.md"): {"filename": "x.md", "content": "a"}}
    assert server.read_chapter(NOVEL, "x.md") == {"error": "RuntimeError: splitter bug"}


# ---------------------------------------------------------------------------
# Read-only guarantee
# ---------------------------------------------------------------------------

ALL_TOOL_CALLS = {
    "list_novels": lambda: server.list_novels(),
    "get_story_state": lambda: server.get_story_state(NOVEL),
    "list_chapters": lambda: server.list_chapters(NOVEL),
    "read_chapter": lambda: server.read_chapter(NOVEL, "chapter-001-a.md"),
    "read_world": lambda: (server.read_world(NOVEL), server.read_world(NOVEL, "world.md")),
    "search_memory": lambda: server.search_memory(NOVEL, "q"),
    "entity_profile": lambda: server.entity_profile(NOVEL, "Nguyễn Huệ"),
    "get_writer_context": lambda: server.get_writer_context(NOVEL, "q", ["a"], ["b"]),
    "search_history": lambda: server.search_history(NOVEL, "q", "1789", 3),
    "check_history": lambda: server.check_history(NOVEL, "t", "1789"),
    "get_rules_and_skills": lambda: server.get_rules_and_skills(NOVEL),
}


def test_every_registered_tool_is_exercised_by_the_read_only_check():
    """
    Adding a tool without adding it here would let it escape the read-only
    proof below. The registry is the source of truth, not this file.
    """
    assert set(ALL_TOOL_CALLS) == _tool_names()


def test_no_tool_issues_a_write_or_touches_a_forbidden_path(recorder):
    """
    The design promise in the README: an MCP host can never generate, save,
    ingest, clear or reload through this server. Proven over the recorded
    requests of every tool, not by reading the source.
    """
    for call in ALL_TOOL_CALLS.values():
        call()
    assert recorder.log, "the tools must have made requests"
    for method, path, _, _ in recorder.log:
        assert method in ("GET", "POST"), (method, path)
        for forbidden in FORBIDDEN_PATHS:
            assert not path.startswith(forbidden), (method, path)


def test_every_tool_is_annotated_read_only():
    """Hosts skip confirmation prompts on readOnlyHint; the hint must be on every tool."""
    for tool in asyncio.run(server.mcp.list_tools()):
        assert tool.annotations is not None and tool.annotations.readOnlyHint is True, tool.name


# ---------------------------------------------------------------------------
# FastMCP registration — the guard wrapper must not hide the real signature
# ---------------------------------------------------------------------------

def test_tool_schemas_come_from_the_real_signatures():
    """
    ``_guarded`` wraps every tool; if FastMCP saw the wrapper's ``*args``
    instead of the real parameters, every tool would advertise no arguments
    and hosts would call them with nothing.
    """
    by_name = {t.name: t for t in asyncio.run(server.mcp.list_tools())}
    read_chapter = by_name["read_chapter"].inputSchema
    assert set(read_chapter["properties"]) == {"novel", "filename", "part"}
    assert read_chapter["required"] == ["novel", "filename"]
    assert read_chapter["properties"]["part"]["default"] == 1
    assert set(by_name["search_history"].inputSchema["properties"]) == {"novel", "query", "before", "limit"}


def test_call_tool_through_fastmcp_returns_the_error_dict_as_structured_content(recorder):
    """
    The error dict must survive the SDK's output validation: the tools are
    declared ``dict[str, Any]`` so ``{"error", "status"}`` is valid output
    rather than a schema violation the host reports as a broken tool.
    """
    recorder.responses = {("GET", "/api/novels/nope/state"): httpx.Response(404, json={"detail": "Novel not found"})}
    _, structured = asyncio.run(server.mcp.call_tool("get_story_state", {"novel": "nope"}))
    assert structured == {"error": "Novel not found", "status": 404}


def test_chapter_resource_reads_the_body_by_uri(recorder):
    """The resource template is the URI form of read_chapter; it must resolve slug and filename."""
    recorder.responses = {("GET", "/api/chapters/chapter-001-a.md"): {
        "filename": "chapter-001-a.md", "content": "Thân chương.", "meta": {}}}
    contents = list(asyncio.run(server.mcp.read_resource("novel://%s/chapters/chapter-001-a.md" % NOVEL)))
    assert contents[0].content == "Thân chương."
    assert contents[0].mime_type == "text/markdown"
    assert recorder.last[:3] == ("GET", "/api/chapters/chapter-001-a.md", {"novel": NOVEL})


def test_resource_failure_raises_instead_of_returning_an_error_string(recorder):
    """A resource has no error-dict channel; a 404 body served as content would read as a chapter."""
    recorder.responses = {("GET", "/api/context/nope.md"): httpx.Response(404, json={"detail": "Context file not found."})}
    with pytest.raises(Exception, match="Context file not found"):
        list(asyncio.run(server.mcp.read_resource("novel://%s/world/nope.md" % NOVEL)))


# ---------------------------------------------------------------------------
# Prompts and docstrings
# ---------------------------------------------------------------------------

def test_suggest_next_chapter_prompt_orchestrates_the_right_tools():
    """
    The prompt is the workflow: state first, history filtered to before the
    latest scene date, cite ids. A prompt that forgot ``before`` would let
    the agent propose events that have not happened yet.
    """
    text = server.suggest_next_chapter(NOVEL)
    assert text.strip()
    for needle in ("get_story_state", "search_history", "before", "latest_scene_date", NOVEL, "id"):
        assert needle in text, needle


def test_review_chapter_prompt_orchestrates_the_right_tools():
    """Deterministic checks (history, entities) must be demanded before prose notes."""
    text = server.review_chapter(NOVEL, "chapter-003-loi-moi.md")
    assert text.strip()
    for needle in ("read_chapter", "check_history", "entity_profile", "scene_date",
                   "chapter-003-loi-moi.md", NOVEL):
        assert needle in text, needle
    assert text.index("check_history") < text.index("văn xuôi"), "history before prose"


def test_prompts_render_through_fastmcp_as_user_messages():
    """The SDK must accept the plain-string return and expose both arguments."""
    result = asyncio.run(server.mcp.get_prompt("review_chapter",
                                               {"novel": NOVEL, "filename": "chapter-001-a.md"}))
    assert result.messages[0].role == "user"
    assert "read_chapter" in result.messages[0].content.text
    names = {p.name: [a.name for a in (p.arguments or [])] for p in asyncio.run(server.mcp.list_prompts())}
    assert names == {"suggest_next_chapter": ["novel"], "review_chapter": ["novel", "filename"]}


def test_every_tool_and_prompt_docstring_mentions_vietnamese():
    """
    Hosts show the description to the model; it must know the content is
    Vietnamese before it phrases a search query or judges a passage.
    """
    for tool in asyncio.run(server.mcp.list_tools()):
        assert "Vietnamese" in (tool.description or ""), tool.name
    for prompt in asyncio.run(server.mcp.list_prompts()):
        assert "Vietnamese" in (prompt.description or ""), prompt.name


def test_server_is_named_per_contract():
    """Hosts key config on the name; ``claude mcp add novel-generator`` must match it."""
    assert server.mcp.name == "novel-generator"
