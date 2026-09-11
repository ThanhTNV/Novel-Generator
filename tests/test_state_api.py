# -*- coding: utf-8 -*-
"""
The story-state snapshot and the writer-context endpoint.

``GET /api/novels/{slug}/state`` is the first call an agent makes, so it has
to answer for any novel: no chapters, no outline, a corpus with an error in
it. And ``POST /api/memory/context`` claims to return exactly what the writer
is handed, which is only true while it calls the same composition the writer
does — so that equality is asserted, not assumed.
"""

import asyncio
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient  # noqa: E402

from backend import history, novels  # noqa: E402
from backend.agent import NovelAgent, compose_retrieval_query  # noqa: E402
import backend.rag_pipeline as rp  # noqa: E402
from backend.config import settings  # noqa: E402
from backend.server import app  # noqa: E402

SCIFI = u"""# Nhân vật

## Văn Tâm (Nhân vật chính)

- **Năng lực**: Sở hữu Đồ Lục, cuốn sổ đen ghi 108 Vật Chất.
- **Ngoại hình**: Cao 1m78, tóc đen rối nhẹ.

## Nguyên Khang

- **Vai trò**: Bạn thân của Văn Tâm, làm việc tại phòng thí nghiệm.
"""

# A novel-level override of a shipped project record, declaring a divergence.
DIVERGENCE = textwrap.dedent(u"""
records:
  - id: quang-trung-death
    claim: "Quang Trung băng hà tại Phú Xuân"
    date: 1792-09-16
    entities: ["Quang Trung", "Phú Xuân"]
    confidence: attested
    sources: ["Đại Nam thực lục"]
    diverges: true
    divergence_note: "Trong truyện, hoàng đế qua khỏi cơn bệnh."
""")

UNCITED = textwrap.dedent(u"""
records:
  - id: bad-record
    claim: "Một điều không có nguồn"
""")

# Unquoted, so PyYAML builds it with datetime.date — and fails to.
IMPOSSIBLE_DATE = textwrap.dedent(u"""
records:
  - id: x
    claim: "c"
    date: 1789-02-30
    sources: ["s"]
""")


@pytest.fixture()
def workspace(tmp_path, monkeypatch):
    """Point the novels registry at a scratch directory, engines included."""
    monkeypatch.setattr(settings, "novels_dir", str(tmp_path / "novels"))
    monkeypatch.setattr(settings, "default_novel", "default")
    # A fresh cache per test: engines and history indexes are keyed by slug,
    # and slugs repeat across tests.
    monkeypatch.setattr(rp, "_engines", {})
    monkeypatch.setattr(history, "_indexes", {})
    monkeypatch.setenv("EMBEDDING_PROVIDER", "hash")
    monkeypatch.setattr(settings, "embedding_provider", "hash")
    monkeypatch.setattr(settings, "zero_mem_extractor", "local")
    client = TestClient(app)
    yield client
    for engine in list(rp._engines.values()):
        try:
            engine.store.close()
        except Exception:
            pass


def _make(client, title):
    r = client.post("/api/novels", json={"title": title})
    assert r.status_code == 200, r.text
    return r.json()["slug"]


def _save(client, slug, number, title, content, **extra):
    payload = {"novel": slug, "chapter_number": number, "title": title, "content": content}
    payload.update(extra)
    r = client.post("/api/chapters/save", json=payload)
    assert r.status_code == 200, r.text
    return r.json()


def _state(client, slug):
    r = client.get("/api/novels/%s/state" % slug)
    assert r.status_code == 200, r.text
    return r.json()


def _prose(words):
    """``words`` distinct words in paragraphs of 50, so the tail is checkable."""
    tokens = ["từ%d" % i for i in range(words)]
    return "\n\n".join(" ".join(tokens[i:i + 50]) for i in range(0, words, 50))


# ---------------------------------------------------------------------------
# GET /api/novels/{slug}/state
# ---------------------------------------------------------------------------

def test_an_empty_novel_has_the_full_shape(workspace):
    """Zero chapters, no outline, no context: every key present, nothing raised."""
    client = workspace
    slug = _make(client, "Book A")
    state = _state(client, slug)

    assert set(state) >= {"novel", "chapters", "latest", "latest_scene_date", "outline",
                          "context_files", "divergences", "history_records", "memory",
                          "rules", "skills"}
    assert "history_error" not in state
    assert state["novel"]["slug"] == slug
    assert state["chapters"] == []
    assert state["latest"] is None
    assert state["latest_scene_date"] is None
    assert state["outline"] == ""
    assert state["context_files"] == []
    assert state["divergences"] == []
    shipped = history.load_index(("project", settings.history_dir))
    assert state["history_records"] == len(shipped) > 0
    assert set(state["memory"]) == {"segments", "sources", "entities", "relations"}
    assert state["rules"] and state["skills"]
    for entry in state["rules"] + state["skills"]:
        assert set(entry) == {"name", "scope"}, "state lists rules; it does not inline them"


def test_latest_is_the_highest_number_with_the_last_600_words(workspace):
    """
    The tail is the continuation point. It must come from the highest-numbered
    chapter (not the last saved), and be the last 600 words of its body with
    the paragraphs the author wrote.
    """
    client = workspace
    slug = _make(client, "Book A")
    long_prose = _prose(900)
    ch2 = _save(client, slug, 2, u"Dài", long_prose, scene_date="1790", arc=1,
                summary=u"Nhiều chuyện.")
    _save(client, slug, 1, u"Ngắn", u"Mở đầu rất ngắn.")

    state = _state(client, slug)
    assert [c["number"] for c in state["chapters"]] == [1, 2]
    latest = state["latest"]
    assert set(latest) == {"filename", "number", "title", "scene_date", "arc", "summary", "tail"}
    assert latest["filename"] == ch2["filename"]
    assert latest["number"] == 2 and latest["title"] == u"Dài"
    assert latest["scene_date"] == "1790" and latest["arc"] == 1
    assert latest["summary"] == u"Nhiều chuyện."

    body = client.get("/api/chapters/%s?novel=%s" % (ch2["filename"], slug)).json()["content"]
    assert latest["tail"].split() == body.split()[-600:]
    assert len(latest["tail"].split()) == 600
    assert "\n\n" in latest["tail"]


def test_a_short_latest_chapter_is_returned_whole(workspace):
    client = workspace
    slug = _make(client, "Book A")
    _save(client, slug, 1, "Mo dau", u"Chỉ vài từ thôi.")
    tail = _state(client, slug)["latest"]["tail"]
    assert tail == u"# Chapter 1: Mo dau\n\nChỉ vài từ thôi."


def test_latest_scene_date_follows_chapter_order_not_the_calendar(workspace):
    """
    The date the story has reached is the last *dated chapter by number*. A
    flashback chapter carries an earlier date and must not be skipped for a
    later one; an undated final chapter must not reset the clock to null.
    """
    client = workspace
    slug = _make(client, "Book A")
    _save(client, slug, 1, "a", "x", scene_date="1791")
    _save(client, slug, 2, "b", "x", scene_date="1789-02")
    _save(client, slug, 3, "c", "x")
    assert _state(client, slug)["latest_scene_date"] == "1789-02"


def test_outline_is_capped_with_a_marker(workspace):
    client = workspace
    slug = _make(client, "Book A")
    context_dir = novels.get(slug).context_dir

    short = u"# Cốt truyện\n\nBa hồi."
    (context_dir / "main-story.md").write_text(short, encoding="utf-8")
    assert _state(client, slug)["outline"] == short

    long = (u"dòng " * 20 + "\n") * 80          # well past 6000 chars
    assert len(long) > 6000
    (context_dir / "main-story.md").write_text(long, encoding="utf-8")
    outline = _state(client, slug)["outline"]
    assert outline.endswith(u"\n…[truncated]")
    assert outline == long[:6000] + u"\n…[truncated]"

    files = _state(client, slug)["context_files"]
    assert files == [{"filename": "main-story.md", "size": len(long)}]


def test_divergences_come_from_the_novels_own_history(workspace):
    """
    A novel overrides a shipped record by id to declare where it departs.
    The state must surface that override — and only records that diverge.
    """
    client = workspace
    slug = _make(client, "Book A")
    hist = novels.get(slug).root / "history"
    hist.mkdir()
    (hist / "alt.yaml").write_text(DIVERGENCE, encoding="utf-8")

    state = _state(client, slug)
    assert "history_error" not in state
    assert state["divergences"] == [{
        "id": "quang-trung-death",
        "claim": u"Quang Trung băng hà tại Phú Xuân",
        "date": "1792-09-16",
        "divergence_note": u"Trong truyện, hoàng đế qua khỏi cơn bệnh.",
    }]
    shipped = history.load_index(("project", settings.history_dir))
    assert state["history_records"] == len(shipped), "an override must not add a record"


def test_a_malformed_corpus_is_reported_not_raised(workspace):
    """
    An uncited record is refused at load, and /api/history answers 422. The
    snapshot instead names the problem and still delivers everything else,
    because an agent that cannot get the state cannot tell the author either.
    """
    client = workspace
    slug = _make(client, "Book A")
    _save(client, slug, 1, "Mo dau", u"Vẫn có chương.")
    hist = novels.get(slug).root / "history"
    hist.mkdir()
    (hist / "bad.yaml").write_text(UNCITED, encoding="utf-8")

    state = _state(client, slug)
    assert "no source" in state["history_error"]
    assert "bad.yaml" in state["history_error"]
    assert state["divergences"] == []
    assert state["history_records"] == 0
    assert state["latest"]["number"] == 1, "the rest of the snapshot still arrives"


def test_an_unquoted_impossible_date_in_the_corpus_is_reported_not_raised(workspace):
    """
    ``date: 1789-02-30`` unquoted is built by PyYAML with datetime.date,
    whose ValueError is no YAMLError and used to pass every HistoryError
    guard: the snapshot 500ed, and /api/history answered 500 instead of the
    documented 422. (Quoted, the same date was already a proper HistoryError.)
    """
    client = workspace
    slug = _make(client, "Book A")
    _save(client, slug, 1, "Mo dau", u"Vẫn có chương.")
    hist = novels.get(slug).root / "history"
    hist.mkdir()
    (hist / "alt.yaml").write_text(IMPOSSIBLE_DATE, encoding="utf-8")

    state = _state(client, slug)
    assert "alt.yaml" in state["history_error"]
    assert state["divergences"] == [] and state["history_records"] == 0
    assert state["latest"]["number"] == 1, "the rest of the snapshot still arrives"
    assert client.get("/api/history?novel=%s" % slug).status_code == 422


def test_unknown_slug_is_404(workspace):
    assert workspace.get("/api/novels/nope/state").status_code == 404


@pytest.mark.parametrize("bad", ["..%5Cetc", "..%5C..%5Cx", "A%20B"])
def test_traversal_slug_is_400(workspace, bad):
    """The same slug rules as every other route: a decoded backslash is a 400."""
    assert workspace.get("/api/novels/%s/state" % bad).status_code == 400


# ---------------------------------------------------------------------------
# POST /api/memory/context
# ---------------------------------------------------------------------------

def test_the_retrieval_query_composition_is_fixed():
    """Changing this string changes what every writer is handed."""
    assert compose_retrieval_query("q", ["a", "b"], ["c"]) == "q \na b \nc"
    assert compose_retrieval_query("q", [], None) == "q"


def test_memory_context_matches_what_the_writer_is_handed(workspace):
    """
    The endpoint exists so an agent can see the writer's context. If it drifted
    from ``gather_context`` by so much as a space in the query, the agent would
    be reasoning about evidence the writer never saw.
    """
    client = workspace
    slug = _make(client, "Book A")
    r = client.put("/api/context/characters.md", json={"novel": slug, "content": SCIFI})
    assert r.status_code == 200, r.text

    query, characters, locations = u"Văn Tâm gặp bạn", [u"Nguyên Khang"], [u"phòng thí nghiệm"]
    r = client.post("/api/memory/context", json={
        "novel": slug, "query": query, "characters": characters, "locations": locations,
    })
    assert r.status_code == 200, r.text
    got = r.json()
    assert set(got) >= {"context", "used", "route", "profile", "tokens", "novel"}
    assert got["novel"] == slug
    assert got["context"], "seeded context must be retrievable"

    agent = NovelAgent(novel=novels.get(slug))
    expected = asyncio.run(agent.gather_context(query, characters, locations))
    assert got["context"] == expected


def test_memory_context_scopes_to_the_novel(workspace):
    client = workspace
    a = _make(client, "Book A")
    b = _make(client, "Book B")
    assert client.put("/api/context/characters.md",
                      json={"novel": a, "content": SCIFI}).status_code == 200
    got = client.post("/api/memory/context", json={"novel": b, "query": u"Văn Tâm"}).json()
    assert got["context"] == "" and got["used"] == []
    assert client.post("/api/memory/context",
                       json={"novel": "nope", "query": "x"}).status_code == 404
