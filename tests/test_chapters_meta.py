# -*- coding: utf-8 -*-
"""
Chapter front matter.

The manuscript is a directory of Markdown files that predate the metadata
block, get hand-edited, and get read by agents that never see the UI. So the
rules under test are about tolerance and containment: a file with no block or
a broken block is still a chapter; a saved chapter round-trips; and nothing
from the block leaks into the memory index.
"""

import codecs
import datetime
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient  # noqa: E402

from backend import chapters, history, novels  # noqa: E402
import backend.rag_pipeline as rp  # noqa: E402
from backend.config import settings  # noqa: E402
from backend.server import app  # noqa: E402

LEGACY = u"# Chapter 1: Mở đầu\n\nVăn Tâm mở cuốn sổ đen.\n\nMột giọng nói vang lên."


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


def _write(slug, name, text):
    path = novels.get(slug).chapters_dir / name
    path.write_text(text, encoding="utf-8")
    return path


def _save(client, slug, number, title, content, **extra):
    payload = {"novel": slug, "chapter_number": number, "title": title, "content": content}
    payload.update(extra)
    return client.post("/api/chapters/save", json=payload)


# ---------------------------------------------------------------------------
# The module, on its own
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("filename,expected", [
    ("chapter-007-loi-moi.md", 7),
    ("chapter-12.md", 12),
    ("chapter-003-lời-mời-từ-bóng-tối.md", 3),
    ("notes.md", None),
    ("", None),
])
def test_chapter_number_comes_from_the_filename(filename, expected):
    assert chapters.chapter_number(filename) == expected


def test_render_then_parse_returns_meta_and_body_untouched():
    """
    The pitfalls a round trip has to survive: a full date that YAML would
    otherwise turn into a ``datetime.date``, a year that would become an
    ``int``, a Vietnamese title, a summary with a colon, and a horizontal
    rule inside the body that must not be mistaken for the closing fence.
    """
    meta = {
        "chapter": 3,
        "title": u"Lời mời từ bóng tối",
        "scene_date": "1789-02-15",
        "arc": 1,
        "summary": u"Hai dòng: tóm tắt. Kết thúc.",
        "words": 2143,
        "saved_at": "2026-09-11T10:00:00+00:00",
    }
    body = u"# Chapter 3: Lời mời từ bóng tối\n\nĐoạn một.\n\n---\n\nĐoạn hai."
    text = chapters.render_chapter(meta, body)
    assert text.startswith("---\n")
    assert chapters.parse_front_matter(text) == (meta, body)

    year_only = dict(meta, scene_date="1789")
    parsed, _ = chapters.parse_front_matter(chapters.render_chapter(year_only, body))
    assert parsed["scene_date"] == "1789", "a year must stay a string, not become an int"


def test_a_legacy_file_has_no_front_matter():
    """Every chapter saved before the block existed starts at '# Chapter N'."""
    assert chapters.parse_front_matter(LEGACY) == ({}, LEGACY)
    assert chapters.parse_front_matter("") == ({}, "")


@pytest.mark.parametrize("text", [
    u"---\ntitle: \"unclosed\nchapter: [\n---\n# Chapter 1\n\nprose",   # invalid YAML
    u"---\ntitle: x\n# Chapter 1\n\nprose",                             # never closed
    u"---\njust words\n---\n# Chapter 1\n\nprose",                      # not a mapping
    u"--- not a fence\nprose",
    # Valid YAML syntax whose *values* PyYAML cannot build: an unquoted
    # impossible date is a plain ValueError out of datetime.date, and an
    # explicit !!timestamp that misses the regex an AttributeError. Neither
    # is a YAMLError, and a guard that only caught those let them through.
    u"---\nchapter: 2\nscene_date: 1789-02-30\n---\n# Chapter 2\n\nprose",
    u"---\nchapter: 2\nscene_date: 1789-13-01\n---\n# Chapter 2\n\nprose",
    u"---\nscene_date: !!timestamp abc\n---\n# Chapter 2\n\nprose",
])
def test_a_broken_block_degrades_to_the_whole_text(text):
    """
    Someone hand-edits a summary and leaves a quote open, or types a date
    that does not exist. The file must still read as a chapter, with nothing
    lost, rather than fail to parse.
    """
    assert chapters.parse_front_matter(text) == ({}, text)


def test_a_bom_before_the_fence_is_not_a_legacy_file():
    """
    An editor saving "UTF-8 with BOM" leaves U+FEFF before the fence, and
    ``read_text("utf-8")`` keeps it. The chapter then listed as legacy: meta
    empty, scene_date silently gone, the whole block leaked into the body
    and the state tail, and the title fell back to the filename stem.
    """
    text = u"﻿---\nchapter: 2\ntitle: Hai\nscene_date: \"1790\"\n---\n# Chapter 2: Hai\n\nprose"
    meta, body = chapters.parse_front_matter(text)
    assert meta == {"chapter": 2, "title": "Hai", "scene_date": "1790"}
    assert body.startswith("# Chapter 2")
    # A legacy file with a BOM must still give up its heading title.
    meta, body = chapters.parse_front_matter(u"﻿" + LEGACY)
    assert meta == {} and chapters.heading_title(body) == u"Mở đầu"


def test_as_int_answers_none_for_what_int_cannot_hold():
    """
    YAML's ``.inf`` is a float that int() refuses with OverflowError — not
    the ValueError ``.nan`` gets — and it escaped as a 500 on the chapter
    list and the state snapshot for the whole novel.
    """
    for bad in (float("inf"), float("-inf"), float("nan"), True, "abc", [1], None):
        assert chapters.as_int(bad) is None, bad
    assert chapters.as_int("7") == 7 and chapters.as_int(2.0) == 2


def test_clean_meta_hands_back_only_what_json_can_carry():
    """
    The read endpoint returns the block as JSON; Starlette refuses NaN and
    FastAPI strict-decodes bytes, so a value PyYAML accepted but JSON cannot
    carry made a readable chapter a 500. Everything must be normalised here
    — and no key invented, because an absent scene_date means "undated".
    """
    meta = {
        "chapter": "3", "arc": float("nan"), "words": float("inf"),
        "title": b"\xff", "summary": None,
        "saved_at": datetime.datetime(2026, 9, 11, 10, 0, tzinfo=datetime.timezone.utc),
        "scene_date": datetime.date(1789, 2, 15),
        "tags": [float("-inf"), b"ok", {"k": float("nan")}, ("a", 1)],
        7: {1, 2},
    }
    out = chapters.clean_meta(meta)
    assert out["chapter"] == 3 and out["arc"] is None and out["words"] is None
    assert out["title"] == u"�" and out["summary"] == ""
    assert out["saved_at"] == "2026-09-11T10:00:00+00:00"
    assert out["scene_date"] == "1789-02-15"
    assert out["tags"] == [None, "ok", {"k": None}, ["a", 1]]
    assert sorted(out["7"]) == [1, 2]
    assert set(out) == {"chapter", "arc", "words", "title", "summary",
                        "saved_at", "scene_date", "tags", "7"}
    assert chapters.clean_meta({}) == {}
    json.dumps(out, allow_nan=False)   # the encoder's own verdict


def test_crlf_front_matter_still_parses():
    """A Windows editor saves CRLF; the fence must be recognised with the \\r."""
    text = u"---\r\nchapter: 2\r\ntitle: Hai\r\n---\r\n# Chapter 2: Hai\r\n\r\nprose"
    meta, body = chapters.parse_front_matter(text)
    assert meta == {"chapter": 2, "title": "Hai"}
    assert body.startswith("# Chapter 2")


def test_heading_title_recovers_the_legacy_title():
    assert chapters.heading_title(LEGACY) == u"Mở đầu"
    assert chapters.heading_title(u"# Chapter 2\n\nprose") == "Chapter 2"
    assert chapters.heading_title(u"# Một tiêu đề khác\n\nprose") == u"Một tiêu đề khác"
    assert chapters.heading_title(u"Just prose.") is None


def test_scene_date_text_normalises_what_yaml_hands_back():
    """
    An unquoted 1789-02-15 in a hand-edited block arrives as a date object;
    the API promises a string, and a hand edit must not change the type.
    """
    assert chapters.scene_date_text(datetime.date(1789, 2, 15)) == "1789-02-15"
    assert chapters.scene_date_text("1789-02") == "1789-02"
    assert chapters.scene_date_text(1789) == "1789"
    assert chapters.scene_date_text(None) is None
    assert chapters.scene_date_text("") is None


def test_tail_words_keeps_paragraph_breaks():
    """
    The tail is what an agent reads to continue the prose. Re-joining words
    with spaces would hand it the chapter's ending as one run-on line.
    """
    paragraphs = [" ".join("w%d" % (p * 100 + i) for i in range(100)) for p in range(7)]
    text = "\n\n".join(paragraphs)
    tail = chapters.tail_words(text, 600)
    assert tail.split() == text.split()[-600:]
    assert "\n\n" in tail
    assert chapters.tail_words(u"ngắn thôi", 600) == u"ngắn thôi"


# ---------------------------------------------------------------------------
# Through the API
# ---------------------------------------------------------------------------

def test_legacy_chapter_lists_and_reads(workspace):
    """A pre-front-matter file lists with its heading title and reads whole."""
    client = workspace
    slug = _make(client, "Book A")
    _write(slug, "chapter-001-mo-dau.md", LEGACY)

    items = client.get("/api/chapters?novel=%s" % slug).json()["chapters"]
    assert len(items) == 1
    item = items[0]
    assert item["number"] == 1
    assert item["title"] == u"Mở đầu"
    assert item["scene_date"] is None and item["arc"] is None and item["summary"] == ""
    assert item["words"] == len(LEGACY.split())

    r = client.get("/api/chapters/chapter-001-mo-dau.md?novel=%s" % slug)
    assert r.status_code == 200, r.text
    assert r.json()["content"] == LEGACY
    assert r.json()["meta"] == {}


@pytest.mark.parametrize("text", [
    u"---\ntitle: \"unclosed\nchapter: [\n---\n# Chapter 2: Hỏng\n\nprose",
    u"---\nchapter: 2\nscene_date: 1789-02-30\n---\n# Chapter 2: Hỏng\n\nprose",
])
def test_malformed_front_matter_does_not_500(workspace, text):
    """
    One hand-edited block must not take the manuscript down. The unquoted
    impossible date is the case that did: a ValueError out of PyYAML went
    through the list, the read and the state snapshot alike, and every MCP
    tool reported a 500 for the whole novel.
    """
    client = workspace
    slug = _make(client, "Book A")
    _write(slug, "chapter-002-hong.md", text)

    r = client.get("/api/chapters?novel=%s" % slug)
    assert r.status_code == 200, r.text
    assert r.json()["chapters"][0]["number"] == 2

    r = client.get("/api/chapters/chapter-002-hong.md?novel=%s" % slug)
    assert r.status_code == 200, r.text
    assert r.json()["content"] == text, "a broken block must not eat the text"
    assert r.json()["meta"] == {}

    r = client.get("/api/novels/%s/state" % slug)
    assert r.status_code == 200, r.text
    assert r.json()["latest"]["number"] == 2


def test_a_bom_saved_chapter_lists_with_its_meta(workspace):
    """
    Through the API, the BOM case above: the file re-saved by a Windows
    editor listed as ``(3, 'chapter-003-ba', None)`` with its scene_date in
    the file, and the state tail began with the raw YAML block.
    """
    client = workspace
    slug = _make(client, "Book A")
    text = u"---\nchapter: 3\ntitle: Ba\nscene_date: \"1790\"\n---\n# Chapter 3: Ba\n\nprose"
    path = novels.get(slug).chapters_dir / "chapter-003-ba.md"
    path.write_bytes(codecs.BOM_UTF8 + text.encode("utf-8"))

    item = client.get("/api/chapters?novel=%s" % slug).json()["chapters"][0]
    assert (item["number"], item["title"], item["scene_date"]) == (3, "Ba", "1790")
    state = client.get("/api/novels/%s/state" % slug).json()
    assert state["latest_scene_date"] == "1790"
    assert state["latest"]["tail"].startswith("# Chapter 3")


def test_non_finite_numbers_in_front_matter_still_list_read_and_snapshot(workspace):
    """
    ``words: .inf`` and ``arc: .nan`` are valid YAML, so the block parses;
    the list then died in int() (OverflowError) and the read in the JSON
    encoder (``allow_nan=False``). The chapter must list with a counted word
    total, read with the bad numbers as null, and not take the snapshot down.
    """
    client = workspace
    slug = _make(client, "Book A")
    body = u"# Chapter 1\n\nprose here"
    _write(slug, "chapter-001-inf.md", u"---\nchapter: 1\nwords: .inf\narc: .nan\n---\n" + body)

    r = client.get("/api/chapters?novel=%s" % slug)
    assert r.status_code == 200, r.text
    item = r.json()["chapters"][0]
    assert item["words"] == len(body.split()) and item["arc"] is None

    r = client.get("/api/chapters/chapter-001-inf.md?novel=%s" % slug)
    assert r.status_code == 200, r.text
    assert r.json()["meta"] == {"chapter": 1, "words": None, "arc": None}

    assert client.get("/api/novels/%s/state" % slug).status_code == 200


def test_a_binary_title_in_front_matter_still_reads(workspace):
    """
    A ``!!binary`` scalar whose bytes are not UTF-8 passed the parser and
    the list, then blew up FastAPI's encoder on the read — the agent saw a
    chapter it could not open. Both endpoints must agree on one string.
    """
    client = workspace
    slug = _make(client, "Book A")
    _write(slug, "chapter-002-bin.md",
           u"---\nchapter: 2\ntitle: !!binary |\n  /w==\n---\n# Chapter 2\n\nprose")

    item = client.get("/api/chapters?novel=%s" % slug).json()["chapters"][0]
    r = client.get("/api/chapters/chapter-002-bin.md?novel=%s" % slug)
    assert r.status_code == 200, r.text
    assert isinstance(r.json()["meta"]["title"], str)
    assert r.json()["meta"]["title"] == item["title"]


def test_save_writes_front_matter_and_read_returns_body_only(workspace):
    client = workspace
    slug = _make(client, "Book A")
    content = u"Văn Tâm mở Đồ Lục.\n\nMột giọng nói vang lên."
    r = _save(client, slug, 3, u"Lời mời từ bóng tối", content,
              scene_date="1789-02", arc=1, summary=u"Hai dòng tóm tắt.")
    assert r.status_code == 200, r.text
    meta = r.json()["meta"]
    assert set(meta) == {"chapter", "title", "scene_date", "arc", "summary", "words", "saved_at"}
    assert meta["chapter"] == 3 and meta["scene_date"] == "1789-02" and meta["arc"] == 1
    assert meta["words"] == len(content.split())
    assert meta["saved_at"].endswith("+00:00")

    raw = (novels.get(slug).chapters_dir / r.json()["filename"]).read_text(encoding="utf-8")
    assert raw.startswith("---\n")
    assert chapters.parse_front_matter(raw)[0]["scene_date"] == "1789-02"

    read = client.get("/api/chapters/%s?novel=%s" % (r.json()["filename"], slug)).json()
    assert read["content"] == u"# Chapter 3: Lời mời từ bóng tối\n\n" + content
    assert not read["content"].startswith("---")
    assert read["meta"]["summary"] == u"Hai dòng tóm tắt."
    assert read["meta"]["scene_date"] == "1789-02"


def test_only_the_prose_is_indexed(workspace):
    """
    Neither the block nor the heading may become a retrievable segment —
    otherwise "saved_at" is a fact the writer can be handed.
    """
    client = workspace
    slug = _make(client, "Book A")
    r = _save(client, slug, 1, "Mo dau", u"Văn Tâm mở cuốn sổ.",
              scene_date="1789", summary=u"Tóm tắt riêng.")
    assert r.status_code == 200, r.text
    texts = [t.text for t in rp.get_engine(novels.get(slug)).traces.values()]
    assert texts, "the prose itself must be indexed"
    for text in texts:
        assert "saved_at" not in text and "scene_date" not in text
        assert u"Tóm tắt riêng" not in text
        assert not text.startswith("# Chapter")


def test_bad_scene_date_is_a_400_and_writes_nothing(workspace):
    """
    The same parser the history check uses decides; a date it cannot read
    would silently disable the anachronism check for that chapter.
    """
    client = workspace
    slug = _make(client, "Book A")
    r = _save(client, slug, 1, "x", "y", scene_date=u"khoảng 1789")
    assert r.status_code == 400, r.text
    assert u"khoảng 1789" in r.json()["detail"]
    assert "YYYY" in r.json()["detail"]
    assert list(novels.get(slug).chapters_dir.iterdir()) == []

    assert _save(client, slug, 1, "x", "y", scene_date="1789-13").status_code == 400


def test_blank_scene_date_means_none(workspace):
    """An empty input box is 'no date', not a 400 and not the string ''."""
    client = workspace
    slug = _make(client, "Book A")
    r = _save(client, slug, 1, "x", "y", scene_date="  ")
    assert r.status_code == 200, r.text
    assert r.json()["meta"]["scene_date"] is None
    read = client.get("/api/chapters/%s?novel=%s" % (r.json()["filename"], slug)).json()
    assert "scene_date" not in read["meta"], "absent fields stay out of the file"
    assert client.get("/api/chapters?novel=%s" % slug).json()["chapters"][0]["scene_date"] is None


def test_list_carries_number_date_arc_and_summary(workspace):
    client = workspace
    slug = _make(client, "Book A")
    assert _save(client, slug, 1, "Mo dau", u"Một.").status_code == 200
    assert _save(client, slug, 2, u"Bóng tối", u"Hai.",
                 scene_date="1789-02", arc=2, summary=u"Có khách.").status_code == 200

    items = client.get("/api/chapters?novel=%s" % slug).json()["chapters"]
    assert [i["number"] for i in items] == [1, 2]
    assert set(items[0]) == {"filename", "number", "title", "words", "size",
                             "scene_date", "arc", "summary"}
    assert items[0]["title"] == "Mo dau"
    assert items[0]["scene_date"] is None and items[0]["arc"] is None
    assert items[1]["title"] == u"Bóng tối"
    assert items[1]["scene_date"] == "1789-02"
    assert items[1]["arc"] == 2
    assert items[1]["summary"] == u"Có khách."
    assert items[1]["words"] == 1


def test_an_untitled_save_still_lists_with_a_title(workspace):
    client = workspace
    slug = _make(client, "Book A")
    assert _save(client, slug, 4, "", u"prose").status_code == 200
    item = client.get("/api/chapters?novel=%s" % slug).json()["chapters"][0]
    assert item["number"] == 4
    assert item["title"] == "Chapter 4"
