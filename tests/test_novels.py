# -*- coding: utf-8 -*-
"""
Novel isolation.

The point of separate workspaces is that retrieval for one book can never
reach another. These tests assert that end-to-end through the HTTP API, not
just at the path level — a shared engine cache or a forgotten `novel=` argument
would leak, and both are invisible to a filesystem-only check.
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient  # noqa: E402

from backend import novels  # noqa: E402
import backend.rag_pipeline as rp  # noqa: E402
from backend.config import settings  # noqa: E402
from backend.server import app  # noqa: E402

SCIFI = u"""# Nhân vật

## Văn Tâm (Nhân vật chính)

- **Năng lực**: Sở hữu Đồ Lục, cuốn sổ đen ghi 108 Vật Chất.
- **Ngoại hình**: Cao 1m78, tóc đen rối nhẹ.
"""

NOIR = u"""# Nhân vật

## Lâm Phong (Thám tử)

- **Năng lực**: Trí nhớ tuyệt đối, đọc được nét mặt.
- **Ngoại hình**: Áo khoác xám, mũ phớt cũ.
"""


@pytest.fixture()
def workspace(tmp_path, monkeypatch):
    """Point the novels registry at a scratch directory, engines included."""
    monkeypatch.setattr(settings, "novels_dir", str(tmp_path / "novels"))
    monkeypatch.setattr(settings, "default_novel", "default")
    # A fresh cache per test: engines are keyed by slug, and slugs repeat.
    monkeypatch.setattr(rp, "_engines", {})
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


def _seed(client, slug, filename, content):
    r = client.put("/api/context/%s" % filename, json={"novel": slug, "content": content})
    assert r.status_code == 200, r.text
    return r.json()


# ---------------------------------------------------------------------------
# Isolation
# ---------------------------------------------------------------------------

def test_search_never_crosses_novels(workspace):
    client = workspace
    a = _make(client, u"Đồ Lục Ký Sự")
    b = _make(client, u"Thám Tử Lâm Phong")
    _seed(client, a, "characters.md", SCIFI)
    _seed(client, b, "characters.md", NOIR)

    def sources(slug, query):
        r = client.post("/api/search", json={"novel": slug, "query": query, "top_k": 8})
        assert r.status_code == 200, r.text
        return " ".join(h["text"] for h in r.json()["results"])

    assert u"Văn Tâm" in sources(a, u"nhân vật chính là ai")
    assert u"Lâm Phong" not in sources(a, u"Lâm Phong thám tử")

    assert u"Lâm Phong" in sources(b, u"thám tử là ai")
    assert u"Văn Tâm" not in sources(b, u"Văn Tâm Đồ Lục")


def test_each_novel_has_its_own_store(workspace):
    client = workspace
    a = _make(client, "Book A")
    b = _make(client, "Book B")
    _seed(client, a, "characters.md", SCIFI)

    stats_a = client.get("/api/vectordb/stats?novel=%s" % a).json()
    stats_b = client.get("/api/vectordb/stats?novel=%s" % b).json()
    assert stats_a["segments"] > 0
    assert stats_b["segments"] == 0, "an empty novel must start empty"

    assert novels.get(a).db_path != novels.get(b).db_path
    assert novels.get(a).db_path.exists()


def test_gazetteer_is_per_novel(workspace):
    """
    The entity graph is seeded from context/, so a shared gazetteer would make
    one novel's cast extractable from another's prose.
    """
    client = workspace
    a = _make(client, "Book A")
    b = _make(client, "Book B")
    _seed(client, a, "characters.md", SCIFI)
    _seed(client, b, "characters.md", NOIR)

    found_here = client.get("/api/memory/entity/%s?novel=%s" % (u"Văn Tâm", a)).json()
    found_there = client.get("/api/memory/entity/%s?novel=%s" % (u"Văn Tâm", b)).json()
    assert found_here["found"] is True
    assert found_there["found"] is False


def test_chapters_are_per_novel(workspace):
    client = workspace
    a = _make(client, "Book A")
    b = _make(client, "Book B")

    r = client.post("/api/chapters/save", json={
        "novel": a, "chapter_number": 1, "title": "Mo dau", "content": u"Nội dung A.",
    })
    assert r.status_code == 200, r.text

    assert len(client.get("/api/chapters?novel=%s" % a).json()["chapters"]) == 1
    assert client.get("/api/chapters?novel=%s" % b).json()["chapters"] == []
    assert novels.get(a).chapters_dir.exists()


def test_rules_fall_back_to_project_then_override(workspace):
    client = workspace
    slug = _make(client, "Book A")

    rules = client.get("/api/rules?novel=%s" % slug).json()["rules"]
    names = dict((r["name"], r["scope"]) for r in rules)
    assert names, "project-level rules should still apply"
    assert set(names.values()) == {"project"}

    # A novel-level file with the same name replaces the shared one rather
    # than being concatenated after it.
    target = sorted(names)[0]
    (novels.get(slug).rules_dir).mkdir(parents=True, exist_ok=True)
    (novels.get(slug).rules_dir / ("%s.md" % target)).write_text(
        "NOVEL OVERRIDE", encoding="utf-8")

    rules = client.get("/api/rules?novel=%s" % slug).json()["rules"]
    entry = [r for r in rules if r["name"] == target][0]
    assert entry["scope"] == "novel"
    assert entry["content"] == "NOVEL OVERRIDE"
    assert len(rules) == len(names), "an override must not duplicate the rule"

    from backend.agent import load_rules
    combined = load_rules(novels.get(slug))
    assert combined.count("NOVEL OVERRIDE") == 1


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def test_vietnamese_titles_get_usable_slugs(workspace):
    client = workspace
    assert _make(client, u"Đồ Lục Ký Sự") == "do-luc-ky-su"
    assert _make(client, u"Thám Tử Lâm Phong") == "tham-tu-lam-phong"


def test_duplicate_title_is_rejected(workspace):
    client = workspace
    _make(client, "Book A")
    r = client.post("/api/novels", json={"title": "Book A"})
    assert r.status_code == 400
    assert "already exists" in r.json()["detail"]


def test_delete_removes_the_workspace(workspace):
    client = workspace
    slug = _make(client, "Book A")
    _seed(client, slug, "characters.md", SCIFI)
    root = novels.get(slug).root
    assert root.exists()

    assert client.delete("/api/novels/%s" % slug).status_code == 200
    assert not root.exists()
    assert client.get("/api/chapters?novel=%s" % slug).status_code == 404


def test_default_workspace_cannot_be_deleted(workspace):
    client = workspace
    client.get("/api/novels")
    r = client.delete("/api/novels/default")
    assert r.status_code == 400


def test_unknown_novel_is_404_not_a_silent_default(workspace):
    """A typo'd id must never quietly write into the default workspace."""
    client = workspace
    assert client.get("/api/chapters?novel=nope").status_code == 404
    r = client.post("/api/search", json={"novel": "nope", "query": "x"})
    assert r.status_code == 404


@pytest.mark.parametrize("bad", ["..", "../etc", "..%5Cwin", "A B", "x/y", ""])
def test_slug_validation_rejects_path_traversal(workspace, bad):
    client = workspace
    r = client.get("/api/chapters", params={"novel": bad} if bad else {})
    # An empty id legitimately means "the default workspace".
    assert r.status_code in (200, 400, 404)
    if bad:
        assert r.status_code in (400, 404), bad


def test_omitting_the_novel_uses_the_default(workspace):
    """Every pre-multi-novel URL keeps working unchanged."""
    client = workspace
    assert client.get("/api/chapters").json()["novel"] == "default"
    assert client.get("/api/vectordb/stats").json()["novel"] == "default"
    assert client.post("/api/search", json={"query": "x"}).json()["novel"] == "default"


# ---------------------------------------------------------------------------
# Context files
# ---------------------------------------------------------------------------

def test_context_write_is_immediately_retrievable(workspace):
    client = workspace
    slug = _make(client, "Book A")
    res = _seed(client, slug, "characters.md", SCIFI)
    assert res["segments"] > 0

    files = client.get("/api/context?novel=%s" % slug).json()["files"]
    assert [f["filename"] for f in files] == ["characters.md"]

    hits = client.post("/api/search", json={"novel": slug, "query": u"Đồ Lục"}).json()
    assert hits["results"], "a written context file must be searchable at once"


def test_deleting_context_removes_it_from_memory(workspace):
    client = workspace
    slug = _make(client, "Book A")
    _seed(client, slug, "characters.md", SCIFI)

    assert client.delete("/api/context/characters.md?novel=%s" % slug).status_code == 200
    assert client.get("/api/context?novel=%s" % slug).json()["files"] == []
    hits = client.post("/api/search", json={"novel": slug, "query": u"Đồ Lục"}).json()
    assert not hits["results"], "deleted context must not stay retrievable"


@pytest.mark.parametrize("bad", [
    "..%5C..%5Cevil.md",   # backslash: a separator on Windows, a filename char on Linux
    "..%5Cevil.md",
    "%2e%2e%5cevil.md",
    "..%2Fevil.md",        # these never reach the handler: a path param
    "sub%2Fevil.md",       # cannot match across '/', so the router 404s first
    "..",
    ".hidden.md",
])
def test_context_paths_cannot_escape(workspace, bad):
    """
    Rejected, and — the part that actually matters — nothing written.

    Status alone is not the invariant: `%2F` forms are 404s from the router
    before any handler runs, while backslash forms reach the handler and are
    400. What must hold on every platform is that no file appears. `..\\..\\x`
    traverses on Windows but is a legal filename on Linux, so the containment
    check alone let Linux return 200 and create a file named `..\\..\\evil.md`.
    """
    client = workspace
    slug = _make(client, "Book A")
    r = client.put("/api/context/" + bad, json={"novel": slug, "content": "x"})
    assert r.status_code in (400, 404), r.text

    assert client.get("/api/context?novel=%s" % slug).json()["files"] == []
    assert list(novels.get(slug).context_dir.iterdir()) == [], "a file was created"


@pytest.mark.parametrize("bad", ["..%5C..%5Cevil.md", "..%5Cx.md", "%2e%2e%5cx.md"])
def test_backslash_names_are_refused_by_the_handler(workspace, bad):
    """These do reach the handler, so they must be an explicit 400 everywhere."""
    client = workspace
    slug = _make(client, "Book A")
    assert client.put("/api/context/" + bad,
                      json={"novel": slug, "content": "x"}).status_code == 400
    assert client.get("/api/chapters/%s?novel=%s" % (bad, slug)).status_code == 400
    assert client.delete("/api/chapters/%s?novel=%s" % (bad, slug)).status_code == 400


def test_chapter_paths_cannot_escape_either(workspace):
    client = workspace
    slug = _make(client, "Book A")
    for bad in ("..%5C..%5Cevil.md", "..%2Fevil.md", "sub%2Fx.md"):
        assert client.get("/api/chapters/%s?novel=%s"
                          % (bad, slug)).status_code in (400, 404)
        assert client.delete("/api/chapters/%s?novel=%s"
                             % (bad, slug)).status_code in (400, 404)
    assert list(novels.get(slug).chapters_dir.iterdir()) == []


def test_ordinary_filenames_still_work(workspace):
    """
    The tightening must not reject the names people actually use.

    ASCII names only here: the *content* is Vietnamese, which is what this
    project cares about, while a Unicode filename would be testing the runner's
    filesystem locale rather than any code of ours. Unicode names are covered
    by the pure-string check in test_server_paths.py, which has no such
    dependency.
    """
    client = workspace
    slug = _make(client, "Book A")
    for name in ("characters.md", "dia-diem.md", "world_2.md"):
        r = client.put("/api/context/" + name,
                       json={"novel": slug, "content": u"## Văn Tâm\n\nNội dung."})
        assert r.status_code == 200, (name, r.text)
        assert client.get("/api/context/%s?novel=%s" % (name, slug)).status_code == 200
