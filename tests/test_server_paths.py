# -*- coding: utf-8 -*-
"""
Path-containment tests for the chapter endpoints.

``GET /api/chapters/..%5C..%5Cx`` used to read arbitrary files: Starlette
percent-decodes the path parameter, and on Windows ``Path`` treats the decoded
backslash as a separator, so the join walked out of chapters_dir. The server
binds 0.0.0.0 by default, which made that reachable from the network.
"""

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("EMBEDDING_PROVIDER", "hash")

from fastapi.testclient import TestClient  # noqa: E402

from backend.server import _slugify, app, is_safe_child_name  # noqa: E402

client = TestClient(app)

TRAVERSALS = [
    "..%5Crequirements.txt",
    "..%5C..%5Crequirements.txt",
    "..%2Frequirements.txt",
    "....%5C%5Crequirements.txt",
    "%2e%2e%5crequirements.txt",
]


@pytest.mark.parametrize("probe", TRAVERSALS)
def test_chapter_read_cannot_escape_the_chapters_directory(probe):
    resp = client.get("/api/chapters/" + probe)
    assert resp.status_code in (400, 404), resp.text
    assert "fastapi" not in resp.text.lower(), "leaked requirements.txt"


def test_legitimate_missing_chapter_is_a_404():
    resp = client.get("/api/chapters/chapter-001-nope.md")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Name validation — pure string logic, so this run proves both platforms
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", [
    "..\\..\\evil.md",      # a separator on Windows, a legal filename on Linux
    "..\\evil.md",
    "../evil.md",
    "sub/evil.md",
    "..",
    ".",
    ".hidden",
    "",
    "   ",
    "nul.md",
    "COM1.txt",
    "with\x00null.md",
    "x" * 201,
])
def test_unsafe_names_are_rejected_on_every_platform(name):
    """
    The reason this is a string test and not an HTTP test: `Path` disagrees
    across operating systems about what `..\\..\\x` means, so an HTTP test
    passing on Windows said nothing about Linux — which is exactly how CI
    caught a case that wrote a file named `..\\..\\evil.md`.
    """
    assert is_safe_child_name(name) is False, name


@pytest.mark.parametrize("name", [
    "characters.md",
    "chapter-001-mo-dau.md",
    "nhân-vật.md",
    "world_2.md",
    "a.md",
    "Đồ Lục.md",
])
def test_ordinary_names_are_accepted(name):
    """The tightening must not reject the names people actually use."""
    assert is_safe_child_name(name) is True, name


@pytest.mark.parametrize("title,expected", [
    ("../../etc/passwd", "etc-passwd"),
    ("..\\..\\windows", "windows"),
    ("Khởi đầu", u"khởi-đầu"),
    ("", "untitled"),
    ("///", "untitled"),
    ("Chapter: One!", "chapter-one"),
])
def test_slugify_cannot_produce_a_path_separator(title, expected):
    slug = _slugify(title)
    assert slug == expected
    assert "/" not in slug and "\\" not in slug and ".." not in slug
