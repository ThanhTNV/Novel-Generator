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

from backend.server import _slugify, app  # noqa: E402

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
