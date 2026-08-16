"""
Per-novel workspaces.

Every novel owns its whole world: its context files, its saved chapters, its
rules and skills, and — the point of the exercise — its own Zero-Mem database.
Nothing is shared at retrieval time, so generating chapter 12 of one book can
never surface a character from another.

    novels/
      <slug>/
        novel.json          title, created_at, description
        context/*.md        world bible — the gazetteer is built from this
        chapters/*.md       saved chapters
        rules/*.md          optional; overrides the project-level rule of the
        skills/*.md         same name, and adds any new ones
        memory/zero_mem.db  isolated store: traces, entity graph, embeddings

Rules, skills and prompts still have project-level defaults, because "write in
third person past tense" is usually a house style rather than a per-book
decision. A novel that wants different ones drops a file with the same name
into its own directory.
"""

import json
import os
import re
import shutil
import time
from pathlib import Path
from typing import Dict, List, Optional

from backend.config import ROOT_DIR, settings

# A slug becomes a directory name, so it is validated rather than sanitised:
# anything that is not already safe is rejected, and callers slugify first.
_SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")
_SLUG_STRIP_RE = re.compile(r"[^\w\-]+", re.UNICODE)

# The subdirectories that make a directory a novel workspace.
CONTEXT = "context"
CHAPTERS = "chapters"
RULES = "rules"
SKILLS = "skills"
MEMORY = "memory"

DB_NAME = "zero_mem.db"
MANIFEST = "novel.json"


class NovelNotFound(KeyError):
    pass


class InvalidNovelName(ValueError):
    pass


def slugify(name: str, limit: int = 48) -> str:
    """
    Turn a title into a directory-safe slug.

    Vietnamese titles are the norm here, so accents are folded rather than
    dropped — 'Đồ Lục Ký Sự' has to become something, not nothing.
    """
    import unicodedata

    folded = unicodedata.normalize("NFD", name)
    folded = "".join(c for c in folded if not unicodedata.combining(c))
    folded = folded.replace("đ", "d").replace("Đ", "d")
    slug = _SLUG_STRIP_RE.sub("-", folded.lower())
    slug = re.sub(r"-{2,}", "-", slug).strip("-")
    slug = slug[:limit].strip("-")
    # A title of pure punctuation, or one that folds away entirely, still needs
    # a stable home; the timestamp keeps it unique without a collision check.
    return slug or ("novel-%d" % int(time.time()))


class Novel(object):
    """One isolated workspace. Paths only — no engine, no I/O beyond metadata."""

    __slots__ = ("slug", "root", "_meta")

    def __init__(self, slug: str, root: Path, meta: Optional[Dict] = None):
        self.slug = slug
        self.root = root
        self._meta = meta or {}

    # -- paths --------------------------------------------------------------

    @property
    def context_dir(self) -> Path:
        return self.root / CONTEXT

    @property
    def chapters_dir(self) -> Path:
        return self.root / CHAPTERS

    @property
    def rules_dir(self) -> Path:
        return self.root / RULES

    @property
    def skills_dir(self) -> Path:
        return self.root / SKILLS

    @property
    def db_path(self) -> Path:
        return self.root / MEMORY / DB_NAME

    @property
    def manifest_path(self) -> Path:
        return self.root / MANIFEST

    # -- metadata -----------------------------------------------------------

    @property
    def title(self) -> str:
        return self._meta.get("title") or self.slug

    @property
    def description(self) -> str:
        return self._meta.get("description", "")

    @property
    def created_at(self) -> float:
        return float(self._meta.get("created_at", 0.0))

    def counts(self) -> Dict[str, int]:
        """Cheap filesystem counts for the novel switcher."""
        def n(directory: Path, pattern: str) -> int:
            if not directory.exists():
                return 0
            return sum(1 for _ in directory.glob(pattern))

        return {
            "context_files": n(self.context_dir, "**/*.md"),
            "chapters": n(self.chapters_dir, "chapter-*.md"),
        }

    def to_dict(self) -> Dict:
        d = {
            "slug": self.slug,
            "title": self.title,
            "description": self.description,
            "created_at": self.created_at,
            "has_memory": self.db_path.exists(),
        }
        d.update(self.counts())
        return d

    def write_manifest(self) -> None:
        self._meta.setdefault("created_at", time.time())
        self.manifest_path.write_text(
            json.dumps(self._meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def update(self, title: Optional[str] = None, description: Optional[str] = None) -> None:
        if title is not None:
            self._meta["title"] = title.strip() or self.slug
        if description is not None:
            self._meta["description"] = description.strip()
        self.write_manifest()

    def scaffold(self) -> None:
        for sub in (CONTEXT, CHAPTERS, RULES, SKILLS, MEMORY):
            (self.root / sub).mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def novels_root() -> Path:
    return Path(settings.novels_dir)


def _read_manifest(path: Path) -> Dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _resolve(slug: str) -> Path:
    """
    Map a slug to its directory, refusing anything that could leave the root.

    The slug arrives from an HTTP path or body, so validation is a containment
    check and not a formality — `..%5C..%5C` reaches request handlers, and on
    Windows the decoded backslash is a path separator.
    """
    if not isinstance(slug, str) or not _SLUG_RE.match(slug):
        raise InvalidNovelName(
            "Novel id must be 1-64 chars of lowercase letters, digits or dashes."
        )
    base = novels_root().resolve()
    candidate = (base / slug).resolve()
    if base not in candidate.parents:
        raise InvalidNovelName("Novel id escapes the novels directory.")
    return candidate


def exists(slug: str) -> bool:
    try:
        return _resolve(slug).is_dir()
    except InvalidNovelName:
        return False


def get(slug: str) -> Novel:
    path = _resolve(slug)
    if not path.is_dir():
        raise NovelNotFound(slug)
    return Novel(slug, path, _read_manifest(path / MANIFEST))


def list_novels() -> List[Novel]:
    """Every workspace on disk, newest first, then alphabetical."""
    root = novels_root()
    if not root.exists():
        return []
    found: List[Novel] = []
    for entry in sorted(root.iterdir()):
        if not entry.is_dir() or entry.name.startswith("."):
            continue
        if not _SLUG_RE.match(entry.name):
            continue
        found.append(Novel(entry.name, entry, _read_manifest(entry / MANIFEST)))
    found.sort(key=lambda n: (-n.created_at, n.title.lower()))
    return found


def create(title: str, description: str = "", slug: Optional[str] = None) -> Novel:
    """Create a workspace. Raises InvalidNovelName if the slug is taken."""
    title = (title or "").strip()
    if not title:
        raise InvalidNovelName("A novel needs a title.")
    slug = slug or slugify(title)
    path = _resolve(slug)
    if path.exists():
        raise InvalidNovelName("A novel with the id '%s' already exists." % slug)
    novel = Novel(slug, path, {"title": title, "description": description.strip(),
                               "created_at": time.time()})
    novel.scaffold()
    novel.write_manifest()
    return novel


def delete(slug: str) -> None:
    """Remove a workspace and everything in it."""
    path = _resolve(slug)
    if not path.is_dir():
        raise NovelNotFound(slug)
    shutil.rmtree(path)


# ---------------------------------------------------------------------------
# First run
# ---------------------------------------------------------------------------

def ensure_default(logger=None) -> Novel:
    """
    Guarantee the default workspace exists, seeding it from the pre-multi-novel
    layout on first run.

    The legacy files are **copied, not moved**: an upgrade that silently
    relocated someone's manuscript would be the wrong kind of surprise, and the
    originals cost nothing to leave in place.
    """
    def note(msg: str) -> None:
        if logger:
            logger(msg)

    slug = settings.default_novel
    if exists(slug):
        return get(slug)

    novel = create("My Novel", slug=slug)
    note("novels: created the default workspace at %s" % novel.root)

    legacy_context = Path(settings.context_dir)
    if legacy_context.is_dir() and legacy_context != novel.context_dir:
        copied = 0
        for src in sorted(legacy_context.glob("*.md")):
            dest = novel.context_dir / src.name
            if not dest.exists():
                shutil.copy2(str(src), str(dest))
                copied += 1
        if copied:
            note("novels: copied %d context file(s) from %s into '%s'"
                 % (copied, legacy_context, slug))

    legacy_chapters = Path(settings.chapters_dir)
    if legacy_chapters.is_dir() and legacy_chapters != novel.chapters_dir:
        copied = 0
        for src in sorted(legacy_chapters.glob("chapter-*.md")):
            dest = novel.chapters_dir / src.name
            if not dest.exists():
                shutil.copy2(str(src), str(dest))
                copied += 1
        if copied:
            note("novels: copied %d chapter(s) from %s into '%s'"
                 % (copied, legacy_chapters, slug))

    legacy_db = Path(settings.zero_mem_db)
    if legacy_db.is_file() and legacy_db != novel.db_path and not novel.db_path.exists():
        shutil.copy2(str(legacy_db), str(novel.db_path))
        note("novels: copied the existing Zero-Mem store into '%s'" % slug)

    return novel


def resolve_or_default(slug: Optional[str], logger=None) -> Novel:
    """
    The one entry point request handlers use.

    A blank or absent id means the default workspace, which keeps every
    pre-existing URL working exactly as it did before novels were separated.
    """
    if not slug:
        return ensure_default(logger)
    novel = get(slug)          # raises NovelNotFound / InvalidNovelName
    novel.scaffold()           # tolerate a hand-made directory
    return novel
