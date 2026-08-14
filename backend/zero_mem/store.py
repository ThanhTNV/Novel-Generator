"""
Provenance-preserving trace store (SQLite, stdlib only).

Segments are kept verbatim as the source of record; nothing generated is ever
written back over them.

The critical difference from the previous ChromaDB pipeline is **supersession**:
re-ingesting a source deletes that source's previous segments inside one
transaction. The old pipeline keyed chunks by ``sha256(chunk_text)``, so editing
a chapter inserted new rows and left the old ones behind forever — after a few
revisions the store held several contradictory versions of the same scene, all
retrievable, which is exactly what makes generated chapters drift.
"""

import json
import os
import sqlite3
import threading
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

SCHEMA = """
CREATE TABLE IF NOT EXISTS sources (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT NOT NULL UNIQUE,
    kind        TEXT NOT NULL DEFAULT 'document',
    chapter     INTEGER,
    ordinal     INTEGER NOT NULL DEFAULT 0,
    updated_at  REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS segments (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    source_id   INTEGER NOT NULL REFERENCES sources(id),
    seq         INTEGER NOT NULL,
    scene       INTEGER NOT NULL DEFAULT 0,
    heading     TEXT NOT NULL DEFAULT '',
    kind        TEXT NOT NULL DEFAULT 'prose',
    text        TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_segments_source ON segments(source_id, seq);

CREATE TABLE IF NOT EXISTS entities (
    key         TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    type        TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS mentions (
    segment_id  INTEGER NOT NULL REFERENCES segments(id),
    entity_key  TEXT NOT NULL REFERENCES entities(key),
    count       INTEGER NOT NULL,
    PRIMARY KEY (segment_id, entity_key)
);
CREATE INDEX IF NOT EXISTS idx_mentions_entity ON mentions(entity_key);

CREATE TABLE IF NOT EXISTS embeddings (
    segment_id  INTEGER NOT NULL REFERENCES segments(id),
    space       TEXT NOT NULL,
    vec         TEXT NOT NULL,
    PRIMARY KEY (segment_id, space)
);
"""


class Trace(object):
    """One stored narrative unit with full provenance."""

    __slots__ = ("id", "source", "kind", "chapter", "ordinal", "seq", "scene", "heading", "seg_kind", "text")

    def __init__(self, id, source, kind, chapter, ordinal, seq, scene, heading, seg_kind, text):
        self.id = id
        self.source = source
        self.kind = kind
        self.chapter = chapter
        self.ordinal = ordinal
        self.seq = seq
        self.scene = scene
        self.heading = heading
        self.seg_kind = seg_kind
        self.text = text

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "source": self.source,
            "kind": self.kind,
            "chapter": self.chapter,
            "ordinal": self.ordinal,
            "seq": self.seq,
            "scene": self.scene,
            "heading": self.heading,
            "segment_kind": self.seg_kind,
            "text": self.text,
        }


_TRACE_COLUMNS = """
    seg.id, src.name, src.kind, src.chapter, src.ordinal,
    seg.seq, seg.scene, seg.heading, seg.kind, seg.text
"""


def _row_to_trace(row: Sequence) -> Trace:
    return Trace(*row)


class TraceStore(object):
    def __init__(self, path: str):
        self.path = path
        if path != ":memory:":
            parent = os.path.dirname(os.path.abspath(path))
            if parent:
                try:
                    os.makedirs(parent)
                except OSError:
                    pass
        # FastAPI serves requests from a thread pool, so the connection is
        # shared across threads and guarded by an explicit lock.
        self._lock = threading.RLock()
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.executescript(SCHEMA)
        self._db.commit()

    def close(self) -> None:
        with self._lock:
            self._db.close()

    # -- writing ------------------------------------------------------------

    def replace_source(
        self,
        name: str,
        segments: Sequence,               # Sequence[Segment]
        mentions_per_segment: Sequence,   # Sequence[Iterable[Mention]]
        kind: str = "document",
        chapter: Optional[int] = None,
        ordinal: int = 0,
        timestamp: float = 0.0,
    ) -> List[int]:
        """
        Atomically replace everything previously stored for ``name``.

        Returns the new segment ids, in order.
        """
        with self._lock:
            cur = self._db.cursor()
            cur.execute("BEGIN")
            try:
                cur.execute("SELECT id FROM sources WHERE name = ?", (name,))
                row = cur.fetchone()
                if row:
                    source_id = row[0]
                    cur.execute(
                        "DELETE FROM mentions WHERE segment_id IN "
                        "(SELECT id FROM segments WHERE source_id = ?)", (source_id,))
                    cur.execute(
                        "DELETE FROM embeddings WHERE segment_id IN "
                        "(SELECT id FROM segments WHERE source_id = ?)", (source_id,))
                    cur.execute("DELETE FROM segments WHERE source_id = ?", (source_id,))
                    cur.execute(
                        "UPDATE sources SET kind=?, chapter=?, ordinal=?, updated_at=? WHERE id=?",
                        (kind, chapter, ordinal, timestamp, source_id))
                else:
                    cur.execute(
                        "INSERT INTO sources (name, kind, chapter, ordinal, updated_at) "
                        "VALUES (?, ?, ?, ?, ?)",
                        (name, kind, chapter, ordinal, timestamp))
                    source_id = cur.lastrowid

                ids: List[int] = []
                for seg, mentions in zip(segments, mentions_per_segment):
                    cur.execute(
                        "INSERT INTO segments (source_id, seq, scene, heading, kind, text) "
                        "VALUES (?, ?, ?, ?, ?, ?)",
                        (source_id, seg.index, seg.scene, seg.heading, seg.kind, seg.text))
                    seg_id = cur.lastrowid
                    ids.append(seg_id)
                    for m in mentions:
                        cur.execute(
                            "INSERT OR IGNORE INTO entities (key, name, type) VALUES (?, ?, ?)",
                            (m.key, m.name, m.type))
                        cur.execute(
                            "INSERT INTO mentions (segment_id, entity_key, count) VALUES (?, ?, ?) "
                            "ON CONFLICT(segment_id, entity_key) DO UPDATE SET count = count + excluded.count",
                            (seg_id, m.key, m.count))
                cur.execute(
                    "DELETE FROM entities WHERE key NOT IN (SELECT DISTINCT entity_key FROM mentions)")
                self._db.commit()
                return ids
            except Exception:
                self._db.rollback()
                raise

    def save_embedding(self, segment_id: int, space: str, vec: List[float]) -> None:
        with self._lock:
            self._db.execute(
                "INSERT INTO embeddings (segment_id, space, vec) VALUES (?, ?, ?) "
                "ON CONFLICT(segment_id, space) DO UPDATE SET vec = excluded.vec",
                (segment_id, space, json.dumps([round(v, 6) for v in vec])))
            self._db.commit()

    def save_embeddings(self, items: Iterable[Tuple[int, List[float]]], space: str) -> None:
        with self._lock:
            self._db.executemany(
                "INSERT INTO embeddings (segment_id, space, vec) VALUES (?, ?, ?) "
                "ON CONFLICT(segment_id, space) DO UPDATE SET vec = excluded.vec",
                [(sid, space, json.dumps([round(v, 6) for v in vec])) for sid, vec in items])
            self._db.commit()

    def delete_source(self, name: str) -> int:
        with self._lock:
            cur = self._db.cursor()
            cur.execute("SELECT id FROM sources WHERE name = ?", (name,))
            row = cur.fetchone()
            if not row:
                return 0
            source_id = row[0]
            cur.execute("SELECT COUNT(*) FROM segments WHERE source_id = ?", (source_id,))
            n = cur.fetchone()[0]
            cur.execute("DELETE FROM mentions WHERE segment_id IN "
                        "(SELECT id FROM segments WHERE source_id = ?)", (source_id,))
            cur.execute("DELETE FROM embeddings WHERE segment_id IN "
                        "(SELECT id FROM segments WHERE source_id = ?)", (source_id,))
            cur.execute("DELETE FROM segments WHERE source_id = ?", (source_id,))
            cur.execute("DELETE FROM sources WHERE id = ?", (source_id,))
            cur.execute("DELETE FROM entities WHERE key NOT IN (SELECT DISTINCT entity_key FROM mentions)")
            self._db.commit()
            return n

    def clear(self) -> None:
        with self._lock:
            for table in ("mentions", "embeddings", "segments", "sources", "entities"):
                self._db.execute("DELETE FROM %s" % table)
            self._db.commit()

    # -- reading ------------------------------------------------------------

    def all_traces(self) -> List[Trace]:
        with self._lock:
            rows = self._db.execute(
                "SELECT %s FROM segments seg JOIN sources src ON src.id = seg.source_id "
                "ORDER BY src.ordinal, src.id, seg.seq" % _TRACE_COLUMNS).fetchall()
        return [_row_to_trace(r) for r in rows]

    def all_mentions(self) -> List[Tuple[int, str, int]]:
        with self._lock:
            return self._db.execute(
                "SELECT segment_id, entity_key, count FROM mentions").fetchall()

    def entities(self) -> List[Tuple[str, str, str, int]]:
        with self._lock:
            return self._db.execute(
                "SELECT e.key, e.name, e.type, COALESCE(SUM(m.count), 0) "
                "FROM entities e LEFT JOIN mentions m ON m.entity_key = e.key "
                "GROUP BY e.key ORDER BY 4 DESC").fetchall()

    def load_embeddings(self, space: str) -> Dict[int, List[float]]:
        with self._lock:
            rows = self._db.execute(
                "SELECT segment_id, vec FROM embeddings WHERE space = ?", (space,)).fetchall()
        return dict((sid, json.loads(vec)) for sid, vec in rows)

    def sources(self) -> List[Dict[str, Any]]:
        with self._lock:
            rows = self._db.execute(
                "SELECT src.name, src.kind, src.chapter, src.ordinal, src.updated_at, "
                "COUNT(seg.id) FROM sources src LEFT JOIN segments seg ON seg.source_id = src.id "
                "GROUP BY src.id ORDER BY src.ordinal, src.id").fetchall()
        return [
            {"name": r[0], "kind": r[1], "chapter": r[2], "ordinal": r[3],
             "updated_at": r[4], "segments": r[5]}
            for r in rows
        ]

    def stats(self) -> Dict[str, int]:
        with self._lock:
            def one(sql: str) -> int:
                return self._db.execute(sql).fetchone()[0]
            return {
                "segments": one("SELECT COUNT(*) FROM segments"),
                "sources": one("SELECT COUNT(*) FROM sources"),
                "entities": one("SELECT COUNT(*) FROM entities"),
                "mentions": one("SELECT COUNT(*) FROM mentions"),
            }
