# -*- coding: utf-8 -*-
"""
Verified historical reference for the writing model.

    from backend.history import get_index, TOOLS, dispatch

The corpus lives in two places and merges by record id:

    history/*.yaml                  shared factual record (project-level)
    novels/<slug>/history/*.yaml    this novel's additions and divergence points

A record without a source is refused at load time, so anything this package can
hand a model is traceable. See ``records`` for the rest of the integrity rules.
"""

import threading
from typing import Any, Dict, List, Optional

from backend import novels
from backend.config import settings

from .index import ANACHRONISMS, HistoryIndex, load_index
from .records import HistoryError, HistoryRecord, format_date, parse_date

__all__ = [
    "HistoryIndex", "HistoryRecord", "HistoryError",
    "get_index", "reload_index", "TOOLS", "dispatch",
    "ANACHRONISMS", "format_date", "parse_date",
]

_indexes: Dict[str, HistoryIndex] = {}
_lock = threading.RLock()


def get_index(novel=None) -> HistoryIndex:
    """The merged corpus for one novel, built on first use and cached."""
    if not isinstance(novel, novels.Novel):
        novel = novels.resolve_or_default(novel)
    cached = _indexes.get(novel.slug)
    if cached is not None:
        return cached
    with _lock:
        cached = _indexes.get(novel.slug)
        if cached is None:
            cached = load_index(
                ("project", settings.history_dir),
                ("novel", str(novel.root / "history")),
            )
            _indexes[novel.slug] = cached
        return cached


def reload_index(novel=None) -> HistoryIndex:
    """Drop the cache after the corpus on disk changes."""
    if not isinstance(novel, novels.Novel):
        novel = novels.resolve_or_default(novel)
    with _lock:
        _indexes.pop(novel.slug, None)
    return get_index(novel)


# ---------------------------------------------------------------------------
# The tool surface exposed to the model
# ---------------------------------------------------------------------------

# Provider-neutral. api_client translates these into each vendor's shape.
TOOLS: List[Dict[str, Any]] = [
    {
        "name": "search_history",
        "description": (
            "Tra cứu sử liệu đã được thẩm định cho cuốn tiểu thuyết này. "
            "CHỈ trả về những ghi chép có nguồn dẫn; nếu không có ghi chép nào "
            "khớp, công cụ trả về danh sách rỗng — khi đó hãy nói rõ là không "
            "có cứ liệu, TUYỆT ĐỐI không tự bịa ra sự kiện, niên đại hay nhân "
            "vật lịch sử. Mỗi kết quả kèm nguồn; hãy viết dựa trên chúng.\n\n"
            "Look up vetted historical records for this novel. Returns only "
            "sourced records, or an empty list. Never invent history that this "
            "tool did not return."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Nhân vật, sự kiện, chính sách hoặc địa danh cần tra.",
                },
                "before": {
                    "type": "string",
                    "description": (
                        "Ngày của cảnh đang viết (YYYY, YYYY-MM hoặc YYYY-MM-DD). "
                        "Ghi chép muộn hơn mốc này sẽ bị loại, để cảnh không nhắc "
                        "tới điều chưa xảy ra."
                    ),
                },
                "limit": {"type": "integer", "description": "Tối đa 10. Mặc định 6."},
            },
            "required": ["query"],
        },
    },
    {
        "name": "history_timeline",
        "description": (
            "Liệt kê các ghi chép đã thẩm định trong một khoảng thời gian, theo "
            "thứ tự niên đại. Dùng khi cần biết bối cảnh quanh thời điểm của "
            "chương. Chỉ trả về ghi chép có nguồn."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "start": {"type": "string", "description": "Mốc đầu (YYYY hoặc YYYY-MM-DD)."},
                "end": {"type": "string", "description": "Mốc cuối."},
            },
            "required": ["start", "end"],
        },
    },
]


def _render(hits: List[Dict[str, Any]]) -> str:
    """
    Render results for the model.

    Divergences are called out loudly. If the model reads "Quang Trung băng hà
    1792" without seeing that this novel departs there, it will faithfully
    write the emperor's death into a book whose premise is that he lived.
    """
    if not hits:
        return (u"KHÔNG CÓ GHI CHÉP NÀO ĐÃ THẨM ĐỊNH KHỚP VỚI TRUY VẤN NÀY.\n"
                u"Đừng suy đoán. Hãy viết tránh né chi tiết lịch sử cụ thể, "
                u"hoặc nói rõ trong chương rằng chi tiết đó không được ghi lại.")

    lines = []
    for h in hits:
        head = u"● [%s] %s — %s" % (h["id"], h["date"], h["claim"])
        if h["confidence"] != "attested":
            head += u"  (mức độ: %s)" % h["confidence"]
        lines.append(head)
        lines.append(u"   nguồn: %s" % "; ".join(h["sources"]))
        if h.get("note"):
            lines.append(u"   ghi chú: %s" % h["note"])
        if h.get("diverges"):
            lines.append(u"   ⚠ TRUYỆN NÀY RẼ NHÁNH TẠI ĐÂY: %s" % h["divergence_note"])
            lines.append(u"   → Hãy viết theo hướng rẽ nhánh, KHÔNG theo sử liệu gốc.")
        elif h.get("locked"):
            lines.append(u"   → Ghi chép khóa: chương không được mâu thuẫn với điều này.")
        lines.append("")
    return "\n".join(lines).rstrip()


def dispatch(name: str, arguments: Dict[str, Any], novel=None) -> str:
    """
    Run a tool call and render its result as text for the model.

    Every path returns corpus content or an explicit "nothing found" — there is
    no branch that produces a claim the corpus does not contain.
    """
    index = get_index(novel)
    try:
        if name == "search_history":
            query = str(arguments.get("query") or "").strip()
            if not query:
                return u"Thiếu tham số 'query'."
            limit = arguments.get("limit") or 6
            try:
                limit = max(1, min(int(limit), 10))
            except (TypeError, ValueError):
                limit = 6
            hits = index.search(query, limit=limit, before=arguments.get("before") or None)
            return _render(hits)

        if name == "history_timeline":
            hits = index.timeline(arguments.get("start"), arguments.get("end"))
            return _render(hits)

    except HistoryError as exc:
        return u"Tham số không hợp lệ: %s" % exc

    return u"Không có công cụ tên %r." % name
