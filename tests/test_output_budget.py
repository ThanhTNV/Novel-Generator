# -*- coding: utf-8 -*-
"""
Output-budget regression tests.

The app targeted 2000-word chapters while hard-coding max_tokens=4096.
Vietnamese prose measures ~2.26 tokens/word (cl100k) on this corpus, so a
default-length chapter needed ~4.5k tokens and was silently truncated
mid-sentence on every generation. The budget is now derived from the
requested length.
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.agent import output_budget  # noqa: E402
from backend.config import settings  # noqa: E402
from backend.zero_mem.text import estimate_tokens  # noqa: E402

# Real Vietnamese narrative prose, of the kind this project generates.
VI_PROSE = (
    "Sương xám phủ kín sân trường khi hoàng hôn buông xuống. Văn Tâm đứng "
    "lặng trước cánh cửa phòng thí nghiệm cũ, tay siết chặt quai balo. "
    "Cậu biết rõ mình không nên vào, nhưng Đồ Lục trong túi áo cứ nóng dần "
    "lên như một lời thúc giục. Nguyên Khang huých vai cậu, giọng khẽ khàng: "
    "\"Ông chắc chứ? Thầy Cao mất ở đây đấy.\" Tâm không trả lời."
)


class TestVietnameseTokenRatio:
    def test_prose_costs_more_than_two_tokens_per_word(self):
        words = len(VI_PROSE.split())
        assert estimate_tokens(VI_PROSE) / float(words) > 1.8

    @pytest.mark.skipif(
        pytest.importorskip("tiktoken", reason="tiktoken not installed") is None,
        reason="tiktoken not installed",
    )
    def test_configured_ratio_covers_real_tokenizer(self):
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        actual = len(enc.encode(VI_PROSE)) / float(len(VI_PROSE.split()))
        # The configured rate must not under-estimate real Vietnamese cost.
        assert settings.output_tokens_per_word >= actual


class TestOutputBudget:
    def test_default_chapter_is_not_truncated(self):
        """The exact regression: 2000 words used to be capped at 4096."""
        budget = output_budget(settings.default_target_words)
        needed = settings.default_target_words * 2.26  # measured cl100k rate
        assert budget > 4096
        assert budget >= needed

    def test_scales_with_target_length(self):
        assert output_budget(1000) < output_budget(2000) < output_budget(4000)

    def test_respects_floor_and_ceiling(self):
        assert output_budget(1) >= settings.min_output_tokens
        assert output_budget(1_000_000) == settings.max_output_tokens

    def test_none_falls_back_to_configured_default(self):
        assert output_budget(None) == output_budget(settings.default_target_words)


class TestCallSitesUseDerivedBudget:
    """max_tokens must be optional end-to-end, or the default leaks back in."""

    def test_agent_signatures_default_to_none(self):
        import inspect
        from backend.agent import NovelAgent
        for name in ("generate_chapter", "generate_chapter_stream",
                     "revise_chapter", "revise_chapter_stream"):
            sig = inspect.signature(getattr(NovelAgent, name))
            assert sig.parameters["max_tokens"].default is None, name

    def test_request_models_default_to_none(self):
        from backend.server import GenerateRequest, ReviseRequest
        assert GenerateRequest(chapter_instructions="x").max_tokens is None
        assert ReviseRequest(draft="x", feedback="y").max_tokens is None

    def test_frontend_does_not_pin_max_tokens(self):
        js = (ROOT / "frontend" / "static" / "app.js").read_text(encoding="utf-8")
        assert "max_tokens: 4096" not in js
