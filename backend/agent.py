"""
Novel Agent: assembles prompts from context, skills, rules, and Zero-Mem
evidence, then calls the LLM to generate or revise chapter drafts.

The only LLM calls in this file are the ones that write prose. Memory
operations (store / index / retrieve) are zero-token by construction.
"""

from pathlib import Path
from typing import AsyncIterator, Dict, List, Optional, Union

from backend.config import settings
from backend.api_client import LLMClient
from backend.rag_pipeline import get_engine


def _load_markdown_dir(dirpath: Union[str, Path]) -> str:
    """Concatenate all .md files in a directory into a single string."""
    dirpath = Path(dirpath)
    if not dirpath.exists():
        return ""
    parts = []
    for f in sorted(dirpath.glob("*.md")):
        parts.append(f.read_text(encoding="utf-8"))
    return "\n\n---\n\n".join(parts)


def _load_file(filepath: Union[str, Path]) -> str:
    path = Path(filepath)
    if path.exists():
        return path.read_text(encoding="utf-8")
    return ""


def load_rules() -> str:
    return _load_markdown_dir(settings.rules_dir)


def load_skills() -> str:
    return _load_markdown_dir(settings.skills_dir)


def load_base_prompt() -> str:
    return _load_file(Path(settings.prompts_dir) / "base-prompt.md")


def load_chapter_template() -> str:
    return _load_file(Path(settings.prompts_dir) / "chapter-prompt.md")


def _estimate_tokens(text: str) -> int:
    """Rough token estimate: ~1 token per 4 chars for English, ~1 per 2 for Vietnamese."""
    return max(len(text) // 3, len(text.split()))


def output_budget(target_words: Optional[int] = None) -> int:
    """
    Output-token budget for a piece of prose of ``target_words``.

    Vietnamese runs ~2.26 tokens/word (measured against cl100k on this
    project's corpus), so a 2000-word chapter needs ~4.5k tokens — more than
    the 4096 that used to be hard-coded here, which silently truncated every
    chapter written at the default length.
    """
    words = target_words or settings.default_target_words
    needed = int(words * settings.output_tokens_per_word * settings.output_token_headroom)
    return max(settings.min_output_tokens, min(needed, settings.max_output_tokens))


def _sources_from_used(used: List[Dict]) -> List[Dict]:
    """Shape Zero-Mem provenance for the /api/chat sources panel."""
    return [
        {
            "text": u["text"],
            "source": "%s%s" % (
                u["source"],
                " — " + u["heading"] if u.get("heading") else "",
            ),
        }
        for u in used
    ]


class NovelAgent:
    """Orchestrates RAG retrieval, rule/skill injection, and LLM generation."""

    def __init__(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
    ):
        self.llm = LLMClient(provider=provider, model=model)

    def build_system_prompt(self) -> str:
        base = load_base_prompt()
        rules = load_rules()
        skills = load_skills()

        sections = [base]
        if rules:
            sections.append(f"## Active Rules\n\n{rules}")
        if skills:
            sections.append(f"## Skill Guidance\n\n{skills}")
        return "\n\n---\n\n".join(sections)

    @staticmethod
    def build_revise_system_prompt() -> str:
        return (
            "You are an expert novelist revising a draft. "
            "Maintain consistency with all established characters, settings, and plot. "
            "Follow the revision instructions precisely. Return the complete revised chapter."
        )

    def build_chapter_prompt(
        self,
        chapter_instructions: str,
        retrieved_context: str,
        story_summary: str = "",
        target_words: Optional[int] = None,
    ) -> str:
        target = target_words or settings.default_target_words
        template = load_chapter_template()

        prompt = template
        prompt = prompt.replace("{{ retrieved_chunks }}", retrieved_context)
        prompt = prompt.replace("{{ story_summary }}", story_summary or "(Start of story)")
        prompt = prompt.replace("{{ chapter_instructions }}", chapter_instructions)
        prompt = prompt.replace("{{ target_words }}", str(target))

        return prompt

    async def gather_context(
        self,
        query: str,
        characters: Optional[List[str]] = None,
        locations: Optional[List[str]] = None,
        top_k: Optional[int] = None,
    ) -> str:
        """
        Retrieve chapter-writing context from Zero-Mem.

        One retrieval, not four: naming the characters/locations in the query
        seeds the entity graph directly, and evidence closure brings in their
        surrounding prose. (The old pipeline additionally fired a hard-coded
        English "unresolved conflict tension mystery" query at this Vietnamese
        corpus on every call.)
        """
        engine = get_engine()
        parts = [query]
        if characters:
            parts.append(" ".join(characters))
        if locations:
            parts.append(" ".join(locations))
        result = engine.build_context(
            query=" \n".join(parts),
            max_tokens=settings.max_context_tokens,
            top_k=max(top_k or settings.default_top_k, 8),
        )
        if not result["context"]:
            return "(No relevant context stored yet.)"
        return result["context"]

    async def generate_chapter(
        self,
        chapter_instructions: str,
        story_summary: str = "",
        characters: Optional[List[str]] = None,
        locations: Optional[List[str]] = None,
        target_words: Optional[int] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Full pipeline: retrieve context -> assemble prompt -> generate chapter."""
        max_tokens = max_tokens or output_budget(target_words)
        context = await self.gather_context(
            query=chapter_instructions,
            characters=characters,
            locations=locations,
        )

        system = self.build_system_prompt()
        user = self.build_chapter_prompt(
            chapter_instructions=chapter_instructions,
            retrieved_context=context,
            story_summary=story_summary,
            target_words=target_words,
        )

        return await self.llm.generate(
            system_prompt=system,
            user_prompt=user,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    async def generate_chapter_stream(
        self,
        chapter_instructions: str,
        story_summary: str = "",
        characters: Optional[List[str]] = None,
        locations: Optional[List[str]] = None,
        target_words: Optional[int] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> AsyncIterator[str]:
        """Streaming version of generate_chapter."""
        max_tokens = max_tokens or output_budget(target_words)
        context = await self.gather_context(
            query=chapter_instructions,
            characters=characters,
            locations=locations,
        )

        system = self.build_system_prompt()
        user = self.build_chapter_prompt(
            chapter_instructions=chapter_instructions,
            retrieved_context=context,
            story_summary=story_summary,
            target_words=target_words,
        )

        async for token in self.llm.generate_stream(
            system_prompt=system,
            user_prompt=user,
            temperature=temperature,
            max_tokens=max_tokens,
        ):
            yield token

    @staticmethod
    def build_chat_system_prompt() -> str:
        return (
            "You are a knowledgeable assistant for a novel project. "
            "Answer questions about the story's characters, plot, world-building, "
            "and writing using ONLY the retrieved context provided. "
            "If the context doesn't contain the answer, say so honestly. "
            "Reply in the same language as the user's question."
        )

    def _prepare_chat(
        self,
        message: str,
        history: Optional[List[Dict]],
        top_k: Optional[int],
    ) -> tuple:
        """Shared retrieval + prompt assembly for chat / chat_stream."""
        engine = get_engine()

        # Recent conversation turns often hold the referent of "cậu ấy/him/it";
        # folding them into the retrieval query keeps follow-ups grounded.
        recent_user = " ".join(
            t.get("content", "") for t in (history or [])[-4:] if t.get("role") == "user"
        )
        result = engine.build_context(
            query=message if not recent_user else "%s\n%s" % (recent_user, message),
            max_tokens=settings.max_context_tokens,
            top_k=max(top_k or settings.default_top_k, 6),
        )
        context = result["context"] or "(No relevant context stored yet.)"
        sources = _sources_from_used(result["used"])

        conv_parts = []
        if history:
            for turn in history[-6:]:
                role = turn.get("role", "user")
                conv_parts.append(f"{'User' if role == 'user' else 'Assistant'}: {turn['content']}")

        user_prompt = f"## Retrieved Context\n\n{context}\n\n"
        if conv_parts:
            user_prompt += "## Conversation History\n\n" + "\n".join(conv_parts) + "\n\n"
        user_prompt += f"## Question\n\n{message}"

        return self.build_chat_system_prompt(), user_prompt, sources

    async def chat(
        self,
        message: str,
        history: Optional[List[Dict]] = None,
        top_k: Optional[int] = None,
    ) -> Dict:
        """Q&A over the story memory: structured evidence selection, then answer."""
        system, user_prompt, sources = self._prepare_chat(message, history, top_k)
        answer = await self.llm.generate(
            system_prompt=system,
            user_prompt=user_prompt,
            temperature=0.3,
            max_tokens=2048,
        )
        return {"answer": answer, "sources": sources}

    async def chat_stream(
        self,
        message: str,
        history: Optional[List[Dict]] = None,
        top_k: Optional[int] = None,
    ) -> tuple:
        """Streaming Q&A. Returns (async_iterator, sources)."""
        system, user_prompt, sources = self._prepare_chat(message, history, top_k)
        stream = self.llm.generate_stream(
            system_prompt=system,
            user_prompt=user_prompt,
            temperature=0.3,
            max_tokens=2048,
        )
        return stream, sources

    @staticmethod
    def _fit_draft_to_budget(draft: str, budget_tokens: int) -> str:
        """Keep beginning + end of draft if it exceeds the token budget."""
        est = _estimate_tokens(draft)
        if est <= budget_tokens:
            return draft
        words = draft.split()
        keep = int(budget_tokens * 0.75)
        half = keep // 2
        head = " ".join(words[:half])
        tail = " ".join(words[-half:])
        return f"{head}\n\n[... middle truncated for token budget ...]\n\n{tail}"

    def _build_revise_user_prompt(self, draft: str, feedback: str) -> str:
        system_est = _estimate_tokens(self.build_revise_system_prompt())
        feedback_est = _estimate_tokens(feedback)
        overhead = 120
        draft_budget = settings.max_context_tokens * 2 - system_est - feedback_est - overhead
        draft_budget = max(draft_budget, 1000)

        trimmed = self._fit_draft_to_budget(draft, draft_budget)
        return (
            "## Current Draft\n\n"
            f"{trimmed}\n\n"
            "## Revision Instructions\n\n"
            f"{feedback}\n\n"
            "## Task\n\n"
            "Revise the draft above according to the instructions. "
            "Return the complete revised chapter."
        )

    async def revise_chapter(
        self,
        draft: str,
        feedback: str,
        temperature: float = 0.5,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Revise a draft based on user feedback."""
        # A revision returns the whole chapter, so the budget scales with the
        # draft it is rewriting, not with a fixed constant.
        max_tokens = max_tokens or output_budget(len(draft.split()))
        system = self.build_revise_system_prompt()
        user = self._build_revise_user_prompt(draft, feedback)

        return await self.llm.generate(
            system_prompt=system,
            user_prompt=user,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    async def revise_chapter_stream(
        self,
        draft: str,
        feedback: str,
        temperature: float = 0.5,
        max_tokens: Optional[int] = None,
    ) -> AsyncIterator[str]:
        """Streaming revision."""
        max_tokens = max_tokens or output_budget(len(draft.split()))
        system = self.build_revise_system_prompt()
        user = self._build_revise_user_prompt(draft, feedback)

        async for token in self.llm.generate_stream(
            system_prompt=system,
            user_prompt=user,
            temperature=temperature,
            max_tokens=max_tokens,
        ):
            yield token
