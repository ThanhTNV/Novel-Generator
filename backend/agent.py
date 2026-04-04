"""
Novel Agent: assembles prompts from context, skills, rules, and retrieved RAG chunks,
then calls the LLM to generate or revise chapter drafts.
"""

from pathlib import Path
from typing import AsyncIterator, Dict, List, Optional, Union

from backend.config import settings
from backend.api_client import LLMClient
from backend.rag_pipeline import (
    retrieve,
    retrieve_for_characters,
    retrieve_for_locations,
    retrieve_for_plot,
)


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


def _format_hits(hits: List[Dict]) -> str:
    if not hits:
        return "(No relevant context found in vector store.)"
    parts = []
    for i, h in enumerate(hits, 1):
        meta = h.get("metadata", {})
        source = meta.get("source", "unknown")
        parts.append(f"[{i}] (source: {source})\n{h['text']}")
    return "\n\n".join(parts)


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
        prompt = prompt.replace("{{ rules }}", load_rules())
        prompt = prompt.replace("{{ skills }}", load_skills())
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
        """Retrieve relevant context from the vector store using skills-based queries."""
        k = top_k or settings.default_top_k
        all_hits: List[Dict] = []

        general_hits = retrieve(query, top_k=k)
        all_hits.extend(general_hits)

        if characters:
            char_hits = retrieve_for_characters(characters, top_k=min(k, 5))
            all_hits.extend(char_hits)

        if locations:
            loc_hits = retrieve_for_locations(locations, top_k=min(k, 5))
            all_hits.extend(loc_hits)

        plot_hits = retrieve_for_plot(top_k=min(k, 5))
        all_hits.extend(plot_hits)

        seen = set()
        deduped = []
        for h in all_hits:
            key = h["text"][:100]
            if key not in seen:
                seen.add(key)
                deduped.append(h)

        return _format_hits(deduped)

    async def generate_chapter(
        self,
        chapter_instructions: str,
        story_summary: str = "",
        characters: Optional[List[str]] = None,
        locations: Optional[List[str]] = None,
        target_words: Optional[int] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> str:
        """Full pipeline: retrieve context -> assemble prompt -> generate chapter."""
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
        max_tokens: int = 4096,
    ) -> AsyncIterator[str]:
        """Streaming version of generate_chapter."""
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

    async def revise_chapter(
        self,
        draft: str,
        feedback: str,
        temperature: float = 0.5,
        max_tokens: int = 4096,
    ) -> str:
        """Revise a draft based on user feedback."""
        system = self.build_system_prompt()

        user = (
            "## Current Draft\n\n"
            f"{draft}\n\n"
            "## Revision Instructions\n\n"
            f"{feedback}\n\n"
            "## Task\n\n"
            "Revise the draft above according to the feedback. "
            "Maintain all established character, plot, and world details. "
            "Apply all active rules and skill guidance. "
            "Return the complete revised chapter."
        )

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
        max_tokens: int = 4096,
    ) -> AsyncIterator[str]:
        """Streaming revision."""
        system = self.build_system_prompt()

        user = (
            "## Current Draft\n\n"
            f"{draft}\n\n"
            "## Revision Instructions\n\n"
            f"{feedback}\n\n"
            "## Task\n\n"
            "Revise the draft above according to the feedback. "
            "Maintain all established character, plot, and world details. "
            "Apply all active rules and skill guidance. "
            "Return the complete revised chapter."
        )

        async for token in self.llm.generate_stream(
            system_prompt=system,
            user_prompt=user,
            temperature=temperature,
            max_tokens=max_tokens,
        ):
            yield token
