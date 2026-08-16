"""
Novel Agent: assembles prompts from context, skills, rules, and Zero-Mem
evidence, then calls the LLM to generate or revise chapter drafts.

The only LLM calls in this file are the ones that write prose. Memory
operations (store / index / retrieve) are zero-token by construction.
"""

from pathlib import Path
from typing import AsyncIterator, Dict, List, Optional, Union

from backend import history
from backend.config import settings
from backend.api_client import LLMClient
from backend.rag_pipeline import get_engine

# Told to the model during the research phase only. It is deliberately blunt
# about the one failure mode that matters: a model that cannot find a fact will
# otherwise supply a plausible one, and plausible invented history is exactly
# what this whole feature exists to prevent.
_RESEARCH_SYSTEM = (
    "Bạn đang chuẩn bị tư liệu lịch sử cho một chương tiểu thuyết dã sử. "
    "Hãy dùng công cụ tra cứu để lấy những ghi chép đã thẩm định liên quan tới "
    "nhân vật, sự kiện và mốc thời gian của chương sắp viết. "
    "Gọi công cụ nhiều lần nếu cần, mỗi lần một chủ đề hẹp.\n\n"
    "Quy tắc tuyệt đối: bạn KHÔNG được nêu bất kỳ sự kiện, niên đại hay nhân "
    "vật lịch sử nào mà công cụ chưa trả về. Nếu công cụ không có ghi chép, "
    "hãy chấp nhận là không có cứ liệu. Đừng viết văn ở bước này — chỉ tra cứu."
)


def _collect_markdown(*dirs: Optional[Union[str, Path]]) -> str:
    """
    Concatenate .md files across directories, later ones winning by filename.

    "Write in third person past tense" is usually a house style, so rules and
    skills keep project-level defaults. A novel that needs a different `tone.md`
    drops its own into `novels/<slug>/rules/`, and it replaces the shared one
    rather than being appended alongside it — two contradictory tone rules in
    the same system prompt is worse than either alone.
    """
    by_name: Dict[str, str] = {}
    for dirpath in dirs:
        if not dirpath:
            continue
        directory = Path(dirpath)
        if not directory.is_dir():
            continue
        for f in sorted(directory.glob("*.md")):
            by_name[f.stem] = f.read_text(encoding="utf-8")
    return "\n\n---\n\n".join(by_name[k] for k in sorted(by_name))


def _load_file(filepath: Union[str, Path]) -> str:
    path = Path(filepath)
    if path.exists():
        return path.read_text(encoding="utf-8")
    return ""


def load_rules(novel=None) -> str:
    return _collect_markdown(
        settings.rules_dir, novel.rules_dir if novel is not None else None
    )


def load_skills(novel=None) -> str:
    return _collect_markdown(
        settings.skills_dir, novel.skills_dir if novel is not None else None
    )


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
        novel=None,
    ):
        self.llm = LLMClient(provider=provider, model=model)
        # Every retrieval this agent performs is scoped to this novel's engine.
        # There is no cross-novel path: a different novel is a different store.
        self.novel = novel

    def build_system_prompt(self) -> str:
        base = load_base_prompt()
        rules = load_rules(self.novel)
        skills = load_skills(self.novel)

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
        history_reference: str = "",
    ) -> str:
        target = target_words or settings.default_target_words
        template = load_chapter_template()

        prompt = template
        prompt = prompt.replace("{{ retrieved_chunks }}", retrieved_context)
        prompt = prompt.replace("{{ story_summary }}", story_summary or "(Start of story)")
        prompt = prompt.replace("{{ chapter_instructions }}", chapter_instructions)
        prompt = prompt.replace("{{ target_words }}", str(target))
        # Templates predating the history corpus have no placeholder; append
        # rather than lose the references entirely.
        if "{{ history_reference }}" in prompt:
            prompt = prompt.replace("{{ history_reference }}", history_reference)
        elif history_reference:
            prompt = "%s\n\n%s" % (history_reference, prompt)

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
        engine = get_engine(self.novel)
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

    # -- historical research -------------------------------------------------

    async def research_history(
        self,
        chapter_instructions: str,
        scene_date: Optional[str] = None,
        characters: Optional[List[str]] = None,
    ) -> Dict:
        """
        Let the model look up vetted history before it writes.

        Research and writing are separate calls on purpose. The model may query
        freely here, but only the tool's own output survives into the writing
        prompt — its commentary is dropped, so nothing unsourced can ride along
        into the chapter as if it were a record.

        Never raises: a corpus that is empty, missing or misconfigured must
        degrade to writing without references, not block the chapter.
        """
        empty = {"reference": "", "calls": [], "records": [], "available": 0}
        if not settings.history_tool_enabled:
            return empty
        try:
            index = history.get_index(self.novel)
        except history.HistoryError as exc:
            self._log("history: corpus rejected (%s); writing without references." % exc)
            return empty
        if not len(index):
            return empty

        brief = [u"Chương sắp viết:", chapter_instructions]
        if scene_date:
            brief.append(u"\nThời điểm của cảnh: %s. Hãy truyền tham số 'before' "
                         u"bằng mốc này để loại bỏ những gì chưa xảy ra." % scene_date)
        if characters:
            brief.append(u"\nNhân vật xuất hiện: %s" % ", ".join(characters))

        seen: Dict[str, Dict] = {}

        def run_tool(name: str, arguments: Dict) -> str:
            if scene_date and name == "search_history":
                arguments.setdefault("before", scene_date)
            # Record which records were actually served, so the writing prompt
            # and the UI cite exactly what the model was shown.
            try:
                if name == "search_history":
                    for hit in index.search(
                        str(arguments.get("query") or ""),
                        limit=min(int(arguments.get("limit") or 6), 10),
                        before=arguments.get("before") or None,
                    ):
                        seen[hit["id"]] = hit
                elif name == "history_timeline":
                    for hit in index.timeline(arguments.get("start"), arguments.get("end")):
                        seen[hit["id"]] = hit
            except history.HistoryError:
                pass
            return history.dispatch(name, arguments, novel=self.novel)

        try:
            calls = await self.llm.run_tools(
                system_prompt=_RESEARCH_SYSTEM,
                user_prompt="\n".join(brief),
                tools=history.TOOLS,
                run_tool=run_tool,
                max_rounds=settings.history_max_tool_calls,
            )
        except Exception as exc:
            # A provider without tool support, a network blip, a bad key: the
            # chapter still gets written, just without references.
            self._log("history: research skipped (%s)." % exc)
            return empty

        records = [seen[k] for k in sorted(seen)]
        return {
            "reference": self._render_reference(records),
            "calls": [{"tool": n, "arguments": a} for n, a, _out in calls],
            "records": records,
            "available": len(index),
        }

    @staticmethod
    def _log(message: str) -> None:
        print(message)

    @staticmethod
    def _render_reference(records: List[Dict]) -> str:
        """The block injected into the writing prompt. Records only, verbatim."""
        if not records:
            return ""
        lines = [u"## Sử liệu đã thẩm định (chỉ dùng những gì có ở đây)", ""]
        for r in records:
            lines.append(u"- **%s** (%s) — %s" % (r["claim"], r["date"], "; ".join(r["sources"])))
            if r.get("diverges"):
                lines.append(u"  - ⚠ Truyện rẽ nhánh tại đây: %s" % r["divergence_note"])
                lines.append(u"  - Viết theo hướng rẽ nhánh, không theo sử liệu gốc.")
            elif r.get("locked"):
                lines.append(u"  - Không được mâu thuẫn với ghi chép này.")
        lines.append("")
        lines.append(u"Ngoài danh sách trên, đừng khẳng định thêm bất kỳ sự kiện, "
                     u"niên đại hay nhân vật lịch sử nào. Nếu cần một chi tiết "
                     u"không có ở đây, hãy viết sao cho không phải nêu nó ra.")
        return "\n".join(lines)

    async def generate_chapter(
        self,
        chapter_instructions: str,
        story_summary: str = "",
        characters: Optional[List[str]] = None,
        locations: Optional[List[str]] = None,
        target_words: Optional[int] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        scene_date: Optional[str] = None,
    ) -> Dict:
        """
        Full pipeline: research history -> retrieve context -> write -> check.

        Returns the draft together with the records it was given and any
        conflicts found in it, so the caller can show both without a second
        pass over the text.
        """
        max_tokens = max_tokens or output_budget(target_words)
        research = await self.research_history(chapter_instructions, scene_date, characters)
        context = await self.gather_context(
            query=chapter_instructions,
            characters=characters,
            locations=locations,
        )

        text = await self.llm.generate(
            system_prompt=self.build_system_prompt(),
            user_prompt=self.build_chapter_prompt(
                chapter_instructions=chapter_instructions,
                retrieved_context=context,
                story_summary=story_summary,
                target_words=target_words,
                history_reference=research["reference"],
            ),
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return {
            "content": text,
            "history": {
                "records": research["records"],
                "calls": research["calls"],
                "check": self.check_history(text, scene_date),
            },
        }

    def check_history(self, text: str, scene_date: Optional[str] = None) -> Dict:
        """Check a draft against the corpus. Never raises; never edits the text."""
        if not settings.history_check_drafts:
            return {"findings": [], "ok": True, "scene_date": scene_date}
        try:
            return history.get_index(self.novel).check(text, scene_date)
        except history.HistoryError as exc:
            return {"findings": [], "ok": True, "scene_date": scene_date,
                    "error": str(exc)}

    async def generate_chapter_stream(
        self,
        chapter_instructions: str,
        story_summary: str = "",
        characters: Optional[List[str]] = None,
        locations: Optional[List[str]] = None,
        target_words: Optional[int] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        scene_date: Optional[str] = None,
        on_research=None,
        on_check=None,
    ) -> AsyncIterator[str]:
        """
        Streaming version.

        Research finishes before the first token, so ``on_research`` fires once
        with the records the model was given; the draft is checked after the
        last token and reported through ``on_check``. Both are callbacks rather
        than yielded values so the token stream stays a stream of prose.
        """
        max_tokens = max_tokens or output_budget(target_words)
        research = await self.research_history(chapter_instructions, scene_date, characters)
        if on_research:
            on_research(research)

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
            history_reference=research["reference"],
        )

        chunks: List[str] = []
        async for token in self.llm.generate_stream(
            system_prompt=system,
            user_prompt=user,
            temperature=temperature,
            max_tokens=max_tokens,
        ):
            chunks.append(token)
            yield token

        if on_check:
            on_check(self.check_history("".join(chunks), scene_date))

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
        engine = get_engine(self.novel)

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
