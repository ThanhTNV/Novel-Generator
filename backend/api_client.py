"""
LLM API Client: unified interface for Claude, OpenAI, Groq, and Ollama.

Tool use lives in ``run_tools`` rather than inside the streaming generators.
Interleaving tool calls with a token stream means reassembling partial JSON
arguments across four vendors' different streaming shapes, for no benefit here:
the model researches first and writes second, so the research can be a plain
non-streaming exchange and the prose stream stays exactly as simple as it was.
"""

import json
from typing import Any, AsyncIterator, Callable, Dict, List, Optional, Tuple

import httpx

from backend.config import settings


class ToolCall(object):
    """One tool invocation requested by a model, in provider-neutral form."""

    __slots__ = ("id", "name", "arguments")

    def __init__(self, id: str, name: str, arguments: Dict[str, Any]):
        self.id = id
        self.name = name
        self.arguments = arguments or {}

    def __repr__(self):
        return "ToolCall(%r, %r)" % (self.name, self.arguments)


def _openai_tools(tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Translate the neutral tool spec into OpenAI/Groq's function shape."""
    return [{
        "type": "function",
        "function": {
            "name": t["name"],
            "description": t["description"],
            "parameters": t["input_schema"],
        },
    } for t in tools]


class LLMClient:
    """Unified async LLM client supporting multiple providers."""

    def __init__(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
    ):
        self.provider = provider or settings.default_llm_provider
        self.model = model or settings.default_model

    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> str:
        """Generate a completion from the configured LLM."""
        dispatch = {
            "claude": self._generate_claude,
            "openai": self._generate_openai,
            "groq": self._generate_groq,
            "ollama": self._generate_ollama,
        }
        fn = dispatch.get(self.provider)
        if fn is None:
            raise ValueError(f"Unknown provider: {self.provider}")
        return await fn(system_prompt, user_prompt, temperature, max_tokens)

    async def generate_stream(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> AsyncIterator[str]:
        """Stream tokens from the configured LLM."""
        dispatch = {
            "claude": self._stream_claude,
            "openai": self._stream_openai,
            "groq": self._stream_groq,
            "ollama": self._stream_ollama,
        }
        fn = dispatch.get(self.provider)
        if fn is None:
            raise ValueError(f"Unknown provider: {self.provider}")
        async for token in fn(system_prompt, user_prompt, temperature, max_tokens):
            yield token

    # ------------------------------------------------------------------
    # Tool use (research phase)
    # ------------------------------------------------------------------

    async def run_tools(
        self,
        system_prompt: str,
        user_prompt: str,
        tools: List[Dict[str, Any]],
        run_tool: Callable[[str, Dict[str, Any]], str],
        max_rounds: int = 6,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> List[Tuple[str, Dict[str, Any], str]]:
        """
        Let the model call ``tools`` until it stops, and return the transcript.

        Returns ``[(tool_name, arguments, result_text), ...]``. The model's own
        prose is discarded on purpose: this phase exists to collect verified
        records, and anything the model *says* about history here would be
        unsourced commentary competing with the records themselves.

        ``max_rounds`` is a hard stop — a confused model that keeps re-querying
        should cost a bounded number of calls, not an unbounded one.
        """
        dispatch = {
            "claude": self._tools_claude,
            "openai": self._tools_openai,
            "groq": self._tools_openai,
            "ollama": self._tools_ollama,
        }
        fn = dispatch.get(self.provider)
        if fn is None:
            raise ValueError("Unknown provider: %s" % self.provider)
        return await fn(system_prompt, user_prompt, tools, run_tool,
                        max_rounds, temperature, max_tokens)

    async def _tools_claude(self, system, user, tools, run_tool, max_rounds, temp, max_tok):
        import anthropic
        client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)
        spec = [{"name": t["name"], "description": t["description"],
                 "input_schema": t["input_schema"]} for t in tools]
        messages = [{"role": "user", "content": user}]
        transcript = []

        for _ in range(max_rounds):
            reply = await client.messages.create(
                model=self.model, max_tokens=max_tok, temperature=temp,
                system=system, tools=spec, messages=messages,
            )
            calls = [b for b in reply.content if getattr(b, "type", None) == "tool_use"]
            if not calls:
                break
            messages.append({"role": "assistant", "content": reply.content})
            results = []
            for block in calls:
                args = dict(block.input or {})
                output = run_tool(block.name, args)
                transcript.append((block.name, args, output))
                results.append({"type": "tool_result", "tool_use_id": block.id,
                                "content": output})
            messages.append({"role": "user", "content": results})
        return transcript

    async def _tools_openai(self, system, user, tools, run_tool, max_rounds, temp, max_tok):
        if self.provider == "groq":
            from groq import AsyncGroq
            client = AsyncGroq(api_key=settings.groq_api_key)
        else:
            from openai import AsyncOpenAI
            client = AsyncOpenAI(api_key=settings.openai_api_key)

        spec = _openai_tools(tools)
        messages = [{"role": "system", "content": system},
                    {"role": "user", "content": user}]
        transcript = []

        for _ in range(max_rounds):
            reply = await client.chat.completions.create(
                model=self.model, temperature=temp, max_tokens=max_tok,
                tools=spec, messages=messages,
            )
            message = reply.choices[0].message
            calls = message.tool_calls or []
            if not calls:
                break
            messages.append({
                "role": "assistant",
                "content": message.content or "",
                "tool_calls": [{"id": c.id, "type": "function",
                                "function": {"name": c.function.name,
                                             "arguments": c.function.arguments}}
                               for c in calls],
            })
            for call in calls:
                try:
                    args = json.loads(call.function.arguments or "{}")
                except ValueError:
                    args = {}
                output = run_tool(call.function.name, args)
                transcript.append((call.function.name, args, output))
                messages.append({"role": "tool", "tool_call_id": call.id,
                                 "content": output})
        return transcript

    async def _tools_ollama(self, system, user, tools, run_tool, max_rounds, temp, max_tok):
        spec = _openai_tools(tools)
        messages = [{"role": "system", "content": system},
                    {"role": "user", "content": user}]
        transcript = []

        async with httpx.AsyncClient(timeout=300) as client:
            for _ in range(max_rounds):
                resp = await client.post(
                    f"{settings.ollama_base_url}/api/chat",
                    json={"model": self.model, "messages": messages, "stream": False,
                          "tools": spec,
                          "options": {"temperature": temp, "num_predict": max_tok}},
                )
                resp.raise_for_status()
                message = resp.json().get("message", {}) or {}
                calls = message.get("tool_calls") or []
                if not calls:
                    break
                messages.append(message)
                for call in calls:
                    fn = call.get("function", {}) or {}
                    args = fn.get("arguments") or {}
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except ValueError:
                            args = {}
                    output = run_tool(fn.get("name", ""), args)
                    transcript.append((fn.get("name", ""), args, output))
                    messages.append({"role": "tool", "content": output})
        return transcript

    # ------------------------------------------------------------------
    # Claude (Anthropic)
    # ------------------------------------------------------------------

    async def _generate_claude(self, system: str, user: str, temp: float, max_tok: int) -> str:
        import anthropic
        client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)
        message = await client.messages.create(
            model=self.model,
            max_tokens=max_tok,
            temperature=temp,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return message.content[0].text

    async def _stream_claude(self, system: str, user: str, temp: float, max_tok: int):
        import anthropic
        client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)
        async with client.messages.stream(
            model=self.model,
            max_tokens=max_tok,
            temperature=temp,
            system=system,
            messages=[{"role": "user", "content": user}],
        ) as stream:
            async for text in stream.text_stream:
                yield text

    # ------------------------------------------------------------------
    # OpenAI
    # ------------------------------------------------------------------

    async def _generate_openai(self, system: str, user: str, temp: float, max_tok: int) -> str:
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=settings.openai_api_key)
        response = await client.chat.completions.create(
            model=self.model,
            temperature=temp,
            max_tokens=max_tok,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return response.choices[0].message.content

    async def _stream_openai(self, system: str, user: str, temp: float, max_tok: int):
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=settings.openai_api_key)
        stream = await client.chat.completions.create(
            model=self.model,
            temperature=temp,
            max_tokens=max_tok,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            stream=True,
        )
        async for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

    # ------------------------------------------------------------------
    # Groq
    # ------------------------------------------------------------------

    async def _generate_groq(self, system: str, user: str, temp: float, max_tok: int) -> str:
        from groq import AsyncGroq
        client = AsyncGroq(api_key=settings.groq_api_key)
        response = await client.chat.completions.create(
            model=self.model,
            temperature=temp,
            max_tokens=max_tok,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return response.choices[0].message.content

    async def _stream_groq(self, system: str, user: str, temp: float, max_tok: int):
        from groq import AsyncGroq
        client = AsyncGroq(api_key=settings.groq_api_key)
        stream = await client.chat.completions.create(
            model=self.model,
            temperature=temp,
            max_tokens=max_tok,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            stream=True,
        )
        async for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

    # ------------------------------------------------------------------
    # Ollama (local)
    # ------------------------------------------------------------------

    async def _generate_ollama(self, system: str, user: str, temp: float, max_tok: int) -> str:
        async with httpx.AsyncClient(timeout=300) as client:
            resp = await client.post(
                f"{settings.ollama_base_url}/api/chat",
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    "stream": False,
                    "options": {"temperature": temp, "num_predict": max_tok},
                },
            )
            resp.raise_for_status()
            return resp.json()["message"]["content"]

    async def _stream_ollama(self, system: str, user: str, temp: float, max_tok: int):
        async with httpx.AsyncClient(timeout=300) as client:
            async with client.stream(
                "POST",
                f"{settings.ollama_base_url}/api/chat",
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    "stream": True,
                    "options": {"temperature": temp, "num_predict": max_tok},
                },
            ) as resp:
                async for line in resp.aiter_lines():
                    if line.strip():
                        data = json.loads(line)
                        content = data.get("message", {}).get("content", "")
                        if content:
                            yield content
