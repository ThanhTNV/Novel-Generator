"""
LLM API Client: unified interface for Claude, OpenAI, Groq, and Ollama.
"""

import json
from typing import AsyncIterator, Optional

import httpx

from backend.config import settings


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
