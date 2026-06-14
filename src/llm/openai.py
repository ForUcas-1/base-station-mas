"""OpenAI GPT API client."""

import json
import os

from src.llm.client import LLMClient


class OpenAIClient(LLMClient):
    """GPT API via OpenAI Python SDK."""

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: str | None = None,
    ):
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        if not self.api_key:
            raise ValueError(
                "OPENAI_API_KEY is required. "
                "Set it in .env or pass as api_key."
            )
        self._client = None

    @property
    def client(self):
        """Lazy-load the OpenAI async client."""
        if self._client is None:
            from openai import AsyncOpenAI
            self._client = AsyncOpenAI(api_key=self.api_key)
        return self._client

    async def complete(
        self,
        prompt: str,
        system: str = "",
        response_format: str = "text",
        max_tokens: int = 4096,
        temperature: float = 0.3,
    ) -> str:
        """Send a completion request to GPT."""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        kwargs = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        # OpenAI supports native JSON mode
        if response_format == "json":
            kwargs["response_format"] = {"type": "json_object"}

        response = await self.client.chat.completions.create(**kwargs)
        return response.choices[0].message.content or ""

    async def chat(self, messages: list[dict]) -> str:
        """Multi-turn chat via GPT."""
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=4096,
        )
        return response.choices[0].message.content or ""

    async def close(self):
        """Close the underlying HTTP client."""
        if self._client is not None:
            await self._client.close()
            self._client = None
