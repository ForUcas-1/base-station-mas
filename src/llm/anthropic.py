"""Anthropic Claude API client."""

import os

from src.llm.client import LLMClient


class AnthropicClient(LLMClient):
    """Claude API via Anthropic Python SDK."""

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        api_key: str | None = None,
    ):
        self.model = model
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        if not self.api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY is required. "
                "Set it in .env or pass as api_key."
            )
        self._client = None

    @property
    def client(self):
        """Lazy-load the Anthropic async client."""
        if self._client is None:
            import anthropic
            self._client = anthropic.AsyncAnthropic(api_key=self.api_key)
        return self._client

    async def complete(
        self,
        prompt: str,
        system: str = "",
        response_format: str = "text",
        max_tokens: int = 4096,
        temperature: float = 0.3,
    ) -> str:
        """Send a completion request to Claude."""
        kwargs = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        }

        if system:
            kwargs["system"] = system

        response = await self.client.messages.create(**kwargs)

        # response.content is a list of ContentBlock
        text = "".join(
            block.text for block in response.content
            if hasattr(block, "text")
        )

        # Strip ```json fences if present
        if response_format == "json":
            text = text.strip()
            if text.startswith("```"):
                lines = text.split("\n")
                text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

        return text

    async def chat(self, messages: list[dict]) -> str:
        """Multi-turn chat via Claude."""
        response = await self.client.messages.create(
            model=self.model,
            max_tokens=4096,
            messages=messages,
        )
        return "".join(
            block.text for block in response.content
            if hasattr(block, "text")
        )

    async def close(self):
        """Close the underlying HTTP client."""
        if self._client is not None:
            await self._client.close()
            self._client = None
