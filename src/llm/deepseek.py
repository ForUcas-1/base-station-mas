"""DeepSeek API client (OpenAI-compatible endpoint)."""

import os

from openai import AsyncOpenAI


class DeepSeekClient:
    """DeepSeek API via OpenAI SDK with custom base_url.

    DeepSeek's API is OpenAI-compatible at https://api.deepseek.com/v1.
    Used via the same interface as AnthropicClient / OpenAIClient.
    """

    BASE_URL = "https://api.deepseek.com/v1"

    def __init__(
        self,
        model: str = "deepseek-v4-pro",
        api_key: str | None = None,
    ):
        self.model = model
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY", "")
        if not self.api_key:
            raise ValueError(
                "DEEPSEEK_API_KEY is required. "
                "Set it in .env or pass as api_key."
            )
        self._client = None

    @property
    def client(self) -> AsyncOpenAI:
        if self._client is None:
            self._client = AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.BASE_URL,
            )
        return self._client

    async def complete(
        self,
        prompt: str,
        system: str = "",
        response_format: str = "text",
        max_tokens: int = 4096,
        temperature: float = 0.3,
    ) -> str:
        """Send a completion request to DeepSeek."""
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

        if response_format == "json":
            kwargs["response_format"] = {"type": "json_object"}

        response = await self.client.chat.completions.create(**kwargs)
        return response.choices[0].message.content or ""

    async def chat(self, messages: list[dict]) -> str:
        """Multi-turn chat via DeepSeek."""
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=4096,
        )
        return response.choices[0].message.content or ""

    async def close(self):
        """Close the underlying HTTP client — aborts in-flight requests."""
        if self._client is not None:
            await self._client.close()
            self._client = None
