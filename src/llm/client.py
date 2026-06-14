"""Abstract LLM client interface and factory."""

import os
from abc import ABC, abstractmethod


class LLMClient(ABC):
    """Abstract base for LLM API clients (Anthropic / OpenAI)."""

    @abstractmethod
    async def complete(
        self,
        prompt: str,
        system: str = "",
        response_format: str = "text",
        max_tokens: int = 4096,
        temperature: float = 0.3,
    ) -> str:
        """Send a single-turn completion request.

        Args:
            prompt: The user message.
            system: System-level instruction.
            response_format: 'text' or 'json'.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature (0.0 = deterministic).

        Returns:
            The model's response text.
        """
        ...

    @abstractmethod
    async def chat(self, messages: list[dict]) -> str:
        """Multi-turn chat interface.

        Args:
            messages: List of {"role": "...", "content": "..."} dicts.

        Returns:
            The model's response text.
        """
        ...


class LLMClientFactory:
    """Factory for creating LLM clients.

    Provider and model are read from environment (.env) by default.
    Only pass config overrides when a specific worker needs different settings
    (e.g., a different temperature for evaluator vs reporter).
    """

    # Default models per provider (used when LLM_MODEL not set in .env)
    _DEFAULT_MODELS = {
        "anthropic": "claude-sonnet-4-20250514",
        "openai": "gpt-4o",
        "deepseek": "deepseek-v4-pro",
    }

    @staticmethod
    def create(config: dict | None = None) -> LLMClient:
        """Create an LLM client.

        Provider and model are resolved in this order:
          1. config dict (per-worker YAML override, if any)
          2. environment variables from .env (LLM_PROVIDER, LLM_MODEL)
          3. built-in defaults

        Args:
            config: Optional overrides. Usually just pass None.
                    Only set keys you need to override, e.g. {temperature: 0.0}

        Returns:
            LLMClient instance.
        """
        config = config or {}

        # Single source of truth: .env → LLM_PROVIDER, LLM_MODEL
        provider = config.get("provider") or os.environ.get("LLM_PROVIDER", "deepseek")
        model = config.get("model") or os.environ.get("LLM_MODEL") or \
                LLMClientFactory._DEFAULT_MODELS.get(provider, "deepseek-v4-pro")

        provider = provider.lower()

        if provider == "anthropic":
            from src.llm.anthropic import AnthropicClient
            return AnthropicClient(model=model, api_key=config.get("api_key"))

        elif provider == "openai":
            from src.llm.openai import OpenAIClient
            return OpenAIClient(model=model, api_key=config.get("api_key"))

        elif provider == "deepseek":
            from src.llm.deepseek import DeepSeekClient
            return DeepSeekClient(model=model, api_key=config.get("api_key"))

        else:
            raise ValueError(
                f"Unknown LLM provider '{provider}'. "
                f"Available: anthropic, openai, deepseek"
            )
