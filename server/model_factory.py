"""
ModelFactory - builds a LangChain chat model from the project config.

The factory reads the `[llm]` section of `config.toml` and instantiates the
appropriate provider (Ollama or Gemini). Provider-specific settings live under
`[llm.ollama]` and `[llm.gemini]` respectively.
"""

from __future__ import annotations

import os
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel

from config import load_config


class ModelFactory:
    """Builds a `BaseChatModel` based on the project configuration."""

    SUPPORTED_PROVIDERS = ("ollama", "gemini")

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or load_config()

    def create(self) -> BaseChatModel:
        llm_cfg = self.config.get("llm", {})
        provider = llm_cfg.get("provider", "ollama")
        temperature = float(llm_cfg.get("temperature", 0.7))

        if provider == "ollama":
            return self._build_ollama(llm_cfg, temperature)
        if provider == "gemini":
            return self._build_gemini(llm_cfg, temperature)

        raise ValueError(
            f"Unsupported LLM provider: {provider!r}. "
            f"Expected one of {self.SUPPORTED_PROVIDERS}."
        )

    def _build_ollama(self, llm_cfg: dict[str, Any], temperature: float) -> BaseChatModel:
        from langchain_ollama import ChatOllama

        ollama_cfg = llm_cfg.get("ollama", {})
        return ChatOllama(
            model=ollama_cfg.get("model", "llama3.2"),
            temperature=temperature,
            base_url=ollama_cfg.get("base_url", "http://localhost:11434"),
        )

    def _build_gemini(self, llm_cfg: dict[str, Any], temperature: float) -> BaseChatModel:
        from langchain_google_genai import ChatGoogleGenerativeAI

        gemini_cfg = llm_cfg.get("gemini", {})
        api_key = gemini_cfg.get("api_key") or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "Gemini API key not provided. Set GEMINI_API_KEY env var or "
                "[llm.gemini].api_key in config.toml."
            )
        return ChatGoogleGenerativeAI(
            model=gemini_cfg.get("model", "gemini-1.5-flash"),
            temperature=temperature,
            google_api_key=api_key,
        )


def get_model() -> BaseChatModel:
    """Convenience helper: build a model from the default config."""
    return ModelFactory().create()