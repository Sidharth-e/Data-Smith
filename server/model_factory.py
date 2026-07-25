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

    SUPPORTED_PROVIDERS = ("ollama", "ollama_cloud", "gemini")

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or load_config()

    def create(self) -> BaseChatModel:
        llm_cfg = self.config.get("llm", {})
        provider = llm_cfg.get("provider", "ollama")
        temperature = float(llm_cfg.get("temperature", 0.7))

        if provider == "ollama":
            return self._build_ollama(llm_cfg, temperature)
        if provider == "ollama_cloud":
            return self._build_ollama_cloud(llm_cfg, temperature)
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
            # A local model can take minutes to emit a full JSON array of
            # samples; raise the underlying httpx client timeout so the SDK
            # doesn't kill the request at its ~120s default.
            sync_client_kwargs={"timeout": float(ollama_cfg.get("timeout", 600))},
            async_client_kwargs={"timeout": float(ollama_cfg.get("timeout", 600))},
        )

    def _build_ollama_cloud(self, llm_cfg: dict[str, Any], temperature: float) -> BaseChatModel:
        from langchain_ollama import ChatOllama

        cloud_cfg = llm_cfg.get("ollama_cloud", {})
        base_url = cloud_cfg.get("base_url", "https://ollama.com")
        # ChatOllama appends /api/chat itself; strip any trailing /api* or /
        # so a user-supplied "https://ollama.com/api" still works.
        base_url = base_url.rstrip("/")
        if base_url.endswith("/api"):
            base_url = base_url[: -len("/api")]

        api_key = cloud_cfg.get("api_key") or os.getenv("OLLAMA_CLOUD_API_KEY")
        if not api_key:
            raise ValueError(
                "Ollama Cloud API key not provided. Set OLLAMA_CLOUD_API_KEY env "
                "var or [llm.ollama_cloud].api_key in config.toml."
            )

        # Auth headers + a generous timeout, passed to the underlying httpx
        # clients. `client_kwargs` is deprecated in favor of the split
        # sync/async variants, which both accept `headers` and `timeout`.
        client_kwargs = {
            "headers": {"Authorization": f"Bearer {api_key}"},
            "timeout": float(cloud_cfg.get("timeout", 600)),
        }
        return ChatOllama(
            model=cloud_cfg.get("model", "mistral:7b-instruct"),
            temperature=temperature,
            base_url=base_url,
            sync_client_kwargs=client_kwargs,
            async_client_kwargs=client_kwargs,
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
            timeout=float(gemini_cfg.get("timeout", 600)),
        )


def get_model() -> BaseChatModel:
    """Convenience helper: build a model from the default config."""
    return ModelFactory().create()