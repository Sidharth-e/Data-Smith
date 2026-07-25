"""
Centralized error handling for Data Smith.

Defines a single `GenerationError` that surfaces a friendly, flat message to
API consumers (no LangChain / provider tracebacks) and maps provider-specific
LLM failures into meaningful HTTP-style states.
"""

from __future__ import annotations

from typing import Any


class GenerationError(Exception):
    """
    Raised when dataset generation cannot proceed for a known reason.

    `user_message` is safe to expose to API clients. `detail` holds optional
    diagnostic context (provider, status, etc.) for logs.
    """

    def __init__(self, user_message: str, *, detail: Any = None) -> None:
        super().__init__(user_message)
        self.user_message = user_message
        self.detail = detail


def map_llm_error(exc: BaseException) -> GenerationError:
    """
    Convert a low-level provider/LangChain exception into a clean
    `GenerationError` with a human-readable message.
    """
    # Lazy imports keep optional providers out of the import graph when unused.
    try:
        from ollama._types import ResponseError as _OllamaResponseError
    except Exception:  # pragma: no cover
        _OllamaResponseError = None

    try:
        import httpx as _httpx
    except Exception:  # pragma: no cover
        _httpx = None

    try:
        from google.genai.errors import APIError as _GeminiAPIError
    except Exception:  # pragma: no cover
        _GeminiAPIError = None

    # --- Ollama (local + cloud) ---
    if _OllamaResponseError is not None and isinstance(exc, _OllamaResponseError):
        status = getattr(exc, "status_code", None)
        text = str(exc) or ""
        lower = text.lower()
        if "not found" in lower and "model" in lower:
            model = _extract_model_name(text)
            who = f"model '{model}'" if model else "the requested model"
            return GenerationError(
                f"{who} is not available on the configured Ollama server. "
                f"Pull it first or pick another model in config.toml.",
                detail={"provider": "ollama", "status": status, "raw": text},
            )
        if status == 401 or "unauthorized" in lower or "api key" in lower:
            return GenerationError(
                "Ollama Cloud rejected the request: invalid or missing API key.",
                detail={"provider": "ollama_cloud", "status": status, "raw": text},
            )
        if status == 404:
            return GenerationError(
                "Ollama server returned 404. Check base_url and that the model is pulled.",
                detail={"provider": "ollama", "status": status, "raw": text},
            )
        return GenerationError(
            f"Ollama request failed (status {status}).",
            detail={"provider": "ollama", "status": status, "raw": text},
        )

    # --- httpx network/transport errors (used by the Ollama SDK) ---
    if _httpx is not None:
        if isinstance(exc, _httpx.ConnectError):
            return GenerationError(
                "Cannot reach the Ollama server. Is Ollama running at the configured base_url?",
                detail={"provider": "ollama", "kind": "connect_error"},
            )
        if isinstance(exc, _httpx.TimeoutException):
            return GenerationError(
                "The LLM request timed out. Try fewer samples or a faster model.",
                detail={"provider": "ollama", "kind": "timeout"},
            )
        if isinstance(exc, _httpx.HTTPError):
            return GenerationError(
                "Network error while contacting the LLM provider.",
                detail={"provider": "ollama", "kind": "http_error", "raw": str(exc)},
            )

    # --- Gemini ---
    if _GeminiAPIError is not None and isinstance(exc, _GeminiAPIError):
        status = getattr(exc, "code", None) or getattr(exc, "status", None)
        lower = str(exc).lower()
        if "api_key" in lower or "permission" in lower or status in (401, 403):
            return GenerationError(
                "Gemini rejected the request: invalid or missing API key.",
                detail={"provider": "gemini", "status": status, "raw": str(exc)},
            )
        if "quota" in lower or "rate" in lower or status == 429:
            return GenerationError(
                "Gemini quota/rate limit reached. Slow down or upgrade your plan.",
                detail={"provider": "gemini", "status": status, "raw": str(exc)},
            )
        return GenerationError(
            f"Gemini request failed (status {status}).",
            detail={"provider": "gemini", "status": status, "raw": str(exc)},
        )

    # --- Generic fallback: surface the message but strip nothing else ---
    return GenerationError(
        f"Dataset generation failed: {exc}",
        detail={"provider": "unknown", "raw": str(exc)},
    )


def _extract_model_name(text: str) -> str:
    """Pull a 'name:tag' token out of an Ollama 'model X not found' message."""
    import re

    m = re.search(r"model\s+['\"]?([A-Za-z0-9_.:\-]+)['\"]?", text)
    return m.group(1) if m else ""