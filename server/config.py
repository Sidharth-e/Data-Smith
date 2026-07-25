"""
Configuration loader for Data Smith server.

Loads environment variables from `.env` (and optional `.env.local`),
then parses `config.toml` and substitutes any `${VAR}` / `${VAR:-default}`
references with values from the environment.

Precedence (highest first):
    1. Real process environment (os.environ)
    2. `.env.local`  (gitignored, machine-specific overrides)
    3. `.env`        (gitignored, project defaults)
    4. `config.toml` literal values / `${VAR:-default}` fallbacks
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib  # type: ignore[no-redef]

SERVER_DIR = Path(__file__).resolve().parent
CONFIG_PATH = SERVER_DIR / "config.toml"
LOCAL_CONFIG_PATH = SERVER_DIR / "config.local.toml"
ENV_PATH = SERVER_DIR / ".env"
ENV_LOCAL_PATH = SERVER_DIR / ".env.local"

# Matches ${VAR} or ${VAR:-default}
_ENV_REF = re.compile(r"\$\{([A-Z0-9_]+)(?::-([^}]*))?\}")


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge `override` into `base` (returns a new dict)."""
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_env() -> None:
    """Load .env files. Later loads do not overwrite already-set vars."""
    # .env first, then .env.local overrides on top. `override=False` means
    # an explicitly exported shell env var wins over the file.
    load_dotenv(ENV_PATH, override=False)
    load_dotenv(ENV_LOCAL_PATH, override=False)


def _substitute(value: Any) -> Any:
    """Recursively replace ${VAR} / ${VAR:-default} refs in strings."""
    if isinstance(value, str):
        def repl(match: re.Match[str]) -> str:
            var, default = match.group(1), match.group(2)
            return os.environ.get(var, default if default is not None else "")
        return _ENV_REF.sub(repl, value)
    if isinstance(value, dict):
        return {k: _substitute(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_substitute(v) for v in value]
    return value


def load_config(path: Path | None = None) -> dict[str, Any]:
    """
    Load configuration from TOML, with env-var substitution.

    Args:
        path: Optional explicit config path. When None, loads `config.toml`
              and merges `config.local.toml` on top if it exists.

    Returns:
        Parsed configuration as a nested dict with `${VAR}` refs resolved.
    """
    _load_env()

    target = path or CONFIG_PATH
    with open(target, "rb") as f:
        config = tomllib.load(f)

    if path is None and LOCAL_CONFIG_PATH.exists():
        with open(LOCAL_CONFIG_PATH, "rb") as f:
            local = tomllib.load(f)
        config = _deep_merge(config, local)

    return _substitute(config)


# Eagerly loaded singleton for convenience
settings = load_config()