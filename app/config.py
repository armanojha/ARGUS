"""ARGUS core configuration (Phase 00.1 + 00.3).

Loads runtime settings from environment variables / a local .env file via
pydantic-settings, and provides small helpers to load the YAML config
stubs (`configs/providers.yaml`, `configs/obsidian.yaml`).

Only foundation-level settings live here. Provider-specific settings are
added when Phase 00.3 / Phase 07 implement the LLM gateway; retrieval and
graph settings are added in their owning phases.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Repository root = parent of the `app` package directory.
REPO_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Core ARGUS application settings.

    Environment variables are read with the ``ARGUS_`` prefix, e.g.
    ``ARGUS_ENV``, ``ARGUS_LOG_LEVEL``, ``ARGUS_DATA_DIR``. A local
    ``.env`` file (see ``.env.example``) is loaded automatically if
    present; it is never committed (see ``.gitignore``).
    """

    model_config = SettingsConfigDict(
        env_prefix="ARGUS_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    env: str = Field(default="development", description="Runtime environment name.")
    log_level: str = Field(default="INFO", description="Logging level (e.g. DEBUG, INFO, WARNING).")
    data_dir: Path = Field(default=REPO_ROOT / "data", description="Root directory for data artifacts.")
    config_dir: Path = Field(default=REPO_ROOT / "configs", description="Directory containing YAML config stubs.")
    obsidian_vault_path: str = Field(
        default="",
        description="Path to the user's Obsidian vault. Empty until Phase 05 wires ingestion.",
    )

    # --- LLM Gateway (Phase 00.3) ---
    llm_provider: str = Field(
        default="groq",
        description="Active LLM provider name (matches providers.yaml entry name).",
    )
    llm_model: str | None = Field(
        default=None,
        description="Override default model for active provider.",
    )
    llm_timeout: float = Field(
        default=30.0,
        description="Request timeout in seconds.",
    )
    llm_max_retries: int = Field(
        default=2,
        description="Max retry attempts for transient failures.",
    )
    llm_call_type: str = Field(
        default="general",
        description="Call type for policy routing (Phase 07).",
    )

    @property
    def providers_config_path(self) -> Path:
        return self.config_dir / "providers.yaml"

    @property
    def obsidian_config_path(self) -> Path:
        return self.config_dir / "obsidian.yaml"


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance (standard FastAPI-style dependency)."""
    return Settings()


def load_yaml_config(path: Path) -> dict[str, Any]:
    """Load a YAML file into a dict. Returns {} if the file is missing or empty.

    Deliberately tolerant: config stub files are allowed to be sparse or
    absent early in the project without crashing the app.
    """
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def load_providers_config(settings: Settings | None = None) -> dict[str, Any]:
    """Load `configs/providers.yaml`. Real provider wiring happens in Phase 00.3 / 07."""
    settings = settings or get_settings()
    return load_yaml_config(settings.providers_config_path)


def load_obsidian_config(settings: Settings | None = None) -> dict[str, Any]:
    """Load `configs/obsidian.yaml`. Real vault ingestion happens in Phase 05 / 09."""
    settings = settings or get_settings()
    return load_yaml_config(settings.obsidian_config_path)