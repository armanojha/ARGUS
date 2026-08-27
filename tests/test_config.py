"""Tests for the Phase 00.1 config loader."""

from __future__ import annotations

from pathlib import Path

from app.config import (
    Settings,
    get_settings,
    load_obsidian_config,
    load_providers_config,
    load_yaml_config,
)


def test_settings_defaults():
    settings = Settings(_env_file=None)
    assert settings.env == "development"
    assert settings.log_level == "INFO"
    assert settings.obsidian_vault_path == ""


def test_settings_env_prefix_override(monkeypatch):
    monkeypatch.setenv("ARGUS_ENV", "test")
    monkeypatch.setenv("ARGUS_LOG_LEVEL", "DEBUG")
    settings = Settings(_env_file=None)
    assert settings.env == "test"
    assert settings.log_level == "DEBUG"


def test_config_paths_derive_from_config_dir(tmp_path):
    settings = Settings(_env_file=None, config_dir=tmp_path)
    assert settings.providers_config_path == tmp_path / "providers.yaml"
    assert settings.obsidian_config_path == tmp_path / "obsidian.yaml"


def test_get_settings_is_cached():
    a = get_settings()
    b = get_settings()
    assert a is b


def test_load_yaml_config_missing_file_returns_empty_dict(tmp_path: Path):
    missing = tmp_path / "does_not_exist.yaml"
    assert load_yaml_config(missing) == {}


def test_load_yaml_config_empty_file_returns_empty_dict(tmp_path: Path):
    empty = tmp_path / "empty.yaml"
    empty.write_text("", encoding="utf-8")
    assert load_yaml_config(empty) == {}


def test_load_yaml_config_reads_real_file(tmp_path: Path):
    sample = tmp_path / "sample.yaml"
    sample.write_text("a: 1\nb: two\n", encoding="utf-8")
    assert load_yaml_config(sample) == {"a": 1, "b": "two"}


def test_providers_config_loads_all_providers():
    """Test that providers.yaml loads all provider configurations (Phase 00.3 + 07)."""
    settings = get_settings()
    data = load_providers_config(settings)
    assert isinstance(data, dict)
    providers = data.get("providers", [])
    assert len(providers) == 3
    
    # Check Groq
    groq = next(p for p in providers if p["name"] == "groq")
    assert groq["enabled"] is True
    assert groq["api_key_env"] == "GROQ_API_KEY"
    assert groq["default_model"] == "openai/gpt-oss-120b"
    assert groq["fallback_models"] == ["openai/gpt-oss-20b"]
    assert groq["capabilities"]["structured_output"] is True
    assert groq["capabilities"]["tool_calling"] is True
    
    # Check Gemini
    gemini = next(p for p in providers if p["name"] == "gemini")
    assert gemini["enabled"] is True
    assert gemini["api_key_env"] == "GEMINI_API_KEY"
    assert gemini["default_model"] == "gemini-2.5-flash-lite"
    assert gemini["fallback_models"] == ["gemini-2.5-flash"]
    assert gemini["capabilities"]["structured_output"] is True
    assert gemini["capabilities"]["tool_calling"] is True
    
    # Check Cerebras
    cerebras = next(p for p in providers if p["name"] == "cerebras")
    assert cerebras["enabled"] is True
    assert cerebras["api_key_env"] == "CEREBRAS_API_KEY"
    assert cerebras["default_model"] == "gpt-oss-120b"
    assert cerebras["capabilities"]["structured_output"] is True
    assert cerebras["capabilities"]["tool_calling"] is True


def test_obsidian_config_loads_with_expected_fields():
    settings = get_settings()
    data = load_obsidian_config(settings)
    assert isinstance(data, dict)
    assert data.get("enabled") is True  # Phase 05 implemented
    assert data.get("write_back_root") == "90_ARGUS"