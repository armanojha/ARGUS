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


def test_providers_stub_loads_and_has_empty_providers_list():
    settings = get_settings()
    data = load_providers_config(settings)
    assert isinstance(data, dict)
    assert data.get("providers") == []


def test_obsidian_stub_loads_and_is_disabled_by_default():
    settings = get_settings()
    data = load_obsidian_config(settings)
    assert isinstance(data, dict)
    assert data.get("enabled") is False
    assert data.get("write_back_root") == "90_ARGUS"
