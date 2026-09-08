"""Tests for explicit run modes (ARGUS_MODE).

Modes collapse the opt-in behavioral flags into four tested configurations.
A mode only supplies DEFAULTS: any explicitly-set flag (env, .env, kwarg)
always wins. No LLM calls. No network.
"""

import pytest
from pydantic import ValidationError

from app.config import ARGUS_MODE_FLAGS, ARGUS_MODES, Settings

_ALL_MODE_ENV = ["ARGUS_MODE"] + [
    f"ARGUS_{f.upper()}" for f in ARGUS_MODE_FLAGS
]


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for var in _ALL_MODE_ENV:
        monkeypatch.delenv(var, raising=False)


class TestBaselineIsHistoricalDefault:
    def test_default_mode_is_baseline(self):
        assert Settings().mode == "baseline"

    def test_baseline_leaves_all_opt_ins_false(self):
        s = Settings()
        for flag in ARGUS_MODE_FLAGS:
            assert getattr(s, flag) is False, flag

    def test_baseline_preserves_core_defaults(self):
        s = Settings()
        assert s.retrieval_policy_enabled is True
        assert s.verification_enabled is True
        assert s.active_evidence_seeking_enabled is True
        assert s.stopping_logic_enabled is True
        assert s.multimodal_enabled is False
        assert s.memory_enabled is False


class TestModeMaps:
    def test_research_enables_adaptive_only(self):
        s = Settings(mode="research")
        assert s.adaptive_research_enabled is True
        for flag in ARGUS_MODE_FLAGS:
            if flag != "adaptive_research_enabled":
                assert getattr(s, flag) is False, flag

    def test_verified_enables_verification_stack(self):
        s = Settings(mode="verified")
        assert s.conflict_filtering_enabled is True
        assert s.conflict_safe_synthesis_enabled is True
        assert s.conflict_semantic_check_enabled is True
        assert s.verified_synthesis_enabled is True
        assert s.adaptive_research_enabled is False
        assert s.memory_enabled is False
        assert s.multiagent_enabled is False

    def test_full_enables_reasoning_stack(self):
        s = Settings(mode="full")
        for flag in (
            "adaptive_research_enabled",
            "conflict_filtering_enabled",
            "conflict_safe_synthesis_enabled",
            "conflict_semantic_check_enabled",
            "verified_synthesis_enabled",
            "memory_enabled",
            "multiagent_enabled",
        ):
            assert getattr(s, flag) is True, flag

    def test_full_leaves_hardware_flags_off(self):
        s = Settings(mode="full")
        assert s.multimodal_enabled is False  # needs Tesseract binary
        assert s.bge_m3_enabled is False  # experimental model download
        assert s.obsidian_enabled is False  # needs vault path

    def test_mode_from_env(self, monkeypatch):
        monkeypatch.setenv("ARGUS_MODE", "research")
        assert Settings().adaptive_research_enabled is True


class TestExplicitBeatsMode:
    def test_env_flag_beats_mode_on(self, monkeypatch):
        monkeypatch.setenv("ARGUS_MODE", "baseline")
        monkeypatch.setenv("ARGUS_MEMORY_ENABLED", "true")
        s = Settings()
        assert s.memory_enabled is True
        assert s.adaptive_research_enabled is False

    def test_env_flag_beats_mode_off(self, monkeypatch):
        monkeypatch.setenv("ARGUS_MODE", "full")
        monkeypatch.setenv("ARGUS_MULTIAGENT_ENABLED", "false")
        s = Settings()
        assert s.multiagent_enabled is False
        assert s.memory_enabled is True  # rest of mode still applies

    def test_kwarg_beats_mode(self):
        s = Settings(mode="full", multiagent_enabled=False)
        assert s.multiagent_enabled is False
        assert s.memory_enabled is True

    def test_kwarg_enables_under_baseline(self):
        s = Settings(adaptive_research_enabled=True)
        assert s.mode == "baseline"
        assert s.adaptive_research_enabled is True


class TestInvalidMode:
    def test_unknown_mode_rejected(self):
        with pytest.raises(ValidationError):
            Settings(mode="turbo")

    def test_mode_map_covers_all_flags(self):
        for mode, flags in ARGUS_MODES.items():
            for flag in flags:
                assert flag in ARGUS_MODE_FLAGS, (mode, flag)
