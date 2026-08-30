"""ARGUS core configuration (Phase 00.1 + 00.3 + 01).

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
from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Repository root = parent of the `app` package directory.
REPO_ROOT = Path(__file__).resolve().parent.parent


def load_dotenv_file(path: Path | None = None) -> bool:
    """Load a local ``.env`` file into os.environ for non-``ARGUS_`` secrets.

    pydantic-settings only reads ``.env`` for ``ARGUS_``-prefixed settings and
    never mutates os.environ, while the LLM gateway reads provider API keys via
    ``os.getenv(<api_key_env>)``. This bridges that gap: ``.env`` supplies
    keys to the gateway while real process environment variables take
    precedence (override=False — keys remain environment-controlled).
    """
    path = path or (REPO_ROOT / ".env")
    if not path.exists():
        return False
    return load_dotenv(path, override=False)


load_dotenv_file()


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

    # --- Retrieval / Evidence Store (Phase 01) ---
    evidence_db_path: Path = Field(
        default=REPO_ROOT / "data" / "evidence.db",
        description="Path to SQLite evidence store database.",
    )
    bm25_index_path: Path = Field(
        default=REPO_ROOT / "data" / "indexes" / "bm25.pkl",
        description="Path to BM25 index file.",
    )
    faiss_index_path: Path = Field(
        default=REPO_ROOT / "data" / "indexes" / "faiss.index",
        description="Path to FAISS vector index file.",
    )
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="Local embedding model for vector retrieval.",
    )
    chunk_size: int = Field(
        default=512,
        description="Target chunk size in tokens for document chunking.",
    )
    chunk_overlap: int = Field(
        default=64,
        description="Overlap between adjacent chunks in tokens.",
    )
    retrieval_top_k: int = Field(
        default=20,
        description="Number of candidates to retrieve from each index before fusion.",
    )
    rerank_top_k: int = Field(
        default=10,
        description="Number of results to return after reranking.",
    )

    # --- Agentic Orchestration (Phase 02) ---
    orchestration_max_iterations: int = Field(
        default=3,
        description="Hard ceiling on retrieval iterations per query, regardless of what the plan requests.",
    )
    orchestration_token_budget: int = Field(
        default=6000,
        description="Hard ceiling on accumulated evidence+usage token estimate per query, "
        "regardless of what the plan requests.",
    )
    orchestration_default_subquestions: int = Field(
        default=3,
        description="Fallback subquestion count used if the planner LLM call fails.",
    )
    orchestration_retrieval_top_k: int = Field(
        default=8,
        description="Number of evidence candidates to retrieve per subquestion in the agentic loop.",
    )
    orchestration_llm_timeout: float = Field(
        default=30.0,
        description="Per-call timeout (seconds) for orchestration LLM calls (analysis/plan/assess/synthesis).",
    )

    # --- Evidence Graph (Phase 03) ---
    graph_path: Path = Field(
        default=REPO_ROOT / "data" / "graph" / "evidence_graph.pkl",
        description="Path to persisted NetworkX graph file.",
    )
    graph_extraction_batch_size: int = Field(
        default=10,
        description="Number of chunks to process per extraction LLM call.",
    )
    graph_max_hops: int = Field(
        default=2,
        description="Maximum hops for graph-based multi-hop retrieval.",
    )
    graph_extraction_enabled: bool = Field(
        default=True,
        description="Whether to run LLM-based extraction during ingestion.",
    )

    # --- Verification (Phase 04) ---
    verification_enabled: bool = Field(
        default=True,
        description="Whether to run verification on claims.",
    )
    verification_max_evidence_items: int = Field(
        default=20,
        description="Max evidence items to send to verifier LLM.",
    )
    verification_contradiction_threshold: float = Field(
        default=0.3,
        description="Minimum severity for contradiction detection.",
    )
    verification_re_retrieval_enabled: bool = Field(
        default=True,
        description="Whether to trigger re-retrieval on evidence gaps (MVP: one cycle).",
    )
    verification_max_re_retrieval_cycles: int = Field(
        default=1,
        description="Max additional retrieval cycles (MVP: 1).",
    )

    # --- Obsidian Ingestion (Phase 05) ---
    obsidian_enabled: bool = Field(
        default=False,
        description="Whether Obsidian vault ingestion is enabled.",
    )
    obsidian_vault_path: str = Field(
        default="",
        description="Path to the user's Obsidian vault. Empty until Phase 05 wires ingestion.",
    )
    obsidian_write_back_root: str = Field(
        default="90_ARGUS",
        description="Write-back root directory name within the vault (per V3 §4.3).",
    )
    obsidian_incremental_sync: bool = Field(
        default=True,
        description="Whether to use incremental sync (only process changed files).",
    )
    obsidian_exclude_patterns: list[str] = Field(
        default_factory=lambda: ["90_ARGUS/**", ".obsidian/**", ".git/**", ".trash/**"],
        description="Glob patterns to exclude from vault scanning.",
    )
    obsidian_chunking_strategy: str = Field(
        default="obsidian_sections_v1",
        description="Chunking strategy for Obsidian notes.",
    )

    # --- Adaptive Research Policy (Phase 06) ---
    retrieval_policy_enabled: bool = Field(
        default=True,
        description="Whether adaptive retrieval policy is enabled.",
    )
    retrieval_policy_config_path: str = Field(
        default="configs/retrieval_policy.yaml",
        description="Path to retrieval policy configuration file.",
    )
    active_evidence_seeking_enabled: bool = Field(
        default=True,
        description="Whether active evidence seeking is enabled.",
    )
    stopping_logic_enabled: bool = Field(
        default=True,
        description="Whether full stopping logic is enabled.",
    )
    stopping_claim_support_threshold: float = Field(
        default=0.7,
        description="Minimum claim support threshold for stopping.",
    )
    stopping_evidence_gain_threshold: float = Field(
        default=0.05,
        description="Minimum evidence gain threshold for stopping.",
    )
    active_seeking_min_priority: float = Field(
        default=0.5,
        description="Minimum evidence-gap priority before active re-retrieval is triggered.",
    )
    active_seeking_quality_threshold: float = Field(
        default=0.4,
        description="Top retrieved score below this flags an evidence-quality gap.",
    )

    # --- Multi-Model Fabric (Phase 07) ---
    multimodel_enabled: bool = Field(
        default=False,
        description="Whether multi-model fabric is enabled.",
    )
    multimodel_providers_config_path: str = Field(
        default="configs/providers.yaml",
        description="Path to multi-model providers configuration.",
    )
    multimodel_routing_enabled: bool = Field(
        default=True,
        description="Whether capability-aware routing is enabled.",
    )
    multimodel_fallback_enabled: bool = Field(
        default=True,
        description="Whether provider fallback is enabled.",
    )
    multimodel_cross_verification_enabled: bool = Field(
        default=True,
        description="Whether cross-model verification is enabled.",
    )
    multimodel_telemetry_enabled: bool = Field(
        default=True,
        description="Whether telemetry collection is enabled.",
    )
    multimodel_call_ceiling: int = Field(
        default=16,
        description="Hard ceiling on logical LLM calls per run.",
    )
    multimodel_verifier_different_provider: bool = Field(
        default=True,
        description="Whether verifier must use different provider from synthesizer.",
    )

    # --- Memory & Self-Evolution (Phase 08) ---
    memory_enabled: bool = Field(
        default=False,
        description="Whether persistent memory is enabled.",
    )
    memory_db_path: Path = Field(
        default=REPO_ROOT / "data" / "memory" / "memory.db",
        description="Path to SQLite memory database.",
    )
    memory_layers: list[str] = Field(
        default_factory=lambda: [
            "working",
            "long_term_knowledge",
            "research_history",
            "source_memory",
            "user_memory",
            "vault_memory",
        ],
        description="Enabled memory layers.",
    )
    memory_max_records_per_layer: int = Field(
        default=10000,
        description="Max records per memory layer.",
    )
    memory_confidence_threshold: float = Field(
        default=0.7,
        description="Minimum confidence for memory promotion.",
    )
    graph_versioning_enabled: bool = Field(
        default=True,
        description="Whether graph versioning/deltas are enabled.",
    )
    graph_delta_retention_days: int = Field(
        default=90,
        description="Days to retain graph deltas.",
    )

    # --- Obsidian Full Integration (Phase 09) ---
    obsidian_full_enabled: bool = Field(
        default=False,
        description="Whether full Obsidian integration is enabled.",
    )
    obsidian_classification_enabled: bool = Field(
        default=True,
        description="Whether 7-class classification is enabled.",
    )
    obsidian_hypothesis_conversion_enabled: bool = Field(
        default=True,
        description="Whether hypothesis-to-research conversion is enabled.",
    )
    obsidian_writeback_enabled: bool = Field(
        default=True,
        description="Whether write-back proposals are enabled.",
    )
    obsidian_vault_graph_alignment_enabled: bool = Field(
        default=True,
        description="Whether vault-graph alignment is enabled.",
    )

    # --- Multi-Agent Challenge (Phase 10) ---
    multiagent_enabled: bool = Field(
        default=False,
        description="Whether multi-agent debate is enabled.",
    )
    multiagent_max_rounds: int = Field(
        default=3,
        description="Maximum debate rounds.",
    )
    multiagent_skeptic_threshold: float = Field(
        default=0.7,
        description="Confidence threshold to activate skeptic.",
    )
    multiagent_disagreement_threshold: float = Field(
        default=0.3,
        description="Disagreement severity threshold for targeted retrieval.",
    )

    # --- Multimodal Intelligence (Phase 11) ---
    multimodal_enabled: bool = Field(
        default=False,
        description="Master switch for multimodal ingestion. When False, all sub-features are disabled regardless of their individual flags.",
    )
    multimodal_ocr_enabled: bool = Field(
        default=True,
        description="Whether OCR fallback is enabled (requires multimodal_enabled=True).",
    )
    multimodal_table_extraction_enabled: bool = Field(
        default=True,
        description="Whether table extraction is enabled (requires multimodal_enabled=True).",
    )
    multimodal_web_ingestion_enabled: bool = Field(
        default=True,
        description="Whether web page ingestion is enabled (requires multimodal_enabled=True).",
    )
    multimodal_spreadsheet_enabled: bool = Field(
        default=True,
        description="Whether spreadsheet ingestion is enabled (requires multimodal_enabled=True).",
    )
    multimodal_chart_extraction_enabled: bool = Field(
        default=False,
        description="Whether chart/image extraction is enabled (later stage).",
    )
    ocr_languages: list[str] = Field(
        default_factory=lambda: ["eng"],
        description="Languages passed to tesseract/pytesseract for OCR fallback (Phase 11).",
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