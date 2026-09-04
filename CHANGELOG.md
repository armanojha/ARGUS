# Changelog

All notable changes to ARGUS are documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.1.0] - 2026-09-04

### Added

#### Phase 00 — Foundation
- FastAPI application skeleton with health endpoint and structured error envelopes
- Pydantic settings with env-var loading and YAML config support
- LLM gateway abstraction with provider-agnostic interface
- Structured logging with request ID propagation

#### Phase 01 — Hybrid RAG
- BM25 keyword retrieval with save/load index
- FAISS vector retrieval with sentence-transformer embeddings
- Hybrid retriever with configurable fusion weights
- Ingestion pipeline with content-checksum deduplication
- Evidence store (SQLite) for sources, documents, and chunks

#### Phase 02 — Agentic RAG
- LangGraph-based orchestration loop with state machine
- Query analysis and research planning
- Adaptive retrieval with evidence gap detection
- Stop conditions (claim support, budget ceiling, negligible gain)
- Simple query fast path for direct questions

#### Phase 03 — Evidence Graph
- NetworkX-backed evidence graph with 8 edge types
- Entity, claim, and event node types
- Temporal edge support with date extraction
- Provenance tracking from evidence to graph

#### Phase 04 — Verification & Contradiction
- Claim verification against source evidence
- Confidence scoring with composite components
- Contradiction detection with predicate similarity
- Evidence gap detection and re-retrieval triggers

#### Phase 05 — Obsidian Ingestion (MVP)
- Obsidian vault scanner with frontmatter parsing
- Wikilink extraction and resolution
- Callout and code block extraction
- Sync manager with change detection

#### Phase 06 — Adaptive Research Policy
- Configurable retrieval policy per question type
- Evidence gap detection with targeted re-retrieval
- Selective verification gate (skip for low-risk, high-confidence)
- Truthful outcome reporting (answered, degraded, fallback, not-found)
- Citation normalization (fullwidth bracket support)

#### Phase 07 — Multi-Model Fabric
- Multi-model router with call-type routing
- Provider health tracking with cooldown and recovery
- Intra-provider fallback (model-level and provider-level)
- Quota tracking with per-model limits
- Telemetry with routing decisions and error classification
- Call ceiling hard stop for budget control
- Cross-model verification support

#### Phase 08 — Memory & Self-Evolution
- SQLite-backed memory store with 5 layers
- Promotion engine with confidence-based rules
- Graph version manager with delta tracking
- Memory-aware planner integration
- Null memory store for graceful degradation

#### Phase 09 — Obsidian Integration (Full)
- 7-class knowledge taxonomy (source, knowledge, hypothesis, project, task, research, reference)
- Hypothesis research runner with objective conversion
- Vault-graph alignment with canonical entities
- Write-back proposal manager with user approval
- Promotion path with wikilink pointers (not content copies)

#### Phase 10 — Multi-Agent Challenge
- 5 agent roles: Researcher, Skeptic, Alternative Hypothesis, Verifier, Judge
- Agent coordinator with activation rules
- Multi-round debate with disagreement detection
- Conflicting evidence handling with score-based merging

#### Phase 11 — Multimodal Intelligence
- Web page ingestion with HTTP fetch and HTML parsing
- Spreadsheet ingestion (XLSX, XLS, XLSM, CSV)
- OCR fallback via Tesseract with confidence scoring
- Table extraction from PDFs and images
- Chart extraction (placeholder for vision model)
- Multimodal-enabled gating via `ARGUS_MULTIMODAL_ENABLED`

#### Phase 12 — Research UI + Evaluation
- Streamlit evidence-explorer UI with four panels
- API endpoints: research, verify, KB status, brain status, vault status, telemetry
- Run trace API with JSONL persistence
- Telemetry dashboard with routing decisions and error classification
- Benchmark harness with ablation studies

### Fixed
- `datetime.utcnow()` deprecation (replaced with `datetime.now(UTC)`)
- FAISS index boundary error (empty index guard)
- `assert` statements replaced with `RuntimeError`
- `Settings()` direct calls replaced with `get_settings()`
- Hardcoded developer paths (`E:/KNOWLEDGE BASE`, `E:\ARGUS_VAULT`)

### Security
- Removed 14 internal development reports from repository
- Removed 3 data files containing leaked Groq organization ID
- Added MIT LICENSE file
- Added SECURITY.md with API key guidance
- Added CONTRIBUTING.md with contribution guidelines
