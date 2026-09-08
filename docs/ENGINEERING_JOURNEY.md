# ARGUS Engineering Journey

## Overview

ARGUS was developed through 43 phases of measurement, experimentation, and production decisions. Each phase discovered a specific problem, performed controlled experiments, and made evidence-based decisions. This document summarizes the major milestones.

## Phase Progression

### Foundation (Phases 00-02)

**Problem:** Build a working RAG system with observable retrieval.

**Experiments:**
- FastAPI application with health, retrieval, and orchestration endpoints
- BM25 lexical retrieval with rank-bm25
- FAISS dense vector retrieval with sentence-transformers
- Hybrid retrieval with configurable fusion weights
- LangGraph state machine for research orchestration

**Decision:** Hybrid BM25+FAISS as the retrieval backbone. LangGraph for orchestration.

### Evidence & Graph (Phases 03-05)

**Problem:** Evidence needs provenance tracking and relationship modeling.

**Experiments:**
- SQLite evidence store with source → document → chunk hierarchy
- NetworkX MultiDiGraph for evidence relationships
- Entity extraction and relationship modeling
- Obsidian vault integration for knowledge management

**Decision:** SQLite for evidence storage (simple, reliable). NetworkX for graph (flexible, testable).

### Adaptive Research (Phases 06-07)

**Problem:** Different query types need different retrieval strategies.

**Experiments:**
- Query pattern classification (20 patterns)
- Adaptive retrieval policy routing
- Multi-model provider routing with fallback
- Evidence need planning for complex queries
- Stopping logic with 5 conditions

**Decision:** Pattern-aware routing with explicit configuration. No autonomous model selection.

### Memory & Multi-Agent (Phases 08-10)

**Problem:** System needs to learn from previous research and handle complex questions.

**Experiments:**
- 6-layer SQLite memory with promotion
- Persistent memory across research sessions
- Multi-agent debate for complex questions
- Coordinator pattern for agent orchestration

**Decision:** Memory is optional (feature-flagged). Multi-agent is validated but adds latency.

### Multimodal & UI (Phases 11-12)

**Problem:** Documents contain more than text; users need visibility.

**Experiments:**
- PaddleOCR + Tesseract for image text extraction
- PDF, spreadsheet, table processing
- Streamlit evidence explorer UI (later replaced)
- Answer quality evaluation framework

**Decision:** Multimodal processing is feature-flagged. Streamlit was replaced by Brain UI.

### Verification & Hardening (Phases 07b-07f)

**Problem:** Claims need verification; the system needs to be robust.

**Experiments:**
- Deterministic claim-support checking
- LLM-based verification with confidence scoring
- Post-synthesis verification (fail-safe)
- Error recovery and graceful degradation
- Parallel retrieval execution

**Decision:** Deterministic verification is the foundation. LLM-based verification is supplementary.

### Evaluation & Optimization (Phases 24-38)

**Problem:** Need to measure and improve quality systematically.

**Experiments:**
- Adaptive research policy with budget management
- Evidence selection optimization
- Retrieval evaluation with precision/recall metrics
- Synthesis quality ablation studies
- Provider health monitoring
- Conflict detection precision tuning

**Decision:** Evidence selection and conflict detection are production-ready. Provider-dependent benchmarks are unreliable.

### Conflict Detection (Phases 40-41)

**Problem:** Sources disagree; contradictions need detection and filtering.

**Experiments:**
- Deterministic pairwise contradiction detection
- Temporal, entity, and metric-aware conflict classification
- Query-aware conflict filtering
- Safe synthesis with conflict acknowledgment

**Decision:** Detection is production-ready. Filtering is feature-flagged. Synthesis rules are experimental.

### Brain UI (Phase 42)

**Problem:** Users need to see how ARGUS thinks.

**Experiments:**
- Single-file HTML/JS/D3.js application
- Pipeline visualization with interactive nodes
- Node inspector with what/why/how details
- Evidence traceability from citation to source
- Architecture view with component explanations
- Demo mode for offline demonstration

**Decision:** Single-file architecture for simplicity. Complete implementation of all spec requirements.

### Correctness Fix (Phase 43)

**Problem:** Real-world query returned irrelevant evidence and false contradictions.

**Experiments:**
- Root cause analysis: missing `"%"` in `_UNIT_NORMALIZE`
- Relevance gate implementation
- Topic coherence requirements
- Safe synthesis fallback

**Decision:** One-line fix (`"%": "%"`) plus safety improvements. Backend frozen.

## Engineering Principles

### 1. Measure Before Changing

Every phase started with measurement. We never changed code without understanding the current state.

### 2. Feature Flags for Experiments

Experimental features are behind `ARGUS_*_ENABLED` flags. This allows validation without risk.

### 3. Deterministic First

Where possible, we use deterministic algorithms (BM25, lexical overlap, regex extraction) over LLM calls. This makes the system testable and predictable.

### 4. Fail-Safe Design

When providers fail, the system degrades gracefully. Evidence is never discarded. Synthesis runs over whatever evidence was accumulated.

### 5. Honest Evaluation

We rejected contaminated benchmark results. Provider instability prevented clean experiments in Phases 36, 37, and 40. We did not use these results to justify production changes.

### 6. Backend Freeze

After Phase 43, the backend is frozen. This is intentional — the architecture is validated and tested. Future work focuses on documentation, presentation, and reproducibility.

## What Was Built

| Component | Lines of Code | Tests | Status |
|-----------|--------------|-------|--------|
| Retrieval | ~1,500 | 100+ | Production-ready |
| Orchestration | ~1,200 | 150+ | Production-ready |
| Verification | ~800 | 50+ | Production-ready |
| Evidence Store | ~500 | 30+ | Production-ready |
| LLM Gateway | ~1,000 | 80+ | Production-ready |
| Brain UI | ~4,000 | N/A | Complete |
| Tests | N/A | 956 | Passing |

## What Was Decided Against

| Proposal | Reason Rejected |
|----------|----------------|
| Replace BM25 with pure vector search | BM25 outperforms for exact queries; hybrid is better |
| Replace embeddings with BGE-M3 | Requires significant VRAM; not default-worthy |
| Add streaming (SSE/WebSocket) | Complexity not justified for current use case |
| Componentize Brain UI | Single-file simplicity is a feature, not a bug |
| Autonomous model selection | Explicit configuration is more predictable |
| Production deployment | ARGUS is a research project, not a production system |
