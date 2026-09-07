# ARGUS

### Evidence-Grounded AI Research Intelligence

An AI research system that makes retrieval, evidence verification, conflicts, and answer provenance visible.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-925%20passed-brightgreen.svg)](#validation)

---

## The Problem

Traditional RAG systems retrieve documents and generate answers. The research process is hidden. You get an answer but cannot see:

- What evidence was found
- How it was verified
- Whether sources conflict
- Why this answer over another

## What ARGUS Does Differently

ARGUS makes the entire research process observable:

```
Query
  ↓
Analysis        — What kind of question is this?
  ↓
Planning        — What evidence do we need?
  ↓
Retrieval       — Hybrid BM25 + dense search
  ↓
Evidence        — Ranked chunks with source paths
  ↓
Verification    — Every claim checked against evidence
  ↓
Conflict Detection — Contradictions acknowledged, not hidden
  ↓
Synthesis       — Answer grounded in verified evidence
  ↓
Answer          — With inline citations and source trail
```

Every step is visible in the **ARGUS Brain UI**.

---

## The ARGUS Brain

The Brain is the primary interface. It visualizes how ARGUS thinks through a research question.

**Open:** `http://localhost:8000/brain`

### Pipeline View

The default view shows the research flow as an interactive graph. Each node represents a stage: query, analysis, plan, evidence, verification, conflict, synthesis, answer. Click any node to inspect what happened at that stage.

### Node Inspector

Every node explains:

- **What is this?** — Simple description
- **Why does ARGUS need it?** — Purpose in the pipeline
- **What did it receive?** — Input data
- **What did it produce?** — Output data
- **Technical details** — Toggle for implementation-level information

### Evidence Traceability

Trace from answer back to source:

```
Answer Claim → Citation [1] → Evidence Chunk → Source Document
```

Click any citation chip to see the evidence text, source path, and relevance score.

### Conflict Visualization

When sources disagree, ARGUS shows:

- Conflict type (different timeframe, genuine contradiction, etc.)
- Confidence level
- Entities and metrics involved
- How the conflict was resolved or acknowledged

### Architecture View

An interactive diagram showing all ARGUS components. Click any component to learn what it does, why it exists, and what goes in/out.

### Demo Mode

No API key? No problem. Click **Load Demo** to see a pre-recorded research result with full pipeline visualization.

---

## Quick Demo

```bash
# 1. Clone and install
git clone https://github.com/armanojha/ARGUS.git
cd ARGUS
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\Activate   # Windows PowerShell

pip install -e ".[core,retrieval,graph,multimodal,dev-test]"

# 2. Configure (add at least one API key)
cp .env.example .env
# Edit .env — add GROQ_API_KEY or GEMINI_API_KEY

# 3. Start the server
uvicorn app.api.main:app --reload

# 4. Open the Brain
# http://localhost:8000/brain

# 5. Try Demo Mode (no API key needed)
# Click "Load Demo" on the pipeline view
```

### Demo Questions

These showcase different ARGUS capabilities:

| Question | Demonstrates |
|----------|-------------|
| "Compare revenue growth of Acme and Globex in 2024 vs 2025" | Multi-source comparison, numerical evidence |
| "What evidence supports the claim that AI adoption is accelerating?" | Evidence gathering, verification |
| "Find conflicting information about climate change impacts" | Conflict detection, source disagreement |

---

## Architecture

```
User Query
    ↓
Query Analysis          — Pattern detection, entity extraction
    ↓
Research Policy         — Strategy selection, iteration budget
    ↓
Evidence Planning       — Sub-query generation
    ↓
Hybrid Retrieval        — BM25 (lexical) + FAISS (dense) fusion
    ↓
Reranking               — Quality scoring, relevance ranking
    ↓
Evidence Verification   — Claim-support checking, confidence scoring
    ↓
Conflict Detection      — Contradiction identification, query-aware filtering
    ↓
Grounded Synthesis      — Answer generation from verified evidence
    ↓
Answer Quality          — Grounding validation, citation checking
    ↓
Final Answer + Citations
```

### Technical Components

| Component | Implementation |
|-----------|---------------|
| **Retrieval** | Hybrid BM25 + FAISS dense search with configurable fusion weights |
| **Reranking** | Pluggable reranker interface (NoOp default, swap-in any model) |
| **Verification** | Deterministic claim-support checking against evidence chunks |
| **Conflict Detection** | Temporal extraction, unit normalization, entity matching, confidence scoring |
| **Synthesis** | Grounded generation with inline citations and conflict acknowledgment |
| **Memory** | SQLite-backed 6-layer memory with promotion and provenance |
| **Graph** | NetworkX MultiDiGraph with 6 node types, 8 edge types |
| **LLM Gateway** | Multi-model router with call-type routing, fallback, and quota tracking |
| **OCR** | PaddleOCR + Tesseract fallback for multimodal document processing |
| **Brain UI** | Single-file HTML/JS/D3.js with Canvas rendering |

### Provider Configuration

ARGUS supports 6 LLM providers with explicit per-call-type routing:

| Provider | Models | API Key |
|----------|--------|---------|
| **Groq** | `gpt-oss-120b`, `gpt-oss-20b` | `GROQ_API_KEY` |
| **Gemini** | `gemini-3.5-flash-lite`, `gemini-3.5-flash` | `GEMINI_API_KEY` |
| **Cerebras** | `gpt-oss-120b` | `CEREBRAS_API_KEY` |
| **Z.ai** | GLM models | `ZAI_API_KEY` |
| **NVIDIA NIM** | Nemotron models | `NVIDIA_NIM_API_KEY` |
| **Zen** | OpenCode models | `OPENCODE_ZEN_API_KEY` |

Models are assigned per call type (query analysis, planning, synthesis, verification) via `configs/model_policy.yaml`. All assignments are explicit configuration — ARGUS never autonomously selects models.

---

## Validation

| Metric | Status |
|--------|--------|
| Test suite | **925 passed**, 26 skipped, 0 failures |
| Backend | **Frozen** — no modifications since Phase 41 |
| Brain UI | **Complete** — 25/25 spec requirements implemented |
| Conflict detection | Production-ready (feature-flagged) |
| Evidence verification | Deterministic, independently tested |

### What Has Been Validated

- Hybrid retrieval (BM25 + FAISS) with configurable fusion
- Multi-query evidence planning
- Deterministic claim verification
- Conflict detection with temporal/entity/unit awareness
- Query-aware conflict filtering (80% → 0% false positive rate)
- Grounded synthesis with inline citations
- Evidence traceability from answer to source
- Memory architecture with 6 layers and promotion
- Multi-model provider routing with fallback

### Known Infrastructure Limitations

- **Provider instability**: Free-tier rate limits affect live E2E synthesis latency and experimental benchmark cleanliness. Core components are independently tested.
- **No streaming**: Backend processes queries synchronously. Progressive pipeline display is a UI simulation.
- **Per-stage timing**: Not currently exposed by the backend. Aggregate telemetry is available.

---

## Project Structure

```
ARGUS/
├── app/
│   ├── api/              FastAPI routes (health, retrieval, orchestration, brain)
│   ├── config.py         Pydantic Settings with env-var loading
│   ├── evaluation/       Answer quality evaluation
│   ├── evidence/         SQLite evidence store
│   ├── graph/            NetworkX evidence graph
│   ├── ingestion/        Document ingestion pipeline, OCR
│   ├── integrations/     Obsidian vault integration
│   ├── llm_gateway/      Multi-model LLM router, providers, telemetry
│   ├── memory/           6-layer SQLite memory store
│   ├── orchestration/    LangGraph research state machine
│   ├── reranking/        Pluggable reranker interface
│   ├── retrieval/        Hybrid BM25 + FAISS retrieval
│   ├── ui/brain/         Brain UI (single-file HTML/JS/D3.js)
│   └── verification/     Claim verification engine
├── benchmarks/           Phase reports, evaluation data, benchmark harness
├── configs/              Provider, model policy, retrieval policy, Obsidian config
├── docs/                 Architecture docs, workflow visualization
├── knowledge_base/       User document corpus
├── scripts/              Ingestion, OCR, test runners
├── tests/                Test suite (mirrors app/ structure)
├── .env.example          Environment variable template
├── pyproject.toml        Build config, dependencies, tool settings
└── LICENSE               MIT
```

---

## Configuration

| File | Purpose |
|------|---------|
| `.env` | API keys and environment variables (gitignored) |
| `configs/providers.yaml` | LLM provider definitions |
| `configs/model_policy.yaml` | Call-type routing and model assignment |
| `configs/retrieval_policy.yaml` | 10 retrieval patterns with fusion weights |
| `configs/obsidian.yaml` | Obsidian vault integration |

### Feature Flags

| Flag | Default | Description |
|------|---------|-------------|
| `ARGUS_MEMORY_ENABLED` | `true` | Persistent memory system |
| `ARGUS_MULTIMODAL_ENABLED` | `true` | Multimodal document processing |
| `ARGUS_BGE_M3_ENABLED` | `false` | Experimental BGE-M3 embeddings |
| `ARGUS_CONFLICT_FILTERING_ENABLED` | `false` | Query-aware conflict filtering |
| `ARGUS_CONFLICT_SAFE_SYNTHESIS_ENABLED` | `false` | Conflict-aware synthesis rules |

---

## Documentation

| Document | Description |
|----------|-------------|
| [Architecture](docs/ARCHITECTURE.md) | System architecture and component details |
| [Brain UI](docs/BRAIN_UI.md) | Interface guide and feature documentation |
| [Quickstart](docs/QUICKSTART.md) | Get running in under 10 minutes |
| [Limitations](docs/LIMITATIONS.md) | Known limitations and honest status |
| [Changelog](CHANGELOG.md) | Version history |
| [Contributing](CONTRIBUTING.md) | Contribution guidelines |
| [Security](SECURITY.md) | API key and data security guidance |

---

## Roadmap

### Completed

- Hybrid retrieval (BM25 + FAISS)
- Evidence graph with temporal reasoning
- Deterministic verification engine
- Conflict detection with query-aware filtering
- Multi-model provider routing
- Persistent memory architecture
- Multimodal document processing
- Brain UI with pipeline visualization

### Future

- Streaming research events (SSE/WebSocket)
- Richer per-stage telemetry
- Brain UI component modularization
- Additional multimodal workflows

---

## License

[MIT License](LICENSE) — Copyright (c) 2026 Arman Ojha
