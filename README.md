# ARGUS

### Evidence-Grounded Research Intelligence

An iterative RAG system that makes retrieval, evidence verification, conflicts, and answer provenance visible.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-915-brightgreen.svg)](#testing)
[![Lint](https://img.shields.io/badge/lint-ruff%20clean-brightgreen.svg)](#testing)

---

## Why ARGUS Exists

Traditional RAG systems retrieve documents and generate answers. The research process is hidden. You get an answer but cannot see:

- What evidence was actually found
- How it was verified against the query
- Whether sources disagree with each other
- Why this answer emerged from the evidence

ARGUS makes the research process observable through the **Brain UI** — an interactive interface that exposes pipeline stages, evidence traces, conflict signals, and synthesis decisions.

## What ARGUS Actually Does

| Capability | What It Does | Status |
|------------|--------------|--------|
| **Hybrid Retrieval** | BM25 lexical + FAISS dense vector search with configurable fusion weights | Core |
| **Iterative Research** | Multi-pass retrieval with evidence sufficiency assessment | Core |
| **Query Classification** | Routes queries to 10 configured retrieval patterns (exact_term, conceptual, comparative, etc.) | Core |
| **Conflict Detection** | Pairwise contradiction detection with temporal, entity, and metric awareness | Core |
| **Grounded Synthesis** | Answer generation with inline citations, conflict acknowledgment, and safe degradation | Core |
| **Evidence Traceability** | Every citation traces back to source document, chunk, and retrieval metadata | Core |
| **Brain UI** | Interactive pipeline visualization, node inspection, evidence tracing, demo mode | Core |

### Optional Features (disabled by default)

These features are implemented but not active unless explicitly enabled:

| Feature | Config Flag | What It Does |
|---------|-------------|--------------|
| **Adaptive Research** | `ARGUS_ADAPTIVE_RESEARCH_ENABLED=true` | Pattern-specific research policies (5 patterns have custom policies; 15 use defaults) |
| **Query-Aware Conflict Filtering** | `ARGUS_CONFLICT_FILTERING_ENABLED=true` | Suppresses irrelevant conflicts based on user intent |
| **Verified Synthesis** | `ARGUS_VERIFIED_SYNTHESIS_ENABLED=true` | Two-pass synthesis with claim verification |
| **Multi-Agent Debate** | `ARGUS_MULTIAGENT_ENABLED=true` | 5-agent debate system (Researcher, Skeptic, Alternative, Verifier, Judge) |
| **Persistent Memory** | `ARGUS_MEMORY_ENABLED=true` | 6-layer memory with SQLite storage (SQL LIKE search, not vector) |
| **Obsidian Integration** | `ARGUS_OBSIDIAN_ENABLED=true` | Vault-aware research and write-back |

## The Brain UI

The Brain is the primary interface. It transforms ARGUS from a black-box RAG system into an inspectable research system.

**Open:** `http://localhost:8000/brain`

### What You Can Inspect

- **Pipeline View** — Research flow showing each processing stage as a clickable node
- **Node Inspector** — Click any node to see what it did, what it received, and what it produced
- **Evidence Traceability** — Trace from any citation back to source document, chunk, and retrieval score
- **Conflict Visualization** — See contradiction types, confidence levels, entity overlap, and resolution
- **Architecture View** — Interactive component diagram explaining what each module does
- **Knowledge Graph** — Entity and claim graph with multi-hop traversal
- **Demo Mode** — Pre-recorded research result with full pipeline visualization (no API key required)

### How the UI Transforms ARGUS

| Without Brain UI | With Brain UI |
|------------------|---------------|
| "Here's an answer" | "Here's the answer, here's where it came from, here's what disagrees" |
| Black-box retrieval | Click to see which chunks were retrieved and why |
| Hidden conflicts | Conflict type, confidence, entities, and resolution visible |
| Trust the system | Inspect every decision, verify claims against evidence |

## End-to-End Architecture

```
User Query
    ↓
Query Analysis          — Pattern detection, entity extraction, time window
    ↓
Research Policy         — Strategy selection, iteration budget, method dispatch
    ↓
Evidence Planning       — Sub-query generation, token budget allocation
    ↓
Hybrid Retrieval        — BM25 (lexical) + FAISS (dense) with fusion
    ├── BM25            — Term-frequency matching for exact keywords
    └── FAISS           — Vector similarity for semantic queries
    ↓
Multi-Query Retrieval   — Parallel sub-queries with concurrent execution
    ↓
Reranking               — Pluggable quality scoring and relevance ranking
    ↓
Evidence Selection      — Semantic dedup, diversity selection, token budget enforcement
    ↓
Evidence Verification   — Claim-support checking, confidence scoring, gap detection
    ↓
Contradiction Detection — Pairwise evidence comparison, numerical/temporal/entity analysis
    ↓
Query-Aware Filtering   — Suppress irrelevant conflicts based on user intent
    ↓
Research Sufficiency    — Evaluate whether enough evidence has been gathered
    ↓
Grounded Synthesis      — Answer generation with citations, conflict acknowledgment
    ↓
Answer Quality          — Citation grounding validation, evidence coverage check
    ↓
Answer + Citations      — With inline references and source trail
    ↓
Brain UI                — Interactive visualization of the entire pipeline
```

## Retrieval Architecture

### Hybrid Search

ARGUS combines two retrieval methods:

- **BM25** — Term-frequency matching, good for exact keyword queries
- **FAISS** — Vector similarity via sentence-transformers embeddings, good for semantic queries

Fusion weights are configurable per query pattern via `configs/retrieval_policy.yaml`:

| Query Pattern | BM25 Weight | Dense Weight |
|---------------|-------------|--------------|
| Exact term | 0.7 | 0.3 |
| Conceptual | 0.3 | 0.7 |
| Comparative | 0.5 | 0.5 |
| Causal | 0.4 | 0.6 |
| Procedural | 0.6 | 0.4 |

### Retrieval Policy

The retrieval policy router classifies incoming questions into 10 configured patterns (exact_term, conceptual, entity_relationship, historical, long_report, fresh_missing, multimodal, comparative, causal, procedural) and selects the optimal retrieval mix, method, and fusion strategy for each. An additional 10 evaluation patterns (conflict, multi_hop, complex_research, etc.) are defined in the enum but use default retrieval settings.

### Evidence Selection

After retrieval, the evidence selector performs:
- Semantic deduplication (removes near-duplicate chunks)
- Diversity selection (ensures coverage across different aspects)
- Token budget enforcement (fits evidence into LLM context window)

## Evidence Verification

ARGUS uses two verification approaches:

1. **Deterministic checks** — Regex-based claim extraction and evidence coverage scoring (no LLM required)
2. **LLM-assisted verification** — When a provider is available, the system asks an LLM to evaluate whether claims are supported by evidence

| Status | Meaning |
|--------|---------|
| **Supported** | Evidence directly backs the claim |
| **Partial** | Some evidence supports, some is missing |
| **Contradicted** | Evidence conflicts with the claim |
| **Unsupported** | No relevant evidence found |

Verification produces confidence scores for evidence coverage, source quality, and cross-source agreement. Temporal relevance is currently a stub (returns 1.0).

## Contradiction Detection

ARGUS detects contradictions across evidence sources using deterministic pairwise analysis:

**How it works:** Regex-based claim extraction → pairwise comparison → same category + same unit group + different values = contradiction. This is keyword/regex matching, not semantic understanding. It cannot detect paraphrased contradictions or contradictions requiring contextual reasoning.

### Conflict Types

| Type | Description |
|------|-------------|
| `GENUINE_CONTRADICTION` | Sources disagree on the same claim, same entity, same timeframe |
| `DIFFERENT_TIMEFRAME` | Data from different time periods (not a real conflict) |
| `DIFFERENT_SOURCE` | Different methodologies or scopes (not a real conflict) |
| `POSSIBLE_CONTRADICTION` | Uncertain — needs human review |
| `IRRELEVANT_DIFFERENCE` | Not a real conflict |

### Detection Methods

- **Negation pairs** — Detects opposing claims (supports vs contradicts) with topic coherence and entity overlap requirements
- **Numerical discrepancies** — Compares extracted numbers near shared metrics with unit normalization ($, %, billion, million)
- **Temporal context** — Extracts years and date ranges to distinguish historical differences from genuine contradictions
- **Entity overlap** — Requires shared entity context before flagging conflicts

### Query-Aware Filtering

When `ARGUS_CONFLICT_FILTERING_ENABLED=true`:
- Temporal differences are filtered if the query asks for current data
- Low-confidence conflicts are suppressed unless query asks about conflicts
- Metric-relevance filtering ensures only query-relevant conflicts surface

See [docs/CORRECTNESS.md](docs/CORRECTNESS.md) for the evolution of the correctness system.

## Research Orchestration

ARGUS uses a LangGraph state machine for research orchestration:

### Pipeline Modes

- **Fast Path** — Simple factual queries skip planning and go directly to retrieval → synthesis
- **Normal Path** — Complex queries go through analysis → plan → retrieve → assess → stop-check loop

### What "Iterative" Actually Means

The core loop is: **retrieve → assess evidence sufficiency → retrieve again if needed**. This is iterative RAG, not autonomous research. The system does not dynamically change search strategy, query decomposition, retrieval method, source priorities, evidence budget, verification depth, or model choice based on what was discovered.

When enabled, the optional **Adaptive Research** feature adds pattern-specific policies for 5 query types (conflict, multi_hop, complex_research, absent_info, simple_lookup). The remaining 15 patterns use a generic default policy.

### Stopping Conditions

The research loop terminates when:
1. User requests early stop
2. Token budget exhausted
3. Negligible evidence gain between iterations
4. All claims are supported by evidence
5. No unresolved contradictions remain

### Memory Integration

Persistent memory architecture with SQLite storage. Uses SQL LIKE for text search (not vector similarity). Currently disabled by default and not part of the default research path.

## Grounded Synthesis

The synthesis stage generates answers that are:

- **Grounded** — Synthesis is designed to tie claims to retrieved evidence with inline citations
- **Conflict-aware** — Contradictions are acknowledged, not hidden
- **Degraded-safe** — When providers fail, evidence is shown without full synthesis
- **Citation-validated** — Referenced sources are verified to exist

## Evaluation Methodology

### What Has Been Validated

| Component | Validation Method | Status |
|-----------|------------------|--------|
| Hybrid retrieval | Deterministic tests with mock evidence | Validated |
| Query classification | Pattern-matching unit tests | Validated |
| Evidence selection | Token budget and diversity tests | Validated |
| Conflict detection | 8-case benchmark (all scenarios) | 8/8 pass |
| Query-aware filtering | False positive reduction tests | Validated |
| Contradiction normalization | Unit normalization regression tests | Validated |
| Grounded synthesis | Citation extraction and grounding tests | Validated |
| Evidence traceability | End-to-end citation chain tests | Validated |
| Multi-model routing | Provider fallback and quota tests | Validated |
| Memory architecture | Layer promotion and query tests | Validated |

### Honest Limitations of Evaluation

Some benchmarks were affected by provider instability and could not be run cleanly:
- Phase 36 (Clean Provider Benchmark) — all runs had provider fallbacks
- Phase 37 (LLM Call Minimization) — infrastructure-blocked
- Phase 40 (Conflict-Aware Synthesis) — provider contamination

These results were not used to justify production changes. See [docs/EVALUATION.md](docs/EVALUATION.md) for full methodology.

## Current Results

| Metric | Value |
|--------|-------|
| Test suite | **915 tests** (906 + 9 semantic-verification), 26 skipped |
| Contradiction benchmark | **8/8 cases** on synthetic 1-3 sentence snippets (not real documents) |
| Backend status | **Frozen** — no modifications since Phase 43 |
| Brain UI | Complete — all spec requirements implemented |
| Conflict detection | Implemented, feature-flagged, validated on synthetic data |
| Evidence verification | Regex-based deterministic + LLM-assisted (when provider available) |

### What Has NOT Been Validated

- End-to-end retrieval recall on real document corpora
- Claim grounding accuracy on complex multi-source queries
- Adversarial or prompt-injection resistance
- Latency and cost under production load
- Memory system effectiveness (disabled by default, SQL LIKE search only)

## Tech Stack

| Layer | Technology |
|-------|------------|
| **API** | FastAPI, Pydantic, structlog |
| **Orchestration** | LangGraph state machine |
| **Retrieval** | BM25 (rank-bm25), FAISS (faiss-cpu), sentence-transformers |
| **Embeddings** | all-MiniLM-L6-v2 (default), BGE-M3 (experimental) |
| **LLM Gateway** | Multi-provider router (Groq, Gemini, Cerebras, Z.ai, NVIDIA NIM, Zen) |
| **Evidence Store** | SQLite with WAL mode |
| **Knowledge Graph** | NetworkX MultiDiGraph |
| **Memory** | SQLite-backed 6-layer architecture |
| **Verification** | Deterministic checks + LLM-based claim verification |
| **Brain UI** | Single-file HTML/JS/D3.js/Canvas (no build step) |
| **Testing** | pytest, 956 tests across 71 test files |
| **CI** | GitHub Actions (Python 3.11/3.12/3.13, ruff lint) |

## Project Structure

```
ARGUS/
├── app/
│   ├── api/              FastAPI routes (health, retrieval, orchestration, brain, knowledge-base)
│   ├── config.py         Pydantic Settings with env-var loading (ARGUS_ prefix)
│   ├── evaluation/       Answer quality evaluation
│   ├── evidence/         SQLite evidence store with provenance tracking
│   ├── graph/            NetworkX evidence graph (entities, claims, events)
│   ├── ingestion/        Document ingestion pipeline, OCR, multimodal processing
│   ├── integrations/     Obsidian vault integration
│   ├── llm_gateway/      Multi-model LLM router, providers, health tracking, quota
│   ├── memory/           6-layer SQLite memory with promotion and versioning
│   ├── orchestration/    LangGraph research state machine, nodes, prompts, models
│   ├── reranking/        Pluggable reranker interface
│   ├── retrieval/        Hybrid BM25 + FAISS, policy router, evidence selector
│   ├── ui/brain/         Brain UI (single-file HTML/JS/D3.js)
│   └── verification/     Claim verification engine, contradiction detection
├── benchmarks/           Phase reports, evaluation data, benchmark harness
├── configs/              Provider, model policy, retrieval policy, Obsidian config
├── docs/                 Architecture docs, workflow visualization, guides
├── knowledge_base/       User document corpus (created at runtime, not in repo)
├── scripts/              Ingestion, OCR, test runners, diagnostic tools
├── tests/                Test suite (mirrors app/ structure, 71 test files)
├── .env.example          Environment variable template
├── pyproject.toml        Build config, dependencies, tool settings
└── LICENSE               MIT
```

## Installation

### Prerequisites

- Python 3.11+ (developed against 3.11, tested on 3.11/3.12/3.13)
- At least one LLM provider API key (Groq recommended — free tier available)
- Git

### Setup

```bash
# Clone the repository
git clone https://github.com/armanojha/ARGUS.git
cd ARGUS

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\Activate   # Windows PowerShell

# Install dependencies
pip install -e ".[core,retrieval,graph,multimodal,dev-test]"
```

## Configuration

### Environment Variables

Copy the template and add your API key:

```bash
cp .env.example .env
# Edit .env — add at least one API key
```

| Variable | Description |
|----------|-------------|
| `GROQ_API_KEY` | Groq API key (free tier, recommended) |
| `GEMINI_API_KEY` | Google Gemini API key |
| `CEREBRAS_API_KEY` | Cerebras API key |
| `ZAI_API_KEY` | Z.ai (Zhipu AI) API key |
| `NVIDIA_NIM_API_KEY` | NVIDIA NIM API key |
| `OPENCODE_ZEN_API_KEY` | Zen (OpenCode) API key |

### Configuration Files

| File | Purpose |
|------|---------|
| `configs/providers.yaml` | LLM provider definitions (endpoints, models, rate limits) |
| `configs/model_policy.yaml` | Call-type → model routing (analysis, planning, synthesis, verification) |
| `configs/retrieval_policy.yaml` | 20 retrieval patterns with fusion weights |
| `configs/obsidian.yaml` | Obsidian vault integration settings |

### Feature Flags

| Flag | Default | Description |
|------|---------|-------------|
| `ARGUS_MEMORY_ENABLED` | `false` | Persistent memory system |
| `ARGUS_MULTIMODAL_ENABLED` | `false` | Multimodal document processing |
| `ARGUS_BGE_M3_ENABLED` | `false` | Experimental BGE-M3 embeddings |
| `ARGUS_CONFLICT_FILTERING_ENABLED` | `false` | Query-aware conflict filtering |
| `ARGUS_CONFLICT_SAFE_SYNTHESIS_ENABLED` | `false` | Conflict-aware synthesis rules |
| `ARGUS_MULTIAGENT_ENABLED` | `false` | Multi-agent debate |
| `ARGUS_ADAPTIVE_RESEARCH_ENABLED` | `false` | Adaptive research policy |

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

Models are assigned per call type via `configs/model_policy.yaml`. All assignments are explicit configuration — ARGUS never autonomously selects models.

## Running ARGUS

### Start the Server

```bash
uvicorn app.api.main:app --reload
```

Wait for the "Application startup complete" message.

### Open the Brain UI

```
http://localhost:8000/brain
```

### Try Demo Mode (No API Key)

1. Open `http://localhost:8000/brain`
2. Click **Load Demo** on the pipeline view
3. Explore the pre-recorded research result with full pipeline visualization

### Ask a Question

1. Click **Research** in the sidebar
2. Type a question
3. Press Enter or click Send
4. Click any node to inspect what happened at that stage

### Demo Questions

| Question | Demonstrates |
|----------|-------------|
| "Compare revenue growth of Acme and Globex in 2024 vs 2025" | Multi-source comparison, numerical evidence |
| "What evidence supports the claim that AI adoption is accelerating?" | Evidence gathering, verification |
| "Find conflicting information about climate change impacts" | Conflict detection, source disagreement |

### Ingest Your Own Documents

1. Place files (PDF, TXT, MD, CSV, XLSX) in `knowledge_base/`
2. Click **Knowledge Base** in the sidebar
3. Click **Re-ingest knowledge folder**
4. Or use the API:
   ```bash
   curl -X POST http://localhost:8000/api/v1/knowledge-base/ingest
   ```

### Health Check

```bash
curl http://localhost:8000/health
```

## Testing

```bash
# Run the full test suite
python -m pytest tests/ -v --tb=short

# Run with exclusions (if pymupdf not installed)
python -m pytest tests/ -q --ignore=tests/ingestion/test_multimodal.py --ignore=tests/ingestion/test_ocr_regression.py --ignore=tests/ingestion/test_paddle_ocr_runner.py

# Run contradiction benchmark
python benchmarks/phase43_benchmark.py
```

### Test Status

| Metric | Value |
|--------|-------|
| Tests passed | 914 |
| Tests skipped | 26 |
| Pre-existing failures | 2 |
| Known-flaky | 1 (`test_ingest_knowledge_base_idempotent_second_run` — passes standalone, fails intermittently in full-suite runs) |
| New regressions | 0 |

The 2 pre-existing failures are both FastAPI route tests:
- `test_telemetry_endpoints_query_integration` — FastAPI `_IncludedRouter` API change
- `test_verify_route_registered` — FastAPI `_IncludedRouter` API change

These are not caused by any Phase 43+ changes.

## Known Limitations

### Provider Infrastructure

- **Free-tier rate limits** — Rapid sequential queries exhaust rate limits. Demo Mode provides a reliable offline demonstration.
- **Provider instability** — Some providers have inconsistent availability. ARGUS routes through fallback chains.

### Backend

- **No streaming** — Backend processes queries synchronously. Progressive pipeline display is a UI simulation.
- **Per-stage timing not exposed** — Only aggregate telemetry (total duration, calls, tokens) is available.
- **No persistent research history** — Each query is stateless.

### Brain UI

- **Single-file architecture** — 3,673-line HTML file. Intentional for simplicity (no build step), but limits component reuse.
- **Canvas rendering** — Graph performance may degrade with thousands of nodes (current evidence graphs are typically tens to low hundreds).

### What ARGUS Does NOT Do

- **Real-time web search** — ARGUS queries a local document corpus, not the internet
- **Image understanding** — OCR extracts text from images, but ARGUS does not "see" images
- **Perfect accuracy** — ARGUS verifies evidence but can still make mistakes
- **Zero hallucinations** — Grounded synthesis reduces hallucination but does not eliminate it
- **Commercial production use** — ARGUS is a research project, not a production system

See [docs/LIMITATIONS.md](docs/LIMITATIONS.md) for the complete limitations document.

## Roadmap

### Now (Complete)

- Hybrid retrieval (BM25 + FAISS)
- Adaptive research policy with 20 query patterns
- Evidence verification with confidence scoring
- Contradiction detection with query-aware filtering
- Grounded synthesis with citations
- Multi-model provider routing with fallback
- Persistent memory architecture
- Brain UI with pipeline visualization

### Next

- Streaming research events (SSE/WebSocket)
- Richer per-stage telemetry
- Brain UI component modularization
- Stronger synthesis validation

### Later

- Advanced entity linking
- Graph-assisted retrieval
- Additional multimodal workflows
- Research session persistence

## Documentation

| Document | Description |
|----------|-------------|
| [Architecture](docs/ARCHITECTURE.md) | System architecture and component details |
| [Brain UI](docs/BRAIN_UI.md) | Interface guide and feature documentation |
| [Correctness](docs/CORRECTNESS.md) | Evidence verification and contradiction detection evolution |
| [Evaluation](docs/EVALUATION.md) | Benchmark methodology and honest results |
| [Engineering Journey](docs/ENGINEERING_JOURNEY.md) | Phase-by-phase development history |
| [Limitations](docs/LIMITATIONS.md) | Known limitations and honest status |
| [Quickstart](docs/QUICKSTART.md) | Get running in under 10 minutes |
| [Roadmap](docs/ROADMAP.md) | Future development plans |
| [Changelog](CHANGELOG.md) | Version history |
| [Contributing](CONTRIBUTING.md) | Contribution guidelines |
| [Security](SECURITY.md) | API key and data security guidance |

## Engineering Decisions

The backend is **frozen**. No modifications since Phase 43. This is intentional:

- Retrieval architecture is validated and tested
- Contradiction detection is benchmarked at 100% (8/8 cases)
- Evidence verification is deterministic and independently tested
- The Brain UI is complete

Future work focuses on documentation, presentation, and reproducibility — not more engineering.

## License

[MIT License](LICENSE) — Copyright (c) 2026 Arman Ojha
