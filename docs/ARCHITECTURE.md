# ARGUS Architecture

## System Overview

ARGUS is an evidence-grounded AI research system. It retrieves evidence, verifies claims, detects conflicts, and generates answers grounded in source material. Every pipeline stage is observable through the Brain UI.

```
┌─────────────────────────────────────────────────────────────┐
│                      ARGUS Brain UI                         │
│  Pipeline View · Architecture View · Evidence Trail · Demo  │
└──────────────────────────┬──────────────────────────────────┘
                           │ HTTP API
┌──────────────────────────▼──────────────────────────────────┐
│                     FastAPI Application                      │
│  /api/v1/query · /api/v1/retrieve · /api/v1/verify          │
│  /api/v1/brain/* · /api/v1/knowledge-base/*                  │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│                   Orchestration Engine                       │
│              LangGraph State Machine                         │
│  Query → Analysis → Plan → Retrieve → Assess → Synthesize   │
└──────┬───────────┬───────────┬───────────┬──────────────────┘
       │           │           │           │
┌──────▼──┐ ┌──────▼──┐ ┌──────▼──┐ ┌─────▼───┐
│Retrieval│ │Evidence │ │  LLM    │ │Memory   │
│ Hybrid  │ │  Graph  │ │ Gateway │ │ 6-Layer │
│BM25+FAISS│ │NetworkX│ │Multi-   │ │SQLite   │
│         │ │         │ │Provider │ │         │
└─────────┘ └─────────┘ └─────────┘ └─────────┘
```

## Pipeline Flow

When a user submits a research question:

### 1. Query Analysis

- **Purpose:** Detect question pattern, extract entities, identify time references
- **Input:** Raw user query string
- **Output:** `QueryAnalysis` with pattern, entities, time window, risk level
- **Implementation:** LLM-based classification into 18+ patterns (factual, comparison, causal, procedural, multi-hop, conflict, absent-info, adversarial, etc.)
- **Key file:** `app/orchestration/nodes.py` → `make_analyze_node()`

### 2. Research Policy

- **Purpose:** Select retrieval strategy, set iteration and token budgets
- **Input:** `QueryAnalysis` from step 1
- **Output:** `ResearchPlan` with sub-queries, strategy, budget
- **Implementation:** LLM-based planning with budget clamping
- **Key file:** `app/orchestration/nodes.py` → `make_plan_node()`

### 3. Evidence Planning

- **Purpose:** Decompose complex queries into sub-queries, each targeting specific evidence needs
- **Input:** `ResearchPlan`
- **Output:** List of sub-queries with target evidence types
- **Implementation:** Sub-query generation with token budget allocation

### 4. Hybrid Retrieval

- **Purpose:** Retrieve relevant evidence from the corpus
- **Input:** Sub-queries from planning
- **Output:** Ranked evidence chunks with source metadata
- **Implementation:** BM25 lexical search + FAISS dense vector search with configurable fusion weights
- **Key files:** `app/retrieval/hybrid.py`, `app/retrieval/bm25.py`, `app/retrieval/vector.py`

### 5. Reranking

- **Purpose:** Score candidates by relevance to the specific query
- **Input:** Retrieved evidence chunks
- **Output:** Re-ranked evidence
- **Implementation:** Pluggable reranker interface (NoOp default, any cross-encoder can be swapped in)
- **Key file:** `app/reranking/reranker.py`

### 6. Evidence Selection

- **Purpose:** Select minimal high-coverage evidence subset for LLM context
- **Input:** Re-ranked evidence
- **Output:** Selected evidence within token budget
- **Implementation:** Semantic dedup, diversity selection, token budget enforcement
- **Key file:** `app/retrieval/evidence_selector.py`

### 7. Evidence Verification

- **Purpose:** Check whether evidence actually supports planned claims
- **Input:** Selected evidence + research plan
- **Output:** Verification status (Supported/Partial/Contradicted/Unsupported) with confidence scores
- **Implementation:** Deterministic claim-support checking against evidence chunks
- **Key file:** `app/verification/engine.py`

### 8. Contradiction Detection

- **Purpose:** Identify contradictions across evidence sources
- **Input:** All evidence chunks
- **Output:** Conflict signals with type, confidence, entity overlap, metric overlap
- **Implementation:** Deterministic pairwise analysis (negation pairs, numerical discrepancies, temporal context)
- **Key file:** `app/orchestration/nodes.py` → `_detect_contradictions()`

### 9. Query-Aware Filtering

- **Purpose:** Suppress irrelevant conflicts based on what the user is asking about
- **Input:** Conflict signals + user query
- **Output:** Filtered conflicts relevant to the query
- **Implementation:** Metric-relevance filtering, temporal filtering, confidence thresholds
- **Key file:** `app/orchestration/nodes.py` → `filter_contradictions_by_query()`

### 10. Research Sufficiency

- **Purpose:** Evaluate whether enough evidence has been gathered
- **Input:** Evidence assessment, conflict signals, iteration count
- **Output:** Stop/go decision
- **Implementation:** 5 stopping conditions (budget exhausted, claims supported, no contradictions, negligible gain, user early stop)
- **Key file:** `app/orchestration/stopping.py`

### 11. Grounded Synthesis

- **Purpose:** Generate answer from verified evidence with citations
- **Input:** Evidence, verification results, conflict signals
- **Output:** Answer text with inline citations
- **Implementation:** LLM-based synthesis with evidence selection, conflict acknowledgment, safe degradation
- **Key file:** `app/orchestration/nodes.py` → `make_synthesize_node()`

### 12. Answer Quality

- **Purpose:** Validate grounding, check citation accuracy
- **Input:** Synthesized answer + evidence
- **Output:** Grounded answer with validated citations
- **Implementation:** Deterministic citation extraction and grounding checks
- **Key file:** `app/orchestration/nodes.py` → `check_claim_grounding()`, `extract_cited_indices()`

## Retrieval Architecture

### Hybrid Search

ARGUS combines two retrieval methods:

- **BM25** (lexical) — Term-frequency matching via `rank-bm25`, good for exact keyword queries
- **FAISS** (dense) — Vector similarity via `faiss-cpu` + `sentence-transformers`, good for semantic queries

Fusion is configurable per query pattern via `configs/retrieval_policy.yaml`:

| Pattern | BM25 Weight | Dense Weight |
|---------|-------------|--------------|
| factual | 0.6 | 0.4 |
| comparison | 0.5 | 0.5 |
| conflict | 0.4 | 0.6 |
| summary | 0.3 | 0.7 |

### Adaptive Policy Router

The retrieval policy router classifies incoming questions into one of 18+ patterns and selects the optimal retrieval mix, method, and fusion strategy for each.

**Classification patterns:** factual, numerical, exact_term, comparative, causal, procedural, long_report, historical, fresh_missing, entity_relationship, conflict, multi_hop, complex_research, absent_info, adversarial, multimodal, conceptual, summary

**Key file:** `app/retrieval/router.py` → `RetrievalPolicyRouter`

### Multi-Query Retrieval

Complex queries are decomposed into sub-queries, each targeting specific evidence needs. Sub-queries run in parallel against the hybrid index with bounded concurrency.

**Key file:** `app/retrieval/multi_query.py`

### Evidence Selection

After retrieval, the evidence selector performs:
- Semantic deduplication (removes near-duplicate chunks)
- Diversity selection (ensures coverage across different aspects)
- Token budget enforcement (fits evidence into LLM context window)

**Key file:** `app/retrieval/evidence_selector.py`

## Evidence Verification

Every claim in the answer is checked against retrieved evidence:

| Status | Meaning |
|--------|---------|
| **Supported** | Evidence directly backs the claim |
| **Partial** | Some evidence supports, some is missing |
| **Contradicted** | Evidence conflicts with the claim |
| **Unsupported** | No relevant evidence found |

Verification produces confidence scores for:
- Evidence coverage
- Source quality
- Cross-source agreement
- Temporal relevance

**Key files:** `app/verification/engine.py`, `app/verification/confidence.py`

## Contradiction Detection

ARGUS detects contradictions across evidence sources using deterministic pairwise analysis:

### Detection Methods

1. **Negation pairs** — Detects opposing claims (supports vs contradicts) with topic coherence and entity overlap requirements
2. **Numerical discrepancies** — Compares extracted numbers near shared metrics with unit normalization ($, %, billion, million)
3. **Temporal context** — Extracts years and date ranges to distinguish historical differences from genuine contradictions

### Conflict Types

| Type | Description |
|------|-------------|
| `GENUINE_CONTRADICTION` | Sources disagree on the same claim, same entity, same timeframe |
| `DIFFERENT_TIMEFRAME` | Data from different time periods |
| `DIFFERENT_SOURCE` | Different methodologies or scopes |
| `POSSIBLE_CONTRADICTION` | Uncertain — needs human review |
| `IRRELEVANT_DIFFERENCE` | Not a real conflict |

### Query-Aware Filtering

When `ARGUS_CONFLICT_FILTERING_ENABLED=true`:
- Temporal differences are filtered if the query asks for current data
- Low-confidence conflicts are suppressed
- Metric-relevance filtering ensures only query-relevant conflicts surface

**Key file:** `app/orchestration/nodes.py` → `_detect_contradictions()`, `filter_contradictions_by_query()`

## LLM Gateway

The LLM Gateway routes calls by type:

| Call Type | Purpose | Typical Provider |
|-----------|---------|-----------------|
| `query_analysis` | Pattern detection | Groq, Gemini |
| `research_planning` | Sub-query generation | Groq, Gemini |
| `evidence_extraction` | Evidence summarization | Groq, Cerebras |
| `synthesis` | Answer generation | NVIDIA NIM, Groq |
| `verification` | Claim checking | Gemini, Cerebras |

Fallback chains ensure resilience. When a provider is rate-limited, the next provider in the chain is used.

**Key files:** `app/llm_gateway/routing/router.py`, `app/llm_gateway/providers/`

## Memory System

Persistent 6-layer SQLite-backed memory architecture:

- Allows ARGUS to learn from previous research sessions
- Enhances future plans based on past research
- Supports memory promotion and versioning

**Key files:** `app/memory/store.py`, `app/memory/factory.py`

## Evidence Store

SQLite-backed evidence storage with WAL mode:

- Full CRUD for sources, documents, chunks
- Provenance tracking (source → document → chunk)
- Singleton pattern with lazy initialization

**Key file:** `app/evidence/store.py`

## Knowledge Graph

NetworkX MultiDiGraph backing the evidence graph:

- 6 node types: Entity, Claim, Event, Document, Chunk, Source
- 8 edge types: supports, contradicts, derived_from, relates_to, etc.
- Exported as JSON for Brain UI visualization

**Key files:** `app/graph/store.py`, `app/graph/extraction.py`

## Configuration

| Component | Config File | Key Settings |
|-----------|------------|--------------|
| Providers | `configs/providers.yaml` | API endpoints, model names, rate limits |
| Model Policy | `configs/model_policy.yaml` | Call-type → model routing |
| Retrieval | `configs/retrieval_policy.yaml` | Fusion weights per query pattern |
| Obsidian | `configs/obsidian.yaml` | Vault path, write-back root |
| Environment | `.env` | API keys, feature flags |

## Known Limitations

- No streaming research events (synchronous API only)
- Per-stage timing not exposed by backend
- Brain UI is a single HTML file (not componentized)
- Provider instability affects live E2E benchmarks
- PaddleOCR requires Python 3.11 (isolated `.venv-ocr`)

See [LIMITATIONS.md](LIMITATIONS.md) for the complete list.
