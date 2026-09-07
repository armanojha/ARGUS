# ARGUS Architecture

## System Overview

ARGUS is an evidence-grounded AI research system. It retrieves evidence, verifies claims, detects conflicts, and generates answers grounded in source material.

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

## Query Flow

When a user submits a research question:

1. **Query Analysis** — Detects question pattern (comparison, factual, conflict, summary), extracts entities, identifies time references
2. **Research Planning** — Generates sub-queries, selects retrieval strategy, sets iteration and token budgets
3. **Hybrid Retrieval** — BM25 lexical search + FAISS dense vector search, fused with configurable weights
4. **Reranking** — Scores candidates by relevance to the specific query
5. **Evidence Verification** — Checks whether evidence actually supports planned claims
6. **Conflict Detection** — Identifies contradictions across sources with temporal/entity/unit awareness
7. **Grounded Synthesis** — Generates answer from verified evidence with inline citations
8. **Answer Quality** — Validates grounding, checks citation accuracy

## Retrieval Architecture

### Hybrid Search

ARGUS combines two retrieval methods:

- **BM25** (lexical) — Term-frequency matching, good for exact keyword queries
- **FAISS** (dense) — Vector similarity via sentence-transformers embeddings, good for semantic queries

Fusion is configurable per query pattern via `configs/retrieval_policy.yaml`:

| Pattern | BM25 Weight | Dense Weight |
|---------|-------------|--------------|
| factual | 0.6 | 0.4 |
| comparison | 0.5 | 0.5 |
| conflict | 0.4 | 0.6 |
| summary | 0.3 | 0.7 |

### Multi-Query Retrieval

Complex queries are decomposed into sub-queries, each targeting specific evidence needs. Sub-queries run in parallel against the hybrid index.

### Reranking

A pluggable reranker interface scores retrieved evidence by relevance. The default `NoOpReranker` passes through BM25+FAISS scores. Any cross-encoder or learned reranker can be swapped in.

## Evidence Verification

Every claim in the answer is checked against retrieved evidence:

- **Supported** — Evidence directly backs the claim
- **Partial** — Some evidence supports, some is missing
- **Contradicted** — Evidence conflicts with the claim
- **Unsupported** — No relevant evidence found

Verification produces confidence scores for evidence coverage, source quality, cross-source agreement, and temporal relevance.

## Conflict Detection

ARGUS detects contradictions across evidence sources:

### Conflict Types

| Type | Description |
|------|-------------|
| `GENUINE_CONTRADICTION` | Sources disagree on the same claim |
| `DIFFERENT_TIMEFRAME` | Data from different periods |
| `DIFFERENT_SOURCE` | Different methodologies or scopes |
| `POSSIBLE_CONTRADICTION` | Uncertain — needs human review |
| `IRRELEVANT_DIFFERENCE` | Not a real conflict |

### Query-Aware Filtering

When `ARGUS_CONFLICT_FILTERING_ENABLED=true`:

- Temporal differences are filtered if the query asks for current data
- Low-confidence conflicts are suppressed
- Metric-relevance filtering ensures only query-relevant conflicts surface

## Synthesis

Grounded synthesis generates the answer from verified evidence:

- Every claim must cite a source
- Conflicts are acknowledged, not hidden
- Degraded mode activates when providers fail (evidence shown without full synthesis)
- Citation validation ensures referenced sources exist

## Brain UI

The Brain UI is a single-file HTML/JS/D3.js application served at `/brain`.

- **Pipeline view** — Interactive research flow visualization
- **Architecture view** — Component diagram with explanations
- **Knowledge graph** — Force-directed evidence graph
- **Research chat** — Q&A interface with citations
- **Demo mode** — Pre-recorded research result

See [BRAIN_UI.md](BRAIN_UI.md) for detailed documentation.

## Configuration

| Component | Config File | Key Settings |
|-----------|------------|--------------|
| Providers | `configs/providers.yaml` | API endpoints, model names, rate limits |
| Model Policy | `configs/model_policy.yaml` | Call-type → model routing |
| Retrieval | `configs/retrieval_policy.yaml` | Fusion weights per query pattern |
| Obsidian | `configs/obsidian.yaml` | Vault path, write-back root |
| Environment | `.env` | API keys, feature flags |

## Provider Routing

The LLM Gateway routes calls by type:

| Call Type | Purpose | Typical Provider |
|-----------|---------|-----------------|
| `query_analysis` | Pattern detection | Groq, Gemini |
| `research_planning` | Sub-query generation | Groq, Gemini |
| `evidence_extraction` | Evidence summarization | Groq, Cerebras |
| `synthesis` | Answer generation | NVIDIA NIM, Groq |
| `verification` | Claim checking | Gemini, Cerebras |

Fallback chains ensure resilience. When a provider is rate-limited, the next provider in the chain is used.

## Known Limitations

- No streaming research events (synchronous API only)
- Per-stage timing not exposed by backend
- Brain UI is a single HTML file (not componentized)
- Provider instability affects live E2E benchmarks
- PaddleOCR requires Python 3.11 (isolated `.venv-ocr`)
