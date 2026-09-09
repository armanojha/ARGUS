# ARGUS

### An Experimental RAG Research Project

A personal exploration of Retrieval-Augmented Generation techniques, implementing and combining methods from research papers, open-source projects, and hands-on experimentation. This is the foundation I built to understand how different RAG components work together.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-1025%20passed-brightgreen.svg)](#testing)

---

## Why I Built This

I spent months reading papers, exploring open-source RAG implementations, and experimenting with different techniques. But reading about RAG and actually building one are different things. I wanted to understand:

- How hybrid retrieval (BM25 + dense vectors) actually performs in practice
- What happens when retrieved evidence contradicts itself
- How to make the research process visible instead of a black box
- Whether multi-agent systems can improve answer quality

ARGUS is my answer to these questions. It's not a production system — it's a research sandbox where I implemented and tested different approaches to see what works.

## What's Inside

This project combines techniques from multiple papers and projects:

| Component | What I Implemented | Inspired By |
|-----------|-------------------|-------------|
| **Hybrid Retrieval** | BM25 lexical + FAISS dense search with configurable fusion weights | Traditional IR + dense retrieval research |
| **Iterative Research** | Multi-pass retrieval with evidence sufficiency assessment | Self-RAG, Corrective-RAG papers |
| **Query Classification** | Routes queries to 10 retrieval patterns (exact_term, conceptual, comparative, etc.) | Query understanding research |
| **Conflict Detection** | Pairwise contradiction detection with temporal, entity, and metric awareness | Fact verification literature |
| **Grounded Synthesis** | Answer generation with inline citations and conflict acknowledgment | RAG faithfulness research |
| **Evidence Traceability** | Every citation traces back to source document, chunk, and retrieval metadata | Provenance tracking research |

### Optional Features (disabled by default)

These are additional techniques I experimented with:

| Feature | Config Flag | What It Does |
|---------|-------------|--------------|
| **Adaptive Research** | `ARGUS_ADAPTIVE_RESEARCH_ENABLED=true` | Pattern-specific research policies |
| **Query-Aware Conflict Filtering** | `ARGUS_CONFLICT_FILTERING_ENABLED=true` | Suppresses irrelevant conflicts based on user intent |
| **Verified Synthesis** | `ARGUS_VERIFIED_SYNTHESIS_ENABLED=true` | Two-pass synthesis with claim verification |
| **Multi-Agent Debate** | `ARGUS_MULTIAGENT_ENABLED=true` | 5-agent debate system |
| **Persistent Memory** | `ARGUS_MEMORY_ENABLED=true` | 6-layer memory with SQLite storage |
| **Obsidian Integration** | `ARGUS_OBSIDIAN_ENABLED=true` | Vault-aware research and write-back |

## The Brain UI

I built an interactive interface to visualize what the system is doing at each step. You can see:

- **Pipeline View** — Research flow showing each processing stage
- **Node Inspector** — Click any node to see what it did and what it produced
- **Evidence Traceability** — Trace from any citation back to source document
- **Conflict Visualization** — See contradiction types and confidence levels
- **Architecture View** — Interactive component diagram
- **Knowledge Graph** — Entity and claim graph with multi-hop traversal

**Open:** `http://localhost:8000/brain`

## Architecture

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
Research Sufficiency    — Evaluate whether enough evidence has been gathered
    ↓
Grounded Synthesis      — Answer generation with citations, conflict acknowledgment
    ↓
Answer + Citations      — With inline references and source trail
    ↓
Brain UI                — Interactive visualization of the entire pipeline
```

## What I Learned

### Hybrid Retrieval Works Better Than Either Method Alone

BM25 is great for exact keyword matches, while dense vectors excel at semantic queries. Combining them with configurable weights lets each method handle what it's best at. The fusion weights matter — I spent time tuning them for different query patterns.

### Contradictions Are Common and Hard to Detect

Real-world documents often contain conflicting information. My contradiction detection uses regex-based claim extraction and pairwise comparison. It's not perfect (can't detect paraphrased contradictions), but it catches most numerical and temporal conflicts.

### Making the Process Visible Builds Trust

The biggest insight: when you can see what the system retrieved, how it verified claims, and why it made certain decisions, you trust the answer more. The Brain UI isn't just cosmetic — it's essential for understanding and debugging.

### Multi-Agent Systems Are Interesting But Complex

I implemented a 5-agent debate system. It's fascinating to watch agents disagree, but the complexity overhead is significant. For most queries, the single-agent path with adaptive research performs well enough.

## Honest Limitations

This is an experimental project. Here's what it doesn't do well:

- **No real-time learning** — The system doesn't improve from user feedback
- **Limited semantic understanding** — Conflict detection is regex-based, not truly semantic
- **No production hardening** — No authentication, rate limiting is basic, no horizontal scaling
- **Embedding quality depends on the model** — Using sentence-transformers/all-MiniLM-L6-v2 (fast but not the most accurate)
- **Memory system is basic** — SQLite with SQL LIKE search, not vector similarity

## Getting Started

### Prerequisites

- Python 3.11+
- An API key from at least one LLM provider (Groq, Google Gemini, Cerebras, Z.ai, Zen, or NVIDIA NIM)

### Installation

```bash
git clone https://github.com/armanojha/ARGUS.git
cd ARGUS
python -m venv .venv
.\.venv\Scripts\activate  # Windows
pip install -e ".[core,retrieval,graph,dev-test]"
```

### Configuration

```bash
cp .env.example .env
# Edit .env and add your API keys
```

### Running

```bash
# Start the API server
uvicorn app.api.main:app --reload

# Open the Brain UI
# http://localhost:8000/brain
```

### Demo Mode

The Brain UI includes a demo mode with pre-recorded data. No API key required — just click "Load Demo" on the pipeline view.

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=app --cov-report=term-missing
```

## Tech Stack

| Component | Technology |
|-----------|------------|
| **Backend** | Python 3.11, FastAPI, LangGraph |
| **Retrieval** | BM25 (rank-bm25), FAISS (sentence-transformers) |
| **Embeddings** | sentence-transformers/all-MiniLM-L6-v2 |
| **LLM Gateway** | 6 providers (Groq, Gemini, Cerebras, Z.ai, Zen, NVIDIA NIM) |
| **Storage** | SQLite (evidence, memory), NetworkX (knowledge graph) |
| **Frontend** | Vanilla JS, D3.js, Canvas |
| **Testing** | pytest, ruff |
| **CI** | GitHub Actions (Python 3.11/3.12/3.13 matrix) |

## Project Structure

```
ARGUS/
├── app/
│   ├── api/               # FastAPI routes
│   ├── config.py          # Configuration management
│   ├── evidence/          # SQLite evidence store
│   ├── graph/             # NetworkX knowledge graph
│   ├── ingestion/         # Document processing pipeline
│   ├── llm_gateway/       # Multi-provider LLM router
│   ├── memory/            # 6-layer persistent memory
│   ├── orchestration/     # LangGraph research state machine
│   ├── retrieval/         # Hybrid BM25 + FAISS retrieval
│   ├── ui/brain/          # Brain UI (single HTML file)
│   └── verification/      # Claim verification engine
├── benchmarks/            # Evaluation scripts and results
├── configs/               # Provider and policy configurations
├── tests/                 # Test suite
├── docs/                  # Documentation
└── knowledge_base/        # Your document corpus
```

## Documentation

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — Detailed architecture
- [docs/CORRECTNESS.md](docs/CORRECTNESS.md) — Correctness evolution
- [CHANGELOG.md](CHANGELOG.md) — Version history
- [CONTRIBUTING.md](CONTRIBUTING.md) — How to contribute

## License

MIT License — see [LICENSE](LICENSE).

## Acknowledgments

This project was inspired by:

- **RAG papers**: Self-RAG, Corrective-RAG, RETRO, Atlas
- **Open-source projects**: LangChain, LlamaIndex, Haystack
- **The research community** for sharing knowledge and techniques

---

*Built as a learning project to understand RAG systems from the ground up.*
