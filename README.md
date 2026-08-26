# ARGUS

**Adaptive Autonomous Evidence & Reasoning Intelligence System**

A cloud-LLM, local-control-plane RAG system that treats difficult
questions as evidence-backed investigations: adaptive hybrid retrieval,
an evidence graph with temporal reasoning, closed-loop verification, and
(from Phase 05 onward) an Obsidian-integrated personal knowledge layer.


## Status

Phase 00.1 (repository + configuration) only. No API, no LLM gateway, no
retrieval yet — see the vault's phase files for what's next.

## Setup

Requires Python 3.11+.

```powershell
# 1. Clone the repository
git clone https://github.com/armanojha/ARGUS.git
cd ARGUS

# 2. Create and activate a virtual environment
python -m venv .venv
.\.venv\Scripts\Activate

# 3. Install dependencies
pip install -e ".[core,dev-test]"

# 4. Configure environment
copy .env.example .env
# Edit .env — fill in API keys as needed (GROQ_API_KEY for LLM gateway)

# 5. Run the tests
python -m pytest tests/ -v
# Or use the PowerShell test runner:
powershell -ExecutionPolicy Bypass -File scripts\run_tests.ps1
```

### Running the API

```powershell
# Start the server
uvicorn app.api.main:app --reload

# Verify it's running
curl http://localhost:8000/health
```

### Running live LLM tests

Live integration tests against the Groq API are skipped by default. To run them:

```powershell
# Set the environment variable and ensure GROQ_API_KEY is configured
$env:RUN_LIVE_LLM_TESTS = "1"
python -m pytest tests/test_llm_gateway_integration.py -v
```

## Development

The project follows a phased implementation model. Each phase has a specification in the vault (`E:\ARGUS_VAULT\phases/`) and is implemented, tested, reviewed, and committed as a unit.

**Workflow:**

1. Read the active phase specification
2. Implement against the phase's deliverables
3. Write tests that validate the phase's acceptance criteria
4. Run `pytest`, `ruff check`, and `mypy app/`
5. Update project state in the vault

**Code conventions:**

- `from __future__ import annotations` in all modules
- Modern type hints (`dict[str, Any]`, `str | None`)
- No hardcoded provider names, model IDs, or API keys
- LLMs are interpreters, not the database of record
- All evidence and provenance in deterministic stores

## Testing & Quality

Current test suite: **45 passed, 8 skipped** (skipped tests are live LLM integration tests, require `RUN_LIVE_LLM_TESTS=1` and a valid `GROQ_API_KEY`).

| Check | Status |
|-------|--------|
| `pytest tests/ -v` | 45 passed, 8 skipped |
| `ruff check .` | 2 lint suggestions (uncommitted code) |
| `mypy app/` | 3 type errors (uncommitted Phase 00.3 code) |

Tests cover: config loading, API health endpoint, error envelope behavior, request ID propagation, structured JSON logging, LLM gateway unit tests (mock provider), model policy, provider registry, and provider capabilities.

## Roadmap

```
Phase 00 — Foundation (00.1–00.4)         ← IN PROGRESS
  ├─ 00.1 Repository + Configuration       ✓ Complete
  ├─ 00.2 FastAPI Foundation               ✓ Complete
  ├─ 00.3 LLM Gateway Core                 ✓ Complete
  └─ 00.4 Testing Foundation               Next up

Phase 01 — Hybrid RAG (01.1–01.5)         ← NOT STARTED
  Evidence store (SQLite), BM25 + vector retrieval, embedding model

Phase 02 — Agentic RAG (02.1–02.5)        ← NOT STARTED
  LangGraph orchestration loop, query decomposition, retrieval planning

Phase 03 — Evidence Graph (03.1–03.4)     ← NOT STARTED
  Claim→source graph, temporal model, provenance tracking

Phase 04 — Verification (04.1–04.4)       ← NOT STARTED
  Contradiction detection, confidence scoring, revision cycles
  ← MVP CORE LOOP CLOSES HERE

Phase 05 — Obsidian Ingestion (05.1–05.4) ← NOT STARTED
  Minimal vault ingestion, incremental sync
  ← MVP BOUNDARY

Phase 06 — Adaptive Research Policy        ← POST-MVP
Phase 07 — Multi-Model Fabric              ← POST-MVP
Phase 08 — Memory & Self-Evolution         ← POST-MVP
Phase 09 — Obsidian Integration (full)     ← POST-MVP
Phase 10 — Multi-Agent Challenge           ← POST-MVP
Phase 11 — Multimodal Intelligence         ← POST-MVP
Phase 12 — Research UI + Evaluation        ← POST-MVP
```

## Limitations

This is an early-stage project. What does **not** exist yet:

- No retrieval pipeline (BM25, vector search, or hybrid)
- No evidence store or evidence graph
- No claim verification or contradiction detection
- No agentic orchestration loop
- No Obsidian vault integration
- No multi-provider LLM routing or intelligent model selection
- No UI or evaluation benchmarks
- No persistent memory or self-evolution
- No authentication, rate limiting, or production hardening

The LLM gateway currently supports only Groq as a wired provider. Other providers (Gemini, Cerebras) are configured as stubs but not implemented.

## Contributing

ARGUS is in active development. Contributions are welcome for:

- Implementing planned phases (see the roadmap)
- Adding new LLM providers to the gateway
- Improving test coverage
- Documentation and examples

Before contributing, read `E:\ARGUS_VAULT\00_CLAUDE_BOOT.md` for the project's development model. The vault (`E:\ARGUS_VAULT`) is the source of truth for project state and architecture decisions.

## License

Proprietary — see `pyproject.toml` for license details.
