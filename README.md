# ARGUS

**Adaptive Autonomous Evidence & Reasoning Intelligence System**

A cloud-LLM, local-control-plane research system that treats difficult
questions as evidence-backed investigations: adaptive hybrid retrieval, an
evidence graph with temporal reasoning, closed-loop verification, adaptive
research policy, multi-agent challenge, persistent memory, and an
Obsidian-integrated personal knowledge layer with guaranteed provenance.

## Status

Phases 00–11 are **complete** and tested:

- 00 Foundation (repo, config, FastAPI, LLM gateway core)
- 01 Hybrid RAG (evidence store, BM25 + FAISS retrieval, chunking, reranking)
- 02 Agentic RAG (LangGraph orchestration loop, planning, budgets, citations)
- 03 Evidence Graph (entities/claims/events, 8 edge types, temporal model)
- 04 Verification & Contradiction (confidence scoring, gap-driven re-retrieval)
- 05 Obsidian ingestion (MVP: scan/parse/sync/write-back)
- 06 Adaptive research policy (stop conditions, question routing)
- 07 Multi-model fabric (explicit call-type policy, fallback, quota, telemetry)
- 08 Memory & self-evolution (SQLite layers, promotion, graph versioning)
- 09 Obsidian integration (full: 7-class taxonomy, hypothesis research,
  vault-graph alignment, vault-memory sync, write-back proposals)
- 10 Multi-agent challenge (5 roles, activation rules, debate loop)
- 11 Multimodal intelligence (OCR fallback, tables, web, spreadsheets,
  chart/image extraction, multimodal provenance)

Current test suite: **354 passed, 8 skipped** (skips are live-LLM
integration tests behind `RUN_LIVE_LLM_TESTS=1`).

Next phase: **12 — Research UI + Evaluation** (not started; see
`E:\ARGUS_VAULT\phases\PHASE_12_UI_EVALUATION.md`).

## Setup

Requires Python 3.11+ (developed against 3.14).

```powershell
# 1. (Re)create the virtual environment
python -m venv .venv
.\.venv\Scripts\Activate

# 2. Install dependencies
pip install -e ".[core,retrieval,graph,multimodal,dev-test]"

# 3. Configure environment
copy .env.example .env
# Edit .env — add API keys as needed (GROQ_API_KEY / GEMINI_API_KEY /
# CEREBRAS_API_KEY for the LLM gateway). Tesseract is a system dependency
# for the OCR fallback (see configs/providers.yaml + D-011/D-005 in the vault).

# 4. Run the tests
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

# Retrieve over the widget corpus / query the orchestration loop
curl -X POST http://localhost:8000/api/v1/query -H "Content-Type: application/json" -d '{"query": "..."}'
```

### Ingesting the Obsidian vault

```powershell
# Full Phase 09 pipeline: ingest -> memory sync -> hypothesis research
python scripts/run_obsidian_vault.py <VAULT_DIR> --all
# Subset: --ingest, --sync-memory, --research
```

### Running live LLM tests

```powershell
$env:RUN_LIVE_LLM_TESTS = "1"
python -m pytest tests/test_llm_gateway_integration.py -v
```

## Development

The project follows a phased implementation model. Each phase has a
specification in the vault (`E:\ARGUS_VAULT\phases/`) and is implemented,
tested, reviewed, and committed as a unit.

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
- Model assignment is **explicit configuration only** (`configs/model_policy.yaml`);
  ARGUS never autonomously discovers or ranks models
- LLMs are interpreters, not the database of record
- All evidence and provenance in deterministic stores
- Obsidian notes are personal claims, never automatically-trusted evidence

## Testing & Quality

| Check | Status |
|-------|--------|
| `pytest tests/ -v` | 354 passed, 8 skipped |
| `ruff check app/` | clean (36 pre-existing lint items remain in 3 test files: import-sort/unused imports; unchanged baseline) |
| `mypy app/` | passing |

## Roadmap

```
00 Foundation                  COMPLETE
01 Hybrid RAG                  COMPLETE
02 Agentic RAG                 COMPLETE  <- MVP core loop
03 Evidence Graph              COMPLETE
04 Verification & Contradiction COMPLETE
05 Obsidian Ingestion (MVP)    COMPLETE  <- MVP BOUNDARY
06 Adaptive Research Policy    COMPLETE
07 Multi-Model Fabric          COMPLETE
08 Memory & Self-Evolution     COMPLETE
09 Obsidian Integration (full) COMPLETE
10 Multi-Agent Challenge       COMPLETE
11 Multimodal Intelligence     COMPLETE
12 Research UI + Evaluation    NEXT (not started)
```

## Known limitations

- Phase 10.2 disagreement-triggered retrieval is implemented for detection;
  the wiring back into the retriever is documented in the vault (P10-01).
- Phase 11.5 chart extraction is a placeholder (always returns `ChartType.OTHER`
  with empty data points — deferred to a vision model); table/image/web/
  spreadsheet paths are fully functional.
- Web ingestion performs synchronous HTTP fetches (no async/HEAD first).
- Git remote tracking of `origin/main` may need `git fetch` re-establishment.
- Observability/UI surface is deferred to Phase 12.

## Contributing

ARGUS is in active development. Contributions are welcome for:

- Implementing the active phase (see the roadmap and the vault)
- Adding new LLM providers to the gateway
- Improving test coverage
- Documentation and examples

Before contributing, read `E:\ARGUS_VAULT\00_CLAUDE_BOOT.md` for the
project's development model. The vault (`E:\ARGUS_VAULT`) is the source of
truth for project state and architecture decisions.

## License

Proprietary — see `pyproject.toml` for license details.