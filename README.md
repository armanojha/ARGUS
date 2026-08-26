# ARGUS

**Adaptive Autonomous Evidence & Reasoning Intelligence System**

A cloud-LLM, local-control-plane RAG system that treats difficult
questions as evidence-backed investigations: adaptive hybrid retrieval,
an evidence graph with temporal reasoning, closed-loop verification, and
(from Phase 05 onward) an Obsidian-integrated personal knowledge layer.

This repository is implemented phase by phase against the plan in
`E:\ARGUS_VAULT`. **Project state, architecture decisions, and what's
currently implemented live in the vault, not in this README** — read
`E:\ARGUS_VAULT\00_CLAUDE_BOOT.md` before making changes.

## Status

Phase 00.1 (repository + configuration) only. No API, no LLM gateway, no
retrieval yet — see the vault's phase files for what's next.

## Setup

Requires Python 3.11+.

```powershell
# 1. Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate

# 2. Install dependencies (pick the groups you need; dev-test for testing)
pip install -e ".[core,dev-test]"
# add ,retrieval and/or ,graph once those phases are implemented

# 3. Copy the environment template and fill in real values locally
copy .env.example .env

# 4. Run the tests
powershell -ExecutionPolicy Bypass -File scripts\run_tests.ps1
# equivalent to: python -m pytest tests/ -v
```

## Repository layout

```
app/
  api/              FastAPI app (Phase 00.2)
  orchestration/    LangGraph state machine (Phase 02)
  llm_gateway/       Provider abstraction + routing (Phase 00.3, 07)
    providers/
    routing/
    policies/
  retrieval/        BM25 + vector hybrid retrieval (Phase 01)
  reranking/         (Phase 01)
  graph/             Evidence Graph (Phase 03)
  memory/            Persistent memory layers (Phase 08)
  evidence/          Evidence Store (Phase 01)
  verification/      Claim verification + contradiction detection (Phase 04)
  ingestion/         Document ingestion (Phase 01), multimodal (Phase 11)
  integrations/
    obsidian/        Obsidian vault adapter (Phase 05, 09)
  evaluation/        Benchmarks + ablations (Phase 12)
  ui/                Evidence explorer UI (Phase 12)
  config.py          Core settings loader (Phase 00.1 — implemented)
data/                raw / processed / indexes / graph / memory / obsidian_index
benchmarks/          Benchmark question sets (Phase 12)
tests/               pytest suite
notebooks/           exploratory notebooks
scripts/             utility scripts (test runner, etc.)
configs/
  providers.yaml     LLM provider config (stub — Phase 00.3/07)
  obsidian.yaml      Obsidian integration config (stub — Phase 05/09)
```

## Configuration

Settings are loaded via `app.config.get_settings()` (pydantic-settings),
reading `ARGUS_*`-prefixed environment variables and `.env`. See
`.env.example` for the full list. YAML config stubs in `configs/` are
loaded on demand via `load_providers_config()` / `load_obsidian_config()`
and are safe to load even when nearly empty.

## Project state, phases, decisions

All of this lives in the vault, not here:

- `E:\ARGUS_VAULT\00_CLAUDE_BOOT.md` — start here
- `E:\ARGUS_VAULT\01_MASTER_PHASE_INDEX.md` — phase list + current status
- `E:\ARGUS_VAULT\02_PROJECT_STATE.md` — what works / what's next
- `E:\ARGUS_VAULT\03_ARCHITECTURE_DECISIONS.md` — implementation-affecting decisions
- `E:\ARGUS_VAULT\phases\` — one file per phase, with sub-phase checklists
- `E:\ARGUS_VAULT\handoffs\CURRENT_HANDOFF.md` — short session-to-session handoff
