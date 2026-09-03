# ARGUS Knowledge-System Final Report

**Date:** 2026-09-03
**Scope:** Turn ARGUS into a coherent user-facing personal research/knowledge
system with three distinct, non-mixed knowledge layers, all fronted by a
4-section UI. Reused the existing architecture (ingestion pipeline, memory,
Obsidian integration, orchestration) rather than redesigning it.

---

## 1. Executive Summary

ARGUS is now a user-facing knowledge system built on three clearly separated
layers, each with its own identity, storage, and lifecycle:

| Layer | What it is | Storage | Distinct-from guarantee |
|---|---|---|---|
| **Knowledge Base** | The user's document corpus | `ARGUS_KNOWLEDGE_BASE_PATH` (default `E:/KNOWLEDGE BASE`) → EvidenceStore | User facts/documents |
| **ARGUS Brain** | ARGUS's persistent machine memory | `MemoryStore` (SQLite, 6 layers) | Derived knowledge with provenance |
| **ARGUS Obsidian Brain** | ARGUS's dedicated human-readable vault | `ARGUS_BRAIN_VAULT_PATH` Obsidian vault | Structured, provenance-bearing notes |

All three are presented through a single 4-section Streamlit control plane
(**Chat/Research**, **Knowledge Base**, **ARGUS Brain**, **Obsidian Brain**)
that talks only to the ARGUS API. The control plane is deterministic — no LLM
calls are made for status/ingest/promote operations.

**Validation: full suite `534 passed / 20 skipped`; `ruff app/` clean; no new
mypy errors.** Git was left to the owner (per directive; no commits made in this
task).

---

## 2. The Three Layers (non-mixed)

### 2.1 User Knowledge Base (`app/ingestion/knowledge_base.py`)
The managed, deterministic entry point that turns `E:/KNOWLEDGE BASE` into the
EvidenceStore, **reusing** the existing `IngestionPipeline` and its built-in
content-checksum dedup (no second ingestion architecture):
- Recursive discovery (`discover_files`), `.pdf`/`.txt`/`.md` always + spreadsheets
  when multimodal is on (`supported_extensions`/`kind_of`).
- Per-file sync (`ingest_knowledge_base`) reporting ingested / unchanged / errors,
  optional index refresh (`retriever.mark_dirty()` + `ensure_indexes()`).
- Idempotent/incremental: unchanged files are skipped via checksum dedup.
- **No LLM calls on the control plane.**

### 2.2 ARGUS Brain (`app/api/brain.py`, existing `app/memory`)
ARGUS's persistent machine memory (6 layers, provenance, promotion status,
versioning — pre-existing, unmodified). New read-only `GET /api/v1/brain/status`
reports totals, layer/promotion/scope counts, avg confidence, DB size, and the
most recent memory records with provenance (graceful disabled status when memory
is off). ARGUS may *selectively* consult long-term memory when a query needs
continuity — distinct from document evidence.

### 2.3 ARGUS Obsidian Brain (`app/integrations/obsidian/promotion.py`, `app/api/obsidian.py`)
ARGUS's dedicated human-readable vault. New:
- `GET /api/v1/obsidian-brain/status` — vault path, configured/exists, note count, recent notes.
- `POST /api/v1/obsidian-brain/promote` — **selective** memory→vault promotion.
- `promote_eligible_memories()` — only records with `promotion_status == PROMOTED`,
  in `LONG_TERM_KNOWLEDGE`, carrying provenance, and at/above
  `memory_confidence_threshold` become notes. Notes are structured, provenance-bearing
  (frontmatter: `type: argus-knowledge`, confidence, memory_id, subject/predicate/object,
  source_query, `sources:` chunk list, tags), idempotent (existing note skipped),
  written under `<vault>/90_ARGUS/Knowledge/`. The note footer warns it is **derived
  knowledge, not raw source evidence**.

### 2.4 Selective memory consultation visibility (additive, lazy)
The existing `_memory_enhance_node` (pre-existing) already consults long-term/
research/source/user memory selectively to enrich the plan. This task made that
**visible** without changing orchestration architecture, model policy, retrieval,
or the verification stage:
- `OrchestrationState.memory_consulted` (new `NotRequired[list[str]]`, `orchestration/state.py`).
- `_memory_enhance_node` records the consulted derived-knowledge layers when the plan was
  actually influenced (`orchestration/graph.py`).
- `OrchestrationResult.memory_consulted` (additive field, `orchestration/models.py`),
  wired in `_build_result`.
- UI surfaces it as distinct from document evidence (see §4). Empty when memory wasn't consulted.

---

## 3. API Surface (new)

All under `/api/v1` (routers registered in `app/api/main.py`):

| Method | Path | Purpose |
|---|---|---|
| GET | `/knowledge-base/status` | Corpus path, exists, doc/source/chunk counts, supported types, indexed, recent documents |
| POST | `/knowledge-base/ingest` | Re-sync the configured corpus (idempotent), optional index rebuild |
| POST | `/knowledge-base/upload` | Upload files → written into KB root → fed through the **same** `IngestionPipeline`; per-file accepted/rejected |
| GET | `/brain/status` | ARGUS memory status + recent records (graceful when disabled) |
| GET | `/obsidian-brain/status` | Vault status |
| POST | `/obsidian-brain/promote` | Selective memory→Obsidian promotion sweep |

EvidenceStore gained public query methods: `count_documents`, `count_sources`,
`count_chunks`, `list_sources`, `list_documents` (`app/evidence/store.py`).

### Config (new, `app/config.py` + `.env.example`)
- `ARGUS_KNOWLEDGE_BASE_PATH` (default `E:/KNOWLEDGE BASE`).
- `ARGUS_BRAIN_VAULT_PATH` (default `""`, i.e. unconfigured until set).
- `ARGUS_BRAIN_WRITE_BACK_ROOT` (default `90_ARGUS`).

No hardcoded paths — all configurable via `ARGUS_`-prefixed env vars.

### CLI
`scripts/ingest_knowledge_base.py`: `--knowledge-base`, `--db-path`,
`--no-rebuild-indexes`, `--log-level`; prints ingested/unchanged/errors/duration.

---

## 4. UI (4-section Streamlit, `app/ui/`)

`api_client.py` gained methods for the new endpoints. `streamlit_app.py` now has
four sections selected via the sidebar:
1. **Chat / Research** — existing `POST /api/v1/query`; grounded cited answer,
   plan, evidence, source trail, verification; now also shows whether ARGUS memory
   was consulted, explicitly distinct from document evidence.
2. **Knowledge Base** — status counts, recent documents, upload, resync.
3. **ARGUS Brain** — memory stats, layer/promotion breakdown, recent records with provenance.
4. **Obsidian Brain** — vault status, recent notes, selective promote action.

No duplicate ingestion or LLM calls on refresh — each action is explicit.

---

## 5. Validation

- **Full suite:** `pytest -q` → **534 passed, 20 skipped** (skips are the
  pre-existing network/key-gated live LLM tests).
- **New tests:** `tests/knowledge/` (15 tests):
  - `test_knowledge_base.py` — discovery (recursive/sorted/excluded types),
    idempotent sync, content-change re-ingest, result shape.
  - `test_promotion.py` — selective eligibility (only PROMOTED + long-term +
    provenance), idempotency, missing-vault skip, frontmatter/provenance preservation.
  - `test_memory_consulted.py` — `memory_consulted` surfaced from state, distinct from citations.
- **Orchestration regression:** `tests/orchestration` → 84 passed.
- **ruff:** `python -m ruff check app/` → **All checks passed** (added
  `fastapi.File` to ruff's `extend-immutable-calls` for the upload endpoint).
- **mypy:** no **new** errors from this task; the repo carries pre-existing mypy
  debt in modules not touched here (`agents.py`, `web.py`, `tables.py`, `images.py`,
  `coordinator.py`, `research.py`, `openai_compatible.py`).

---

## 6. Honest Limitations

- **Tesseract OCR INSTALLED (2026-09-03):** Ub-Mannheim `tesseract v5.4.0` installed per-user at
  `C:\Users\LOQ\AppData\Local\Programs\Tesseract-OCR` and added to the User PATH.
  `shutil.which("tesseract")` now resolves; validated end-to-end on the arXiv PDF (page 1 → 2855 OCR
  chars @ 94.4% confidence). New test `test_ocr_extracts_rasterized_content` proves OCR recovers text
  from an image-only PDF (skips on machines without tesseract). Multimodal `40 passed` ingestion set green.
  **Note:** the change takes effect for newly launched processes (API server must be restarted to pick up PATH).
- **Obsidian brain consultation is status/promote only** — the query loop consults
  machine *memory* layers selectively, but does **not** retrieve from the Obsidian
  vault as a retrieval source (per the agreed scoping: these layers stay distinct
  and are NOT default retrieval sources).
- **mypy baseline is not clean** repo-wide (pre-existing errors in unrelated modules).
- **Frame-state visibility** is a summary flag (`memory_consulted` layers), not per-memory-record detail in the chat answer; full provenance is inspectable in the ARGUS Brain section.
- **Promotion requires a configured brain vault** (`ARGUS_BRAIN_VAULT_PATH`); until set, the Obsidian Brain section reports unconfigured and promote degrades gracefully.
- **Git not committed** — owner handles Git per directive; working changes are complete and green in-tree.

---

## 7. Files Changed / Added

**New:** `app/ingestion/knowledge_base.py`, `app/api/knowledge_base.py`,
`app/api/brain.py`, `app/api/obsidian.py`, `app/integrations/obsidian/promotion.py`,
`scripts/ingest_knowledge_base.py`, `tests/knowledge/*`.

**Extended:** `app/config.py` (paths), `app/evidence/store.py` (count/list API),
`app/api/main.py` (routers), `app/ui/api_client.py` + `app/ui/streamlit_app.py` (4 sections),
`app/orchestration/{state,models,graph}.py` (`memory_consulted`),
`.env.example`, `pyproject.toml` (`ruff extend-immutable-calls`).

## 8. Next
- Set `ARGUS_BRAIN_VAULT_PATH` to a real Obsidian vault to activate the Obsidian Brain layer end-to-end.
- Owner to commit/push.
- Optional future work (requires owner decision): observe vault retrieval, per-record memory visibility in chat, OCR deployment.