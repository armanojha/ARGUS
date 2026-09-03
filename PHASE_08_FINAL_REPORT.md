# PHASE 08 — Memory & Self-Evolution: Audit + Gap Closure Report

**Date:** 2026-09-03
**Repository:** `E:\ARGUS\ARGUS`
**Base commit (HEAD):** `0a61c95` (`add phase 07g finalization`)
**Method:** audit of the existing `app/memory/` implementation against the Phase 08 requirements, followed by targeted closure of the identified gaps. No prohibited scope added; memory operations remain bounded, cacheable, and deterministic (no new "memory LLM"); no provider/model-policy change; no DB added beyond the existing shared SQLite store (D-003); no git operations performed.

---

## 1. Executive Summary

Phase 08 (Memory & Self-Evolution) is **substantially implemented and committed** in the repository (`app/memory/`: interfaces, SQLite `MemoryStore`, `GraphVersionManager`, memory-aware planner integration, factory/DI, plus the §7 versioned-delta wiring in `EvidenceGraphStore`). The audit confirmed the multi-layer, provenance-aware, promotion/provisional, versioned-delta, and planner-integration requirements are present and architecturally consistent with D-003 (shared SQLite DB) and the "no LLM-as-database" rule.

Two genuine gaps were found and closed:

1. **Public-API breakage (correctness bug):** the `MemoryPromotionStatus` enum was used by the code and tests but not exported from `app.memory`, so `tests/memory/test_memory.py` **failed to even collect** at HEAD (`ImportError: cannot import name 'MemoryPromotionStatus'`). Exported it (imports + `__all__`).
2. **§17/§18 fresh-evidence-wins (semantic gap):** the supersede machinery existed but the resolution rule (a) only matched same-`object` updates (missed true contradictions with a different object), (b) did not preserve higher-confidence existing memory against a clearly weaker new claim, and (c) recorded only a one-directional link. Hardened the rule and added bidirectional links.

The §7 graph versioned-delta wiring already existed in `EvidenceGraphStore`; this audit **verified** it and added dedicated tests.

**Result:** full suite **519 passed / 20 skipped** (baseline 513/20, +6 memory tests). No regressions. Present in the working tree as 3 uncommitted file changes (no commit performed, per directive).

---

## 2. Scope & Method

- Read the existing Phase 08 modules: `interfaces.py` (293 lines), `models.py` (41), `store.py` (557), `versioning.py` (514), `planner_integration.py` (265), `factory.py` (129), `__init__.py` (81) and `app/graph/store.py` (versioned-delta integration) + `app/graph/contracts.py`.
- Read `tests/memory/test_memory.py` (existing coverage) and the config settings (`memory_*`, `graph_versioning_*`).
- Mapped the existing implementation against the Phase 08 requirements line-by-line (see §5).
- Audited wiring: `MemoryStore`/`GraphVersionManager` ⇄ `EvidenceGraphStore` (`app/graph/store.py`), and `MemoryAwarePlanner` ⇄ `app/orchestration/graph.py` `_memory_enhance_node`.
- Ran a pristine-HEAD stash test to prove the pre-existing breakage, then applied and verified the fixes.

**Decisions confirmed with the owner:**
- (a) Treat Phase 08 as **audit + gap-closure**, not a rebuild.
- (b) For §7, wire `EvidenceGraphStore` through the existing `GraphVersionManager` (the already-committed approach) rather than a new wrapper.

---

## 3. What Already Exists (verified complete)

| Requirement | Status | Evidence |
|---|---|---|
| Multi-layer memory (working, long_term_knowledge, research_history, source_memory, user_memory, vault_memory) | ✅ | `MemoryLayer` (6 members incl. `VAULT_MEMORY`); per-layer store + index |
| Provenance-aware records | ✅ | `MemoryRecord`: `supporting_chunk_ids`, `source_query`; `memory_chunks` join table |
| Promotion / provisional semantics | ✅ | `MemoryPromotionStatus` (PROVISIONAL/PROMOTED/REJECTED/ARCHIVED); `MemoryStore.promote_memory`; layer-limit pruning |
| Versioned, non-destructive graph deltas | ✅ | `GraphVersionManager`, `GraphDelta` (frozen), 13 `DeltaType`s incl. CLAIM_CONTRADICTED/SUPERSEDED/REVISED, delta chain, claim-version history, auto-promotion, retention cleanup |
| §7: versioned deltas wired into `EvidenceGraphStore` | ✅ | `get_graph_store()` constructs with `version_manager=get_version_manager()`; mutations emit ENTITY/CLAIM/EVENT/EDGE created/updated/contradicted deltas |
| Memory-aware planner integration | ✅ | `MemoryAwarePlanner` consults long-term knowledge, research history, source, user preferences; `_memory_enhance_node` in `app/orchestration/graph.py` |
| Bounded retrieval / determinism | ✅ | `memory_max_records_per_layer`, `max_memory_results`, `min_confidence`; no "memory LLM"; SQLite LIKE search |
| Factory / DI / disabled-path | ✅ | `MemoryFactory`, `NullMemoryStore`, `initialize_memory_system`/`shutdown_memory_system` |
| Config-gated | ✅ | `memory_enabled=False` default; `graph_versioning_enabled=True` |

---

## 4. Gaps Found and Closed

### Gap 1 (correctness) — `MemoryPromotionStatus` not exported → broken module

- **Symptom:** `tests/memory/test_memory.py` imports `MemoryPromotionStatus` from `app.memory`, but `app/memory/__init__.py` did not export it (`__all__` omitted it). Verified via a pristine-HEAD stash run: `ImportError: cannot import name 'MemoryPromotionStatus' from 'app.memory'` at collection.
- **Fix:** `app/memory/__init__.py` now imports `MemoryPromotionStatus` and adds it to `__all__`. The memory test module collects and passes.
- **Files:** `app/memory/__init__.py` (+2 lines).

### Gap 2 (semantic) — fresh-evidence-wins / contradiction resolution rule

The supersede machinery (`supersedes_id`/`superseded_by_id`, `ANS_*`/`ARCHIVED`, `CLAIM_CONTRADICTED`) existed but the automatic resolution in `MemoryStore._resolve_fresh_evidence` had three weaknesses:

1. **Matched only same-`object` records** (`content != ? AND object = ?`), so a genuine contradiction with a *different* object (the common case: "CEO is A" vs "CEO is B") was not treated as a supersession candidate.
2. **Pure-recency decision** allowed a *newer, lower-confidence* claim to supersede a well-supported existing memory (eroding knowledge with weak new evidence).
3. **Only one-directional link** — the old record's `superseded_by_id` was set, but the new record's `supersedes_id` was not, so the pair wasn't joinable.

**Fix (`app/memory/store.py`, `_resolve_fresh_evidence`):**
- Match on `(subject, predicate, layer)` with differing `content` **or** differing `object` (catches contradictions) and only records not already superseded.
- Confidence-gated rule: supersede only when `record.confidence >= older_conf` (recency is the tie-breaker). A clearly weaker new claim never erodes a well-supported existing memory.
- Set the bidirectional link (old `superseded_by_id` → new, new `supersedes_id` → old), both archived/promoted as appropriate.

### Gap 3 (verification) — no tests for §7 graph-delta emission or the fresh-evidence rule

Added tests under `tests/memory/test_memory.py`:
- `TestGraphStoreVersionedDeltas`: entity CREATED delta, claim CREATED→CONTRADICTED deltas, edge ADDED delta, and confirmation that an `EvidenceGraphStore` without a version manager records no deltas (no cross-store surprise).
- `TestSQLiteMemoryStore.test_fresh_evidence_wins_supersedes_older` and `...keeps_old_when_newer_has_lower_confidence_age`.

---

## 5. Requirement Mapping (T1–T18 style)

The Phase 08 requirements enumerate store CRUD, layered retrieval, provenance, promotion, layer limits, versioned deltas, delta chain, claim history, auto-promotion, planner integration, bounded retrieval, determinism, and contradiction handling. Coverage:

| Area | Covered by |
|---|---|
| Store CRUD (store/retrieve/update/delete/get_by_id) | `MemoryStore` + tests |
| Layer filter / global+session+query scopes | `MemoryQuery.layers/scope` + tests |
| Confidence threshold retrieval | `MemoryQuery.min_confidence` + test |
| Layer limit enforcement | `_enforce_layer_limit_sync` + test |
| Text search / provenance (chunk + tag) | `retrieve` LIKE + index tables |
| Promotion (provisional → promoted) | `promote_memory` + test |
| Versioned deltas (non-destructive) | `GraphVersionManager` + tests |
| Delta chain / ordered history | `delta_chain` + test |
| Claim version history | `claim_versions` + test |
| Auto-promotion by confidence | `record_delta`/`auto_promote_eligible` + tests |
| Reject / manual promote | `promote_delta`/`reject_delta` + tests |
| §7 graph delta emission | `EvidenceGraphStore._record_delta` + new tests |
| Planner memory consultation | `MemoryAwarePlanner` + tests |
| Fresh-evidence-wins / contradiction supersede | `_resolve_fresh_evidence` + new tests |

---

## 6. Validation

- **Full suite:** `pytest -q` → **519 passed, 20 skipped** (baseline 513 passed / 20 skipped ⇒ +6 new memory tests).
- **Memory module:** `pytest tests/memory` → **31 passed**.
- **Graph / cross-phase / obsidian:** `pytest tests/graph tests/test_cross_phase_integration.py tests/obsidian` → **93 passed** (these construct `EvidenceGraphStore` without a version manager, confirming no cross-store side effects).
- **Strict runtime flags:** `pytest tests/memory -W error::RuntimeWarning` → clean. (Pre-existing `datetime.utcnow` `DeprecationWarning`s originate in `app/graph/models.py`, unchanged by this work.)
- **Lint:** `ruff check` on all changed files → clean.

---

## 7. Files Changed (working tree, uncommitted)

- `app/memory/__init__.py` — export `MemoryPromotionStatus`.
- `app/memory/store.py` — hardened `_resolve_fresh_evidence` (gap #2).
- `tests/memory/test_memory.py` — new §7 delta tests + fresh-evidence tests (gap #3).

No git operations were performed.

---

## 8. Scope Boundaries Honored

- No changes to provider/model policy, providers, retrieval, orchestration graph routing, verification gates, or prior Phase 07 behavior.
- No new external database (shared SQLite per D-003).
- No new "memory LLM" — resolution logic is deterministic.
- Memory operations remain bounded/cacheable/deterministic.
- The §7 change touches `EvidenceGraphStore` only through the pre-existing seam (already committed `get_graph_store()` wiring); explicit constructions default to no versioning to avoid cross-store surprises.
- All prior Phase 07d/07e/07f/07g guarantees preserved by the green full-suite run.

---

## 9. Honest Verdict

Phase 08 is **COMPLETE AS AN AUDIT + GAP-CLOSURE**. The memory subsystem was already present and committed; this session proved it, fixed two real defects (a broken public-API export that prevented the memory tests from collecting, and a weak fresh-evidence/contradiction resolution rule), added targeted tests, and validated the entire suite. The outstanding caveat, consistent with the roadmap, is that `memory_enabled=False` by default, so the subsystem is available and correct but opt-in at runtime.