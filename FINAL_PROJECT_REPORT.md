# ARGUS — FINAL PROJECT REPORT

**Date:** 2026-09-03
**Status:** ALL PHASES COMPLETE / PROJECT CLOSED OUT at the 07c roadmap M2 stop-line.
**Head of `main`:** `299db73` (2 commits ahead of origin, pending push: `fb05032`, `299db73`).

---

## 1. Executive Summary

ARGUS is a cloud-LLM, local-control-plane, adaptive **Agentic + GraphRAG + Evidence Verification** system. All phases (00–12) are implemented, tested, and committed. The project has been formally closed out after (a) the Phase 08 Memory & Self-Evolution audit + gap-closure, (b) the owner's latency policy decision, and (c) a final hardening + code-review pass.

**Final validation:** full test suite **519 passed / 20 skipped**, whole-repo **ruff clean**, and offline regressions reconfirming the accepted latency floor.

---

## 2. What Was Delivered This Session

### A. Phase 08 — Memory & Self-Evolution (AUDIT + GAP-CLOSURE), committed `d7d99db`
Per owner decision, the existing committed memory subsystem was **audited, not rebuilt**. It was verified correct:
- Multi-layer SQLite MemoryStore (working / long_term_knowledge / research_history / source_memory / user_memory / vault_memory)
- Provenance records, layer limits, promotion / provisional semantics
- GraphVersionManager (versioned non-destructive deltas, delta chain, claim history, auto-promotion)
- Memory-aware planner integration, factory / DI, `NullMemoryStore`
- §7 versioned-delta emission in `EvidenceGraphStore`

Two real defects were found and closed:
1. **`MemoryPromotionStatus` was not exported from `app.memory`** → `tests/memory/test_memory.py` could not collect at HEAD (`ImportError`). Fixed in `app/memory/__init__.py`.
2. **`MemoryStore._resolve_fresh_evidence`** missed different-object contradictions, let a weaker new claim erode a well-supported existing memory, and set only a one-directional link. Hardened to confidence-gated, object-contradiction-aware, bidirectional supersede links.

+6 tests. Report: `PHASE_08_FINAL_REPORT.md`.

### B. Latency (p95) policy — DECIDED by owner
**Accept ~8.2 s as the current clean-path p95 floor.** Do NOT weaken the 07d verification gate; do NOT introduce a cheaper-latency verification path solely to chase the ≤8 s target. **Latency work is CLOSED** unless future architectural/model changes provide a genuinely safe improvement. (Basis: `PHASE_07G_FINAL_REPORT.md` measured structural wall.)

### C. Standalone lint pass, committed `c964330`
48 safe auto-fixes (unused imports/`noqa` removal, import-block sorting) across `benchmarks/` + test files. Then all **10 remaining judgment-call findings resolved** in the final review pass (see E).

### D. Housekeeping
`.gitignore` now ignores `data/memory/` (the test-run side-effect `memory.db` artifact).

### E. Final hardening + code review, committed `fb05032`
A full codebase review found **no critical default-configuration vulnerability, no hardcoded secrets, no SQL injection, no `eval`/`exec`**. Four real issues were fixed:
| ID | Issue | Fix | File |
|----|-------|-----|------|
| H1 | Web ingestion could fetch SSRF-sensitive targets (loopback/private/metadata) with no guard | New `_guard_ssrf_target()` blocks these before any outbound fetch (defense-in-depth; not externally reachable today) | `app/ingestion/web.py` |
| M2 | Un-closed PIL image handle per chart | Chart image now opened via `with PILImage.open(...)` | `app/ingestion/images.py` |
| M4 | BM25 reached into private `EvidenceStore._conn()/_row_to_chunk()` | New public `EvidenceStore.iter_chunks()` | `app/evidence/store.py`, `app/retrieval/bm25.py` |
| M5 | `max_memory_results` derived from nonsensical `layer_cap // 1000 + 3` | Fixed to sane prompt-budget `5` | `app/memory/planner_integration.py` |

Assessed and deliberately **not** changed (correct as-is):
- `NullMemoryStore` pass-bodies = intentional null-object pattern; init failures are already logged (`memory_system_unavailable`) as a non-fatal optional enhancement.
- `ConfidenceScorer`/`ContradictionDetector` = preserved, unit-tested public API; removal would be a breaking change.
- `graph/store.py` bare `except` logs + re-raises and is **not** ruff-flagged (spurious finding).
- Best-effort metadata/date `pass` in `web.py` is intentional scraper tolerance.

---

## 3. Validation (this session)

| Check | Result |
|-------|--------|
| Full test suite | **519 passed / 20 skipped** (unchanged after all fixes) |
| Affected test modules (post-lint) | 103 passed / 12 skipped |
| Ruff (whole repo) | **All checks passed** |
| Regression 07d (healthy scan, offline) | **38/38 answered, 38/38 grounded**, verification_triggered=38 |
| Regression 07e (recovery, offline) | R-e1..R-e4 all match baseline |
| Regression 07f (latency, offline) | LA1 p50 8192ms / **p95 8238ms** / p99 8271ms; LA2 p50 8172ms / **p95 8230ms** / p99 8235ms; calls/q 4.289; grounded 1.0; RE1 fused-identity 0 mismatches, recall@8 0.987 |

All regressions ran **offline (mock)** — zero live API quota burned, per quota discipline.

---

## 4. Git State

```
299db73 refresh regression baseline results
fb05032 harden web URL fetch, index API, and lint cleanup
c964330 refine benchmarks, eval data, and test suites
d7d99db add phase 08 memory audit gap-closure
0a61c95 add phase 07g finalization
...
```
Working tree **clean**. Branch is **2 commits ahead** of `origin/main` (`fb05032`, `299db73`) — publish pending owner approval.

---

## 5. Accepted Latency Floor (evidence)

- 07f SAFE retrieval-layer parallelism was the only concurrency the dependency graph permits; the orchestration LLM chain (analysis→planning→assess→synthesis→verify ≈ 8.1 s) is strictly serial and cannot be safely overlapped.
- 07g measured that p95 ≤ 8 s is structurally unattainable without fast-pathing the D/E/F/I classes or weakening the 07d verification gate — both forbidden.
- Empirically reconfirmed here: **p95 ≈ 8.2 s** (8.23 s on this run).

---

## 6. Notes / Caveats

- Regenerated `regression_07d.json`/`regression_07f.json` have fresh content-addressed chunk UUIDs (corpus rebuild) but identical results — cosmetic churn, committed as refreshed baselines.
- `data/memory/memory.db` may reappear after test runs; it is now `.gitignore`d.
- No secrets were committed; no provider/model/quota policy changed.

---

## 7. Recommended Next (all require owner approval)

1. `git push origin main` to publish `fb05032` / `299db73`.
2. Optionally enable `memory_enabled=True` in deployment config to turn on the (now-hardened) memory subsystem.
3. Any further capability or latency work requires a **new** owner decision per the 07c roadmap stop-line (`ARGUS_FUTURE_ROADMAP.md`).

---

## 8. Honest Verdict

**All phases complete; project meets its design quality/grounding/citation/verification/reliability guarantees. The sole known limitation is the accepted ~8.2 s p95 latency floor, which is structural and was formally accepted by the owner rather than papered over.** The codebase review found no critical default-configuration vulnerabilities. 519/519 exercised tests pass; ruff is clean.