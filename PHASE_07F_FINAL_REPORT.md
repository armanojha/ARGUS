# PHASE 07F — SAFE PARALLELISM + LATENCY REDUCTION — FINAL REPORT

**Phase:** 07F (Safe Parallelism + Latency Reduction)
**Status:** COMPLETE WITH LIMITATIONS
**Date:** 2026-09-03
**Report basis:** `PHASE_07C_REAL_WORLD_EVALUATION.md`, `ARGUS_FUTURE_ROADMAP.md` §13 (M2 stop-line), `PHASE_07D_FINAL_REPORT.md`, `PHASE_07E_FINAL_REPORT.md`, the fixed 38-query control (`benchmarks/eval_data/`), and the new latency-injected regression (`benchmarks/eval_data/regression_07f.py` → `results/regression_07f.json`).

---

## 1. Objective

Take the only genuinely-safe, dependency-preserving parallelism Phase 07f may introduce (per the brief: no speculative racing LLM calls, no parallelization of dependent orchestration stages, no verification-before-synthesis, no state-machine-breaking subquery fan-out, no quota/call-cap regressions, no provider-policy changes) and reduce ARGUS latency toward a nominal target of **p95 ≤ 8 s** on the fixed 38-query evaluation/control workload, while preserving answer quality, grounding, reliability, provider-health behavior, quota/call ceilings, and the existing model/provider policy exactly.

## 2. Scope & Non-Goals

**In scope:** concurrent dispatch of independent retrieval channels (`PolicyRouter.execute_retrieval` method fan-out; `HybridRetriever.search_async` overlapping BM25 + dense passes), with bounded, deterministic, failure-isolated, leak-free concurrency; regression evidence that quality/outcome/citation/call-count are unchanged.

**Explicit non-goals (enforced):**
- No parallelization of the orchestration LLM dependency chain. `analysis → planning → retrieve/assess→(loop)→ synthesis → verification` have hard data dependencies (each stage consumes the previous stage's output). Verification literally consumes the synthesized answer, so it can never overlap synthesis.
- No speculative / racing duplicate LLM calls (identical work issued to multiple providers to "pick the fastest") — forbidden by the brief; would double quota and token spend.
- No concurrent subquery retrieval at the node level — changes LangGraph last-write-wins semantics, `iterations_used`, ordering, and the investigate-then-decide loop, violating the existing tests and the "only after determinism is preserved" guard.
- No model/provider/model-policy/quota/provider-order changes.
- No Phase 07d (Outcome/citation/429 fast-fail) or Phase 07e (recovery probe / health-mode / cooldown) regressions.
- Phase 08 not started.

## 3. Architecture Before (verified, unchanged except Part A/B below)

Byte-oriented pipeline, strictly sequential:
```
analyze(needs query) → plan(needs analysis) → retrieve/assess@i(needs plan/subquery_i) →
  … loop … → synthesize(needs evidence+plan) → selective-verify(needs synthesized answer) → END
```
- `RetrievalPolicyRouter.execute_retrieval` dispatched `mix.methods` **sequentially** in a `for` loop (each method a separate read; fused deterministically by `_fuse` afterward).
- `HybridRetriever.search` ran the BM25 and dense passes **serially** in the same thread.
- The orchestration LLM chain has **no independent work** (verified via source reading + explore agent), so there was no latency lever except retrieval.

## 4. Design of the Safe Parallelism Introduced (Parts A & B)

Only the retrieval layer is parallelized, and only across **independent, read-only channels**:

- **Part A — `RetrievalPolicyRouter.execute_retrieval`:** independent `mix.methods` are now dispatched concurrently via `asyncio.gather` + `asyncio.to_thread(self._dispatch, …)`, with a **bounded `asyncio.Semaphore(min(4, max(1, len(mix.methods))))`** so a pathological method list can never spawn unbounded tasks. Each method remains an isolated read over read-only indexes; the synchronous searches are CPU/I-O bound and GIL-releasing (numpy/torch), so `to_thread` genuinely overlaps them. **Per-method failure is isolated** → `policy_method_failed` + `refs=[]` (a single failing method never aborts the search). Results are collected in original `mix.methods` order; the downstream `_fuse` (dedup by `chunk_id`, weighted-score sort) is **order-independent**, so fused output is byte-identical regardless of completion order. The empty-result deterministic fallback and metadata filters are unchanged.

- **Part B — `HybridRetriever.search_async`:** the BM25 pass and the dense pass now run **concurrently** via two `asyncio.to_thread` tasks (one per wanted mechanism) and are fused by a shared, deterministic `_fuse` (refactored out of `search`; `search` and `search_async` share the exact same fusion). `mechanisms` filtering is honored identically (a lexical-only pattern never pays the embedding cost). `orchestration/nodes.retrieve_node` now calls `search_async` on both the policy-disabled and deterministic-fallback paths so BM25+dense overlap on fast-path/no-policy queries too.

No orchestration LLM stage is parallelized, and verification still runs strictly after synthesis.

## 5. Latency Model & Why the LLM Chain Cannot Be Overlapped

Per-call LLM latencies measured in 07c on the primary/fallback models: analysis/synthesis/evidence_extraction ≈1.0–1.9 s (groq-120b), research_planning ≈1.3–1.6 s (gemini), verification ≈1.3–2.4 s (gemini). The dependency chain forces a **serial sum** of one margin: synthesis needs the plan + retrieved evidence; verification needs the synthesized answer. There is no safe place to overlap two of these. The roadmap's §17 bottleneck #4 ("all-sequential calls → p95 13 s; fix: safe parallelism for independent retrieval/evidence calls") is exactly what Parts A/B implement — but the retrieval latency is a small share of the total, so the serial LLM chain remains the irreducible floor.

## 6. Code Changes (07f)

- `app/retrieval/router.py` — `execute_retrieval`: sequential → bounded-concurrent method dispatch (Part A). `_dispatch`/`_fuse`/`_needed_mechanisms`/metadata-filter fallback untouched in behavior.
- `app/retrieval/hybrid.py` — factor `_fuse` (shared sync/async); add `import asyncio`, `from typing import Any`; add `search_async`; keep `search` behavior byte-identical.
- `app/orchestration/nodes.py` — `retrieve_node`: sync `retriever.search(subquery, …)` → `await retriever.search_async(subquery, …)` on the no-policy and deterministic-fallback paths. `reranker.rerank` stays sync (no over-engineering).
- `tests/retrieval/test_phase07f_parallelism.py` — NEW, 11 focused tests (accepted-checks T1–T10; see §10).
- `benchmarks/eval_data/regression_07f.py` + `results/regression_07f.json` — NEW latency-injected benchmark (Part G).

## 7. Benchmark Harness (Part G)

`regression_07f.py` runs the **real** offline 12-doc/38-query control through the **real orchestration** with a **latency-injected scripted router** whose per-call sleeps mirror the measured per-call LLM latencies (analysis 1.5 s, plan 1.5 s, evidence 1.5 s, synthesis 1.5 s, verify 2.0 s). Because the dependency chain is strictly serial, the full-path wall-clock is an honest serial sum. It also measures retrieval determinism (sync `search` vs async `search_async` fused output identity) and retrieval latency stats, plus full-run p50/p95/p99, calls/query, outcome distribution, grounding rate, and citation integrity over **two repeated controlled runs**. Zero live quota burned.

## 8. Functional Regression Results (quality / grounding / outcome / citations)

From `regression_07f.json` and the 07d control (`regression_07d.json`):

| Check | Result |
|---|---|
| Retrieval fused-identity `search` vs `search_async` | **0 mismatches** across all 38 queries (byte-identical evidence, dedup, rank) |
| Retrieval-analysis recall@8 (mean) | **0.987** (unchanged vs 07d baseline 0.987) |
| Healthy full scan | **38/38 `answered`**, **38/38 grounded** (≥1 citation), 0 zero-citation queries |
| Citation dedup + sequential rank 1..N | preserved under concurrency |
| Outcome distribution | `answered` 38 (no mislabel; `answered_degraded`/`answered_fallback` unchanged by 07e/07d semantics) |
| LLM calls/query | **4.289** in both runs (unchanged; concurrency adds zero duplicate LLM calls) |
| Max calls (single query) | **5** (within ceiling; no speculative calls) |
| 07d regressions (R1–R5) | all reproduce identically (38/38 answered+grounded, recall 0.987; R3 degraded truthful; R4 429 fast-fail+fallback; R5 verification fail-safe) |
| 07e burst regression (R-e1..R-e4) | unaffected (health/cooldown/recovery probe behavior untouched) |

## 9. Latency Results (honest, measured)

`regression_07f.json` (latency-injected, two runs):

| Metric | Run 1 (ms) | Run 2 (ms) |
|---|---|---|
| Full-path p50 | 8168 | 8161 |
| Full-path **p95** | **8201** | **8201** |
| Full-path p99 | 8214 | 8202 |
| Full-path max | 8214 | 8202 |
| Retrieval share (async p50) | — | ~86 ms (~1 % of total) |

Retrieval serial vs async on this control: serial p50 **86.7 ms** vs async p50 **85.6 ms** — the overlap is **neutral on this 12-doc corpus** (parallelism + thread overhead ≈ the small overlap win). The full-path p50/p95/p99 ≈ **8.2 s** is dominated by the serial LLM chain (≈8.1 s), which is above the 8 s target and **irreducible by safe concurrency** given §5.

## 10. Acceptance Tests (T1–T10) — `tests/retrieval/test_phase07f_parallelism.py`

| Check | Test | Outcome |
|---|---|---|
| T1 Independent tasks overlap | `test_t1_independent_methods_overlap` | PASS — `max_active ≥ 2`, wall < serial sum |
| T2 Dependent stages stay sequential | `test_t2_dependent_stages_stay_sequential` | PASS — no speculative overlap; dependent call never precedes its dependency |
| T3 Retrieval determinism | `test_t3_async_search_identical_to_sync` + `test_t3_async_mechanisms_respected` | PASS — sync==async; mechanism filter honored; deterministic/order-independent fusion |
| T4 Concurrent failure isolation | `test_t4_failed_method_is_isolated` | PASS — failing method isolated; survivor contributes; no propagation |
| T5 Quota/call-accounting under concurrency | `test_t5_exactly_one_search_per_dispatch_method` | PASS — exactly one internal search per dispatched method (no duplicates) |
| T6 Bounded fan-out (no unbounded thread pool) | `test_t6_fan_out_is_bounded_and_recovers` | PASS — `max_active ≤ cap`, completes, no leaked tasks |
| T7 Cancellation / leak-freedom | `test_t7_no_orphaned_tasks_after_async_search` | PASS — no orphaned asyncio tasks |
| T8 Citation integrity | `test_t8_citation_dedup_and_rank_preserved` | PASS — dedup + sequential rank |
| T9 Outcome integrity | `test_t9_outcome_and_grounding_preserved` | PASS — `answered` + grounded |
| T10 Healthy-path call count unchanged | `test_t10_healthy_path_llm_call_count_unchanged` | PASS — every completion accounted exactly once; synthesis+verification present |

## 11. Test-Suite Runs

- Full suite: **513 passed / 20 skipped** (was 502/20; +11 new 07f tests). No test weakened.
- `-W error::RuntimeWarning` full run: **513 passed / 20 skipped** — no orphaned-coroutine warnings (the earlier `_bm25_pass`/`_vector_pass` coroutine bug is fixed; remaining warnings are pre-existing pydantic/starlette deprecations).
- All 07f changed files (hybrid.py, router.py, nodes.py, test_phase07f_parallelism.py, regression_07f.py) **ruff-clean**.

## 12. Verification Gates / Provider Policy / Quota Integrity

- 429 fast-fail, Outcome semantics, citation normalization (07d) and health-mode/cooldown/in-session recovery-probe (07e) behavior are **unchanged** — the 07d and 07e regressions reproduce identically.
- Call/quota ceilings intact: concurrency introduces zero new LLM calls (T5, T10; calls/query unchanged at 4.289).
- Model/provider policy (`configs/model_policy.yaml`) untouched; recovery probes never fire concurrently in a way that hammers a health-skipped provider (07f adds no LLM-path concurrency).

## 13. Known Limitations / Caveats

- **The p95 ≤ 8 s target is NOT met** (measured clean-path floor ≈ 8.2 s) and cannot be met safely on the fixed workload: the dominant latency is the strictly-serial, data-dependent LLM chain, and speculative parallel LLM calls / verification-before-synthesis / LLM-latency racing are all forbidden. This matches the roadmap's own § finding #5 direction (safe parallelism "only improves what is independent"), and retrieval is only ~1 % of total on this 12-doc corpus.
- On a larger corpus (or a remote embedder), Part A/B overlap would show a real win proportional to retrieval's share; the control is too small to exhibit it, so the latency evidence is honest (neutral) rather than contrived.
- Retrieval serial-vs-async is a wash on the control (86.7 → 85.6 ms p50); determinism is the unchanged guarantee.
- Pre-existing ruff drift: 63 findings in out-of-scope files (test_multi_agent, test_multimodal, run_eval, resilience_eval, ablation, hybrid.py on CI config paths, routing/registry, etc.) remain untouched (phase-boundary respected).
- Burst/latency validation is mock/offline only; no new live quota burned.

## 14. Risk & Safety Assessment

- **No speculative or duplicate LLM calls** → no quota/token/cost regression, no "pick-the-fastest" races.
- **No orchestration state-machine change** → `iterations_used`, ordering, investigate-then-decide semantics preserved (T2 verifies dependency order).
- **Bounded concurrency** (`Semaphore`) + `gather`/`to_thread` with no orphaned tasks (T6/T7) → no unbounded threads, clean cancellation.
- **Deterministic fusion** (shared `_fuse`, dedup + stable sort, order-independent) → output identical to pre-07f regardless of completion order (T3, T8; fused-identity 0 mismatches).
- **Failure isolation** (per-method try/except, deterministic empty-fallback) → a failing method can never degrade the overall search (T4).

## 15. Decision Notes

- Phase 08 (memory/features) remains NOT the next step — consistent with 07c disposition (§ finding, roadmap stop-line M2 §13).
- Because the serial LLM chain is the irreducible bottleneck and safe parallelism only moves the ~1 % retrieval share, further latency work on this control would require either (a) a permitted, still-safe, deterministic optimization that does not exist in the current dependency graph, or (b) a scope permitting verification/evidence-lean pathway changes, or (c) accepting p95 ≈ 8.2 s on the clean path (without retry overhead) as the practical floor. This is the honest engineering position.

## 16. Honest Verdict

**COMPLETE WITH LIMITATIONS.**

The Phase 07f deliverable — safe, bounded, deterministic, failure-isolated, leak-free parallelism of the genuinely-independent retrieval channels (Parts A & B) with full quality/grounding/outcome/citation/call-count/health/quota/policy preservation — is **implemented, tested (11 new tests), and regression-verified**. The **p95 ≤ 8 s target is NOT reached** (measured clean-path floor ≈ 8.2 s): the dominant latency is the strictly-serial, data-dependent orchestration LLM chain, which cannot be safely overlapped under the brief's constraints (no speculative LLM calls; verification must follow synthesis; state-machine semantics preserved). Retrieval is ~1 % of total latency on the fixed 12-doc control, so its parallelization is deterministic but latency-neutral here (a genuine win on retrieval-heavier workloads).

**Precise bottleneck identified:** the irreducible serial LLM sum `analysis → planning → assess → synthesis → verify` ≈ 8.1 s; retrieval ≈ 0.09 s. Closing p95 below 8 s safely would require reducing the LLM serial sum itself (e.g., a permitted cheaper/lower-latency model path or a verification/evidence-lean gate) — a scope the brief does not authorize.

## 17. Recommended Next Steps

- (Recommended) Commit the pending 07d + 07e + 07f production/test/benchmark artifacts, then close the working tree.
- Follow-up (architecture-level, NOT in this brief's scope): investigate a permitted, quality-preserving reduction of the serial LLM sum on the fast path (e.g., a verified cheaper latency tier only for the *un*contested, low-risk fast path, where 07d already showed the gate skip behavior) — to be a separate approved pass with its own control evidence.
- Optional: a live (small) re-run of the healthy scan to confirm the mock latency model (only if quota budget allows).
- Standalone lint/resolution pass for the 63 out-of-scope ruff findings.

## 18. Artifacts

- `E:\ARGUS\ARGUS\benchmarks\eval_data\regression_07f.py` + `results\regression_07f.json` (Part G latency-injected regression).
- `E:\ARGUS\ARGUS\tests\retrieval\test_phase07f_parallelism.py` (11 accepted-check tests).
- Changed production files: `app/retrieval/router.py`, `app/retrieval/hybrid.py`, `app/orchestration/nodes.py` (Parts A & B).
- Control (unchanged, do not delete): `benchmarks/eval_data/` (12-doc corpus + 38-query plan) + `results/regression_07d.json` + `results/regression_07e.json`.