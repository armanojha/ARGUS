# PHASE 07B — PRODUCTION DEFAULT : FINAL REPORT

**Date:** 2026-09-03
**Baseline HEAD:** `c989a2b` (Phase 08 readiness audit committed, clean tree)
**Scope:** Make the already-built resilience + verification the DEFAULT runtime, with a bounded, fail-safe verification path on `/query`. A small hardening pass — **not** a new feature phase. Phase 08 was not started.
**Guiding principle from the audit (PHASE_08_READINESS_AUDIT.md §17–§18):** "does not lack features; it lacks default-on activation and validation of what already exists."

---

## 1. Objective

Execute the audit's recommended **"Phase 07b — Production Default"** pass:
1. Make the resilient `MultiModelRouter` the **default** router (health + fallback + quota active out-of-the-box).
2. Wire **selective claim verification** into the default `/api/v1/query` path, reusing the existing `verify_claim` engine (no new verification framework).
3. Keep call budgets bounded and verification **fail-safe** (it annotates, never replaces, a grounded cited answer).
4. Preserve the explicit single-provider escape hatch; keep all existing tests green.

This directly addresses the audit's two `P1` findings: **P0-1** (resilience off by default) and **P1-1** (no automated verification on the query path).

---

## 2. Initial Problem (from the audit)

- `app/config.py` had `multimodel_enabled` defaulting to `False`, so `get_router()` returned the single-provider `LLMRouter` on a stock install → the exact Phase 06.6 single-point-of-failure persisted despite Phase 07 building a correct resilient path.
- `app/orchestration/` **never** imported or called `verify_claim` → a synthesized, cited answer shipped with **no** automated claim-level fact-check or contradiction detection (the `verification` call type in `model_policy.yaml` was dead on the query path).

---

## 3. What Was Implemented (07b.1 — Default-on resilience)

- **`app/config.py`:** `multimodel_enabled` default changed `False` → `True`. A stock `Settings()` now enables the resilient router.
- **`.env.example`:** documented the new default + the explicit opt-out (`ARGUS_MULTIMODEL_ENABLED=false`).
- **`get_router()`** (`app/llm_gateway/__init__.py`) already branches on the flag and is **unchanged**: `multimodel_enabled=True` → `MultiModelRouter`; explicit `False` → `LLMRouter` via `create_provider`. No architectural change; only the default flag flipped.
- **Escape hatch preserved:** `multimodel_enabled=False` still returns the single-provider `LLMRouter`; the legacy `LLMRouter` class is untouched.

## 4. What Was Implemented (07b.2 — Selective verification on `/query`)

- **`app/orchestration/agents/coordinator.py`:** factored the 06.5.4 verification-gate into a reusable module-level predicate `should_skip_verification(plan, evidence, settings)` (single source of truth). The coordinator's `_should_skip_verification` now delegates to it, and the removed duplicate was cleaned up.
- **`app/orchestration/graph.py`:** added `_run_selective_verification(...)`, a post-synthesis, fail-safe step invoked once in `run_query`. It:
  - returns `skipped_reason="disabled"` unless `settings.verification_enabled`.
  - returns `skipped_reason="call_budget"` if `check_call_ceiling()` is already true (never spend the last budget slice on verification).
  - returns `skipped_reason="low_risk"` when the shared 06.5.4 gate says evidence is well-grounded (simple/low-risk answered are **not** verified).
  - otherwise builds a `VerificationRequest` (claim = the cited answer; `supporting_chunk_ids` = cited chunk ids; `entity_names`/`temporal_context` from the plan) and calls the **existing** `verify_claim` engine **at most once**, within `asyncio.wait_for(orchestration_llm_timeout + 5)`.
  - maps the engine result into the new `OrchestrationVerification` trace, including a populated `error` field on an engine-`ERROR` outcome.
- **`app/orchestration/models.py`:** added additive, defaulted-`None` `OrchestrationVerification` model and an `OrchestrationResult.verification` field (mirrors the existing additive `telemetry`/`question_pattern` pattern; `extra="forbid"` respected — the field is declared, not an extra key).

## 5. Verification Trigger Logic (honors 06.5.4, no new confidence framework)

Verification is **selective and deterministic** — no new threshold machinery. It fires (i.e. `triggered=True`) only when `verification_enabled` AND the call budget has headroom AND the shared gate reports a warning sign:
- `plan.risk_level ∈ {medium, high}`, **or** average evidence score < `multiagent_verify_threshold` (`_verify_threshold`), **or** average score < skeptic threshold, **or** high uncertainty (avg < 0.5), **or** conflicting evidence (score CV > disagreement threshold).

For simple/low-risk, high-confidence, non-conflicting evidence, verification is skipped (`skipped_reason="low_risk"`), preserving the cheap default path.

## 6. Failure / Timeout Behavior (fail-safe — never a new SPOF)

- Verification runs **after** the synthesize node has already produced a grounded, cited answer.
- The engine's own `_safe_structured_verification_call` returns a graceful `VerificationResult(status=ERROR)` on LLM failure/malformed output (no exception).
- A true exception or timeout is caught by `_run_selective_verification`'s `try/except` + `asyncio.wait_for`, mapped to `status="error"`.
- In **all** failure cases the grounded answer and citations are **returned unchanged**; verification only annotates `OrchestrationResult.verification`. Confirmed by tests `test_failure_does_not_crash_and_reports_error`, `test_timeout_fails_safe`, `test_answer_preserved_on_failure`.

## 7. Call-Budget Impact (bounded, Part of the design)

- Verification adds **at most one** LLM call, and only when the run is not already at the global ceiling (`check_call_ceiling()`, mirroring `multimodel_call_ceiling=16`).
- The `verification` call is counted in the running telemetry (it goes through `router.complete`), so the existing 16-call run ceiling bounds the total.
- `max_evidence_items` / the engine's prompt caps bound the verifier prompt size.
- **No loop:** `run_query` invokes `_run_selective_verification` exactly once; `verify_claim` performs a single structured call. Test `test_verifier_called_exactly_once` asserts exactly one invocation.

## 8. Tests Added / Updated

- **`tests/orchestration/test_phase07b_verification.py` (NEW, 14 tests):**
  - Part 1 (default-on): default config `multimodel_enabled is True`; `get_router()` on stock config returns `MultiModelRouter`; explicit `multimodel_enabled=False` returns `LLMRouter`.
  - Part 2 (verification wiring): disabled → skipped; low-risk/high-confidence → skipped (06.5.4); high-risk → triggered; low-confidence → triggered; call-ceiling → skipped; verifier called exactly once.
  - Fail-safe: failure → `status="error"` no crash; timeout → `status="error"`; answer preserved on failure.
  - Model: `OrchestrationResult.verification` defaults `None`; serializes.
- **`tests/test_llm_gateway.py`, `tests/test_smoke.py`:** the `get_router()` *singleton* tests exercised the legacy `LLMRouter` branch via `create_provider`; since the default is now MultiModelRouter, they now **explicitly** set `multimodel_enabled=False` (preserving escape-hatch coverage of the single-provider path).
- **`benchmarks/runner.py`:** the benchmark harness previously called `verify_claim` **again** after `run_query`. With verification now on the query path this was duplicative — the harness now reads `result.verification` instead, aligning the benchmark with live `/query` behavior and avoiding double verification. `tests/evaluation/test_benchmark_runner.py` still asserts `verification_status == "supported"`.

## 9. Test Results

- Full suite: **462 passed / 20 skipped** (baseline 448 passed + 14 new; no existing test weakened).
- `python -m pytest tests/test_runtime.py -W error::RuntimeWarning` → **3 passed**, clean.
- `ruff check` on all changed files → clean.

## 10. Live Validation (minimal quota)

Keys configured via `.env` (values never printed): **GROQ SET, GEMINI SET, ZEN SET, CEREBRAS EMPTY**. Two small real calls confirmed the default path is active and reliable:
- **Scenario A (default path):** `get_router()` returned a `MultiModelRouter`; a healthy call routed through the groq primary (`openai/gpt-oss-120b`) → `OK`. **The resilient path is the default.**
- **Verification call-type path:** a call with `call_type="verification"` routed to the gemini primary (`gemini-3.5-flash-lite`) and served correctly — the exact call type the new selective verification stage uses works live.
- Scenarios B/C/D (quota-exhausted fallback, unavailable fallback, verification trigger/skip) were NOT repeated against live providers to respect the "minimal quota" constraint; they are deterministically covered by the offline suite (14 new tests + the full-pipeline benchmark test `test_full_argus_pipeline_integration`, which runs real retrieval + orchestration + selective verification and asserts `verification_status == "supported"`).

## 11. Regression Checklist

- ✅ No secrets committed or logged (only 2 small real provider calls; no key output).
- ✅ No debug code; `ruff` clean.
- ✅ No unrelated modifications; `git status` shows only intended files.
- ✅ No model-policy / providers.yaml changes (configs untouched).
- ✅ No provider additions; no architecture duplication (verification reuses the existing engine).
- ✅ No dead verification path — it is now live on the default query path.
- ✅ No weakened tests — assertions preserved or strengthened.
- ✅ No hidden async/lifecycle issue — verification runs inside `run_query` using the same `router`; fail-safe `try/except`.
- ✅ No unbounded call loop — verification invoked exactly once, gated by the call ceiling.
- ✅ No default-path bypass of MultiModelRouter — default now selects it.
- ✅ Single-provider escape hatch intact (`multimodel_enabled=False` → `LLMRouter`).

## 12. Working-Tree Note (important)

Part-way through this task a commit authored outside this session (`d58e5dd`, "expand config, orchestration agents, graph, and test suites") landed on `main` **on top of** the baseline `c989a2b`, absorbing an early snapshot of the config/models/coordinator/graph edits. As a result:
- `app/config.py`, `app/orchestration/models.py`, `app/orchestration/agents/coordinator.py`, `.env.example`, `tests/test_llm_gateway.py`, `tests/test_smoke.py`, and an earlier revision of `app/orchestration/graph.py` are now part of `HEAD d58e5dd`.
- The **current working tree** (HEAD + uncommitted) is internally consistent and all 462 tests pass against it.
- Still **uncommitted** (my remaining Phase 07b work): the graph.py `error`-mapping refinement, the `benchmarks/runner.py` change, and `tests/orchestration/test_phase07b_verification.py`.

## 13. Remaining P2 Issues (NOT fixed — out of 07b scope per the audit's "minimal hardening, no P2 unless required for correctness")

- All orchestration LLM calls remain sequential (no safe parallelism) — audit P2-2.
- `reasoning`/`revision` model-policy entries still unused on the query path (audit P2).
- Evidence-context budget (audit `07b.3`) and graph multi-hop retrieval (audit `07b.5`) were **not** implemented — they are larger, optional items explicitly guarded in the audit and left for a decision, not bundled into this hardening pass.

## 14. Final Verdict & Recommended Next Action

**Phase 07b is COMPLETE and GREEN.** The default `/api/v1/query` path now:
- runs through the resilient **MultiModelRouter** (zero-config resilience),
- performs **selective, fail-safe claim verification** for medium/high-risk or low-confidence/conflicting evidence, reusing the existing engine, and
- stays **bounded** (single verification call, respected call ceiling) and **non-breaking** (grounded cited answers are never discarded on verification failure).

The audit's two P1 blockers are resolved with no new subsystems. **Do NOT start Phase 08.** The recommended next step, per the audit, is the optional `07b.3` (evidence-context budget) and `07b.5` (graph multi-hop) only if a *measured* gain justifies them, followed by a real live-provider validation run once full quota is available. The user should decide whether to commit the three uncommitted files and advance the vault state.