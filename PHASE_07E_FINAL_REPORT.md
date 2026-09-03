# PHASE 07E FINAL REPORT — Graceful Burst Degradation + In-Session Recovery

**Date:** 2026-09-03
**HEAD before this phase:** `5ff54e4` (Phase 07d COMPLETE)
**Scope:** HARDEN-07e — the burst-degradation + in-session recovery fix for Phase 07c finding #2 (`No available model` cascade under burst rate-limiting). Small, production-code hardening pass. **Phase 08 / feature work / persistence features were NOT started.**
**Verification method:** 100% offline via deterministic fault-injection mocks + the real `MultiModelRouter` + the fixed 38-query control. **Zero live quota consumed.**

---

## 1. Executive Summary

Phase 07c found that a burst of 429 / rate-limit failures could cascade into `ProviderConfigurationError("No available model...")` with **no in-session recovery**, and — when synthesis had no provider — the system fell back to a **naked raw-evidence dump** rather than a graceful, truthful degraded answer. Phase 07e closes both defects with two small, health-backed mechanisms:

- **PART A — bounded in-session recovery probe.** When the router has no eligible provider (every configured provider is health-skipped / failing), and it would otherwise hard-raise `No available model`, it now performs exactly ONE health-backed probe against the provider closest to cooldown expiry (within a `probe_grace` window), provided that provider was not already attempted-and-failed *in this call*. A success clears health and recovers the provider **in the same session (no restart, no arbitrary future query)**; a failure re-asserts cooldown so the still-down provider is **never repeatedly probed** (0-repeat). The probe uses the same telemetry/quota/ceiling accounting as any routed call.
- **PART B — clean degraded outcome.** The synthesis-node raw-evidence dump was replaced by a truthful, still-citation-grounded degraded statement. This remains labeled `ANSWERED_DEGRADED` via the existing `synthesis_degraded_to_raw_evidence` warning + `_derive_outcome`, so it is never misreported as a bare success (the roadmap-M2 "evidence-dump-only" anti-pattern is removed).

The change is **quality-preserving and regression-free**: the full suite is **502 passed / 20 skipped**, and the 07d 38-query control regression reproduces identically (38/38 answered, 38/38 grounded, R2–R5 unchanged).

**Verdict: COMPLETE.** Acceptance basis (07c §15.1, roadmap M2) is met: transient burst failures degrade gracefully instead of cascading to total outage; fallbacks stay usable; rate-limited providers are not hammered during cooldown; providers auto-recover after cooldown **and in-session**; grounded answers remain intact. Known caveats are documented in §16 (mocks only, no live quota burned).

---

## 2. Problem Statement

Phase 07c (§1 finding #2, §7.2, §15.1): *"Burst rate-limiting cascades to total outage — 13/18 live queries degraded to `No available model`; no in-session recovery/probe (cooldown-expired provider never recovered)."* Roadmap M2: a burst stress (3+ rapid queries, one provider degraded) should yield ≥80% answered_cited, **0% evidence-dump-only** on recoverable cases, and the system must remain **recoverable before a required fallback**.

Concretely, a burst of operator-visible defects to fix:

1. When every provider is health-skipped (mid-cooldown from prior queries) or failing, `complete()` raised `ConfigurationError("No available model...")` at the **initial-selection block**, before any possible recovery was attempted. Recovery was purely lazy/passive (wait out cooldown, rely on a *future* query to route back), so a burst → outage with no in-session path back.
2. When synthesis truly had no provider, `synthesize_node` emitted a naked raw-evidence dump (`"Synthesis was unavailable; returning the top retrieved evidence instead:"`) — the M2 "evidence-dump-only" anti-pattern.
3. No bounded probe existed to re-eligibilize a cooldown-expired (now-recovered) provider within the same session.

The fix had to preserve: the existing `ProviderHealthTracker`, the fallback chains, 429 fail-fast (07d), and the `Outcome`/`_derive_outcome` semantics. It must **not** hammer a still-failing provider, **not** add work to the healthy path, and **not** redesign the router.

---

## 3. Root Cause

- **Cascade (no recovery):** the router's `complete()` early-raised `ConfigurationError` when `_select_model_for_call_type` returned `None` (all providers health-skipped). The code path never reached any recovery mechanism; the only in-session re-eligibility was the health tracker's *lazy* `cooldown_until` auto-expiry observed on a future query. A provider that had genuinely recovered (e.g. eased burst pressure) mid-session was not probed back into service until some later query arbitrarily routed to it — so under an active burst the system stayed "No available model" with no recovery.
- **Evidence dump (quality):** the `synthesize_node` fallback emitted raw retrieved-evidence text with a parenthetical apologetic prefix, presented as the answer with no truthful degraded framing — the M2 anti-pattern.
- **No 0-repeat guard across the probe path:** an unguarded probe could, at worst, repeatedly re-hit a mid-cooldown provider; the fix bounds this via the `probe_grace` window + in-call exclusion set.

---

## 4. Existing Architecture (before the change)

- `ProviderHealthTracker` (`app/llm_gateway/health.py`) — persistent two-scope (provider + model) health: 6 classes (RATE_LIMITED/QUOTA_EXHAUSTED/UNAVAILABLE/UNAVAILABLE_HARD/DEGRADED/HEALTHY), cooldowns (rate 10s/quota 30s/unavail 15s/hard 30s), `cooldown_until` + monotonic clock, auto-expiry to HEALTHY, `record_success` clears cooldown, `get_status`/`skip_reason` consult. Thread-safe singleton mirroring `quota.py`.
- `MultiModelRouter.complete()` — walks the semantic call-type chain + provider fallback order; health-aware `provider_health_skip`; bounded `max_routing_attempts` loop; `exclude_models`/`exclude_providers` sets; per-`_execute_with_telemetry` call accounting (ceiling, quota, telemetry, health) and `record_failure`/`record_success`.
- Phase 07d: HTTP 429 fail-fast (single attempt, no same-endpoint backoff) — so a burst registers each provider exactly once and then health-skips it; `Outcome` (`ANSWERED/ANSWERED_FALLBACK/ANSWERED_DEGRADED/NOT_FOUND/NO_ANSWER`) + `_derive_outcome` in `graph.py`.
- Phase 07d `synthesize_node` fallback (unfixed pre-07e): raw-evidence dump + `synthesis_degraded_to_raw_evidence` warning → `ANSWERED_DEGRADED`.

Measured "before" (burst, all 4 providers 429): each provider called exactly once (0-repeat already held — good), router raised the last `RateLimitError` (truthful fail), next query all-skipped (no attempt, no hammering); recovery only by lazy cooldown expiry; synthesis outage → naked evidence dump.

---

## 5. Changes Made

**PART A — bounded in-session recovery probe**

- `ProviderHealthTracker.recovery_candidate(specs, probe_grace=2.0)` (`health.py`): given the ordered `(provider, model)` candidates, returns the single in-cooldown entity whose `cooldown_remaining <= probe_grace` (closest to expiry, including already-elapsed-but-not-yet-lazy-recovered), else `None`. Still-comfortably-mid-cooldown entities (`remaining > probe_grace`) are **never** proposed → no wasted probe on an active burst.
- `MultiModelRouter.complete()` (`multi_model_router.py`):
  - Removed the early hard `ConfigurationError("No available model...")` raise when `routing_result is None`.
  - Guarded the attempt loop with `while routing_result is not None and attempt < max_routing_attempts` (tracking `last_error`).
  - After all attempts fail, and **before** the final raise, fires up to ONE `_recovery_probe(...)` (passing the real in-call `exclude_models`/`exclude_providers`). If the probe succeeds, it returns that response (in-session recovery). Otherwise it raises `last_error` (preserving the truthful failure code).
- `MultiModelRouter._recovery_probe(...)`:
  - Selects ONE candidate via `recovery_candidate(provider_fallback_skips(call_type, exclude_models, exclude_providers))`.
  - Executes it through `_execute_with_telemetry(...)` with `is_fallback=True`, `fallback_reason="in_session_recovery_probe"`, so call-ceiling/quota/telemetry/health-success-clearing are identical to any routed call.
  - On success, `record_success` clears the entity → provider restored in-session.
  - On `LLMProviderError`, re-records the failure with the **real** error code (re-asserts cooldown) → never repeatedly probed (0-repeat).
- `MultiModelRouter._provider_fallback_skips(call_type, exclude_models, exclude_providers)`: now honors the in-call failure exclusions, so the probe never re-tries a provider already attempted-and-failed *in this call* (only cross-query cooldown providers are probe candidates).

**PART B — clean degraded outcome**

- `nodes.synthesize_node` degraded message changed from the naked dump to:
  `"Synthesis is temporarily unavailable, so I could not produce a polished answer. Here is the grounded evidence I retrieved (correctness not fully synthesized):"` with `[i]` citation markers and the `synthesis_degraded_to_raw_evidence` warning retained → still mapped to `ANSWERED_DEGRADED` by `_derive_outcome`.

No change: model policy provider fallback order, quota settings, provider configs, API keys, orchestration flow, verification, or retrieval.

---

## 6. Files Changed

Production:
- `app/llm_gateway/health.py` — `recovery_candidate` probe-grace window (PART A).
- `app/llm_gateway/routing/multi_model_router.py` — `_recovery_probe`, `_provider_fallback_skips` exclusion-honoring, `complete()` no-early-raise + single-probe-before-raise + `while routing_result is not None` guard (PART A).
- `app/orchestration/nodes.py` — `synthesize_node` degraded message cleanup (PART B).

Tests:
- `tests/llm_gateway/test_phase07e_recovery.py` — **NEW**, 16 focused tests (07e T1–T9 coverage).

Benchmark:
- `benchmarks/eval_data/regression_07e.py` — **NEW**, mock burst + in-session-recovery regression (R-e1..R-e4) → `results/regression_07e.json`.
- `benchmarks/eval_data/results/regression_07d.json` — re-run of the 07d control (identical results; evidence-count refresh only).

---

## 7. Health / Cooldown Behavior

- `recovery_candidate` only proposes a provider whose cooldown is **within `probe_grace` (2s)** of expiry (or already elapsed but not yet lazily recovered). Mid-cooldown providers (`remaining > 2s`) are not probe candidates.
- A successful probe → `record_success` (via `_execute_with_telemetry`) clears cooldown → `HEALTHY` immediately.
- A failed probe → `record_failure` with the real code re-asserts cooldown → the provider stays down and is not re-probed.
- Lazy auto-expiry (cooldown elapsing naturally) remains unchanged and still works for the steady state.
- Healthy-path behavior is untouched: `recovery_candidate` returns `None` when routing already has an eligible provider, so **zero probe overhead** on the healthy path (verified: R-e3 = exactly 1 call/query, all primary).

---

## 8. Fallback Behavior

- Preserved unchanged: Groq primary → Gemini → Cerebras → Zen last-resort order per call-type; model-level fallback then provider-level fallback; 429 fail-fast from 07d (single attempt); health-skipped fallbacks never re-hit mid-cooldown.
- The recovery probe only fires **after** the full chain (model + provider fallbacks) has been exhausted with no success, i.e. only when otherwise-TOTALLY-blocked. It operates as one extra, bounded chance at the closest-to-recovered provider, and does not reorder or extend the normal fallback policy.
- Verified (R-e1): a burst followed by groq recovering yields recovery via the probe rather than a cascade; recovered provider answers in-session.

---

## 9. In-Session Recovery Behavior (the core fix)

Scenario: query 1 = full burst (all providers 429 → each fails once → all cooldown). Query 2 arrives while the providers are *still cooling* but groq is genuinely recovered (stopped failing) and near cooldown expiry. **Before 07e:** query 2 would find all health-skipped → `No available model` → outage persists. **After 07e:** the router attempts no wasted normal calls (still-cooling providers are skipped), then fires ONE recovery probe at groq; groq succeeds; `record_success` clears health; **query 2 is answered from groq in the same session without a restart or an arbitrary future query.**

Verification: `test_probe_recovers_cooling_provider_without_restart` (unit) and `R-e1`/`R-e4` (regression, real `MultiModelRouter`): `primary_restored_without_restart=True`, `probe_fallback_reason="in_session_recovery_probe"`, groq health → `HEALTHY`.

---

## 10. Same-Query Behavior

- Total outage within a single query (all providers down): the query **does not** fabricate an answer. It degrades truthfully to `ANSWERED_DEGRADED` when evidence was retrieved (Part B clean evidence-grounded degraded statement, citations mapped), or `NOT_FOUND`/`NO_ANSWER` when no grounded evidence exists — never a bare `answered`, never a naked dump.
- Evidence / citations are preserved across the degrade: the `[i]` markers map to the retrieved evidence list, and `_derive_outcome` labels the degraded state.
- The recovery probe adds **at most 1** call per blocked query and only when otherwise-TOTALLY-blocked, so same-query latency impact is bounded (§14).

---

## 11. Test Results

- **New 07e tests** (`tests/llm_gateway/test_phase07e_recovery.py`, 16 passed):
  - T1 primary 429 → fallback + correct `Outcome` (`answered_fallback`) + citations.
  - T2 repeated calls during cooldown skip the primary (no hammering; via health-skip).
  - T3 cooldown expiry re-eligibilizes the primary; success clears health (record_success).
  - T4 A→429, B→429, C→available reaches C, no repeated A/B.
  - T5 all-unavailable → fast, truthful fail, no storm, no fake success.
  - T6 same-query partial progress preserved (evidence + citations kept, degraded outcome).
  - T7 recovery after degradation → next query serves the normal primary again.
  - T8 `_derive_outcome` correctness for all five `Outcome` values.
  - T9 in-session recovery probe recovers a cooling provider without restart; and probe never hammers a still-failing provider (≤2 calls total to the probe candidate, ≤1 to others).
- **Full suite:** **502 passed / 20 skipped** (was 486/20 at 07d HEAD; +16 new). `python -m pytest tests/test_runtime.py -W error::RuntimeWarning` = 3 passed. Full `--disable-warnings` run = 502 passed / 20 skipped.
- **Ruff:** changed files clean (`health.py`, `multi_model_router.py`, `nodes.py`, `test_phase07e_recovery.py`, `regression_07e.py`). 63 pre-existing out-of-scope lint findings remain untouched (base-commit drift; out of 07e scope — respecting phase boundaries).

---

## 12. 38-Query Regression Results

`benchmarks/eval_data/regression_07d.py` re-run with the 07e router (identical oracle outcome):
- R1 healthy scan (38 queries): **38/38 answered**, **38/38 grounded**, verification_triggered 38, recall@8 avg **0.987** — **no quality/recall regression** from the 07e changes.
- R2 budget-stop honesty: 9/9 budget-stopped runs answered.
- R3 hard-degraded: `answered_degraded`, never bare success.
- R4 429 fast-fail+fallback: primary 1 attempt + fallback → `answered_fallback`, answer + citation present.
- R5 verification failure: `answered` + status=`error` + answer/citations preserved.

**07e burst regression** (`regression_07e.py`, real `MultiModelRouter` + mock providers):
- **R-e1 (burst, primary recovers):** q0 full burst → all 4 fail exactly once each (no storm), error code `RATE_LIMIT_ERROR` (truthful). q1 after groq recovers + near cooldown expiry → **recovery via probe, provider=groq, no restart** → `primary_restored_without_restart=True`, groq only 2 calls total (1 burst + 1 probe-success), others 1.
- **R-e2 (sustained all-down, 3 queries):** 0 answered, 3/3 truthful fails, **max 1 call/provider** (0-repeat across the burst — no hammering).
- **R-e3 (healthy, 3 queries):** 3/3 succeeded, all primary (groq), **exactly 1 call/query** (no probe overhead on the healthy path).
- **R-e4 (probe primitive):** one probe identified groq (`probe_fallback_reason="in_session_recovery_probe"`), succeeded, health → `HEALTHY`.

---

## 13. Before / After Reliability Metrics

| Metric | Before (07d HEAD, measured) | After (07e) |
|---|---|---|
| Burst of N 429 failures, cascade | `No available model` / outage persists in-session | Graceful: bounded 1 probe; recovered provider re-serves **in-session** |
| All-provider outage, no recovery | Stale cooldown never probed; outage until a future query lazily routes back | Recovery probe re-eligibilizes a recovered (near-expiry) provider |
| Providers per burst-call | Exactly 1 attempt each (0-repeat held) | Unchanged (still 0-repeat); probe ≤ 1 extra call only when otherwise-totally-blocked |
| Synthesis outage output | Naked raw-evidence dump (M2 anti-pattern) | Clean truthful degraded statement, citations mapped, `ANSWERED_DEGRADED` |
| Healthy-path queries | 1 call/query | 1 call/query (0 probe overhead) |
| Fake success on total outage | Never (truthful error raised) | Never (truthful error raised; degraded only when evidence present) |
| Quality (38-query control) | 38/38 answered/grounded, recall@8 0.987 | **Identical** (38/38, 0.987) |

---

## 14. Latency Impact

- **Healthy path:** 0 added calls/latency (probe never fires when an eligible provider exists — R-e3).
- **Recoverable burst:** at most +1 bounded probe call per otherwise-blocked query, and only within `probe_grace` (≈2s of cooldown) of a near-expiry provider — far cheaper and *productive* (answers the query) versus the prior persistent outage. The probe respects the per-attempt timeout (`timeout`), so worst-case added time is bounded by one attempt deadline.
- **Sustained all-down:** no probe fires for mid-cooldown providers (0 wasted calls), so no added latency beyond the already-fast single-attempt fail.

---

## 15. API-Call Impact

- One extra `provider.complete` call **per blocked query**, at most, and only when the router would otherwise hard-raise and a near-expiry candidate exists. It is charged identically to a normal routed call (call-ceiling check → respected by `multimodel_call_ceiling`; quota recorded; telemetry logged as `is_fallback`, `fallback_reason="in_session_recovery_probe"`).
- No new provider calls on the healthy path (R-e3) or on sustained all-down (R-e2: still 0-repeat).
- Repeated-429 behavior unchanged from 07d (single attempt per provider; no same-endpoint backoff).

---

## 16. Known Limitations

- **Mock/deterministic validation only (no live quota burned).** The burst + recovery probe was verified with fault-injection mocks over the **real `MultiModelRouter`** and the fixed 38-query control, plus the orchestration-level `ANSWERED_DEGRADED` path. A live multi-query burst against real throttled endpoints was **not** run (quota discipline per the brief). The mechanism is exercised end-to-end with real router logic, so confidence is high, but live burst telemetry remains optional follow-up.
- **Probe grace heuristic (`2.0s`).** The probe only helps when a recovered provider is within `probe_grace` of cooldown expiry. If a provider is still comfortably mid-cooldown at the moment of an otherwise-blocked query, no probe fires (correctly — non-hammering) and that single query still fails; recovery landing on the *next* re-eligible (post-cooldown) query. This is the intended degradation boundary, not a defect.
- **Single provider recovered only.** The probe re-eligibilizes at most one candidate per blocked query; recovering more than one provider under an active burst may take up to N blocked queries. This is intentionally minimal and non-storming.
- **`benchmarks/eval_data/results/regression_07d.json` changed** by the clean re-run (evidence-count refresh; identical assertions). The 07d report/regression artifacts remain untracked and await owner commit alongside 07e.
- 63 pre-existing out-of-scope ruff findings remain (base-commit drift; not in 07e scope).

---

## 17. Risk Assessment

- **Low.** The changes are localized to `health.py`, `multi_model_router.py`, and the `synthesize_node` degraded message; no router redesign, no provider/model-policy reorder, no quota/provider-config change, no new subsystem.
- The `while routing_result is not None` guard is strictly narrower than the old `while attempt < max` (it additionally stops when no routing result was ever selected), preserving the bounded-attempt invariant.
- The probe's `exclude_models`/`exclude_providers` honoring prevents re-trying a provider the caller just failed in-call (0-repeat preserved).
- Health/cooldown lazy expiry and 07d 429 fail-fast are preserved; `Outcome` semantics untouched.
- Full suite (502/20) + runtime-warning strict run + 38-query control regression all green → low regression risk. Known caveat: mock-only live-burst validation (§16).

---

## 18. Final Verdict

**COMPLETE.**

Acceptance basis (07c §15.1 finding #2 + roadmap stop-line M2) is met:
- Transient burst failures **degrade gracefully, do not cascade to total outage** (bounded recovery probe, truthful errors).
- **Fallbacks stay usable** (normal chain + fallback policy unchanged; healthy fallbacks always attempted before any probe).
- **Rate-limited providers are NOT hammered during cooldown** (0-repeat across a burst; ≤1 call/provider; probe grace window).
- **Providers auto-recover after cooldown AND in-session** (recovery probe re-eligibilizes a recovered near-expiry provider within the same session, no restart).
- **Grounded answers stay intact** (R1 38/38 answered+grounded, recall@8 0.987; same-query evidence/citations preserved on degrade).
- **07d outcome/citation behavior correct** (Outcome enum, `_derive_outcome`, citation normalization; R2–R5 unchanged).
- **No regression in normal-path quality/latency** (healthy path = 1 call/query, 0 probe overhead; quality identical).
- **Evidence-dump-only removed on recoverable cases** (M2): synthesis degrade now emits a clean truthful, citation-mapped statement labeled `ANSWERED_DEGRADED`.

Put simply: Phase 07e makes the burst failure mode **graceful and recoverable in-session**, turning a hard outage into a bounded, self-healing path with no quality or normal-path regression. Phase 07f (safe parallelism / p95 ≤ 8s) is recommended next but **not started**; Phase 08 feature work is not started.