# Phase 08 Readiness Audit

**Date:** 2026-09-03
**Repository:** `E:\ARGUS\ARGUS` (branch `main`)
**HEAD:** `8c3628d` — `feat: add provider health and quota-aware routing` (Phase 07 CLOSED)
**Working tree:** clean
**Test baseline:** 448 passed / 20 skipped (20 = opt-in live provider tests gated by `RUN_LIVE_LLM_TESTS=1`)
**Method:** code-driven audit (the repository is the source of truth). No implementation, no policy changes, no benchmarks, no API quota consumed.

---

## 1. Executive Summary

ARGUS is already close to being an **advanced practical RAG**. Phase 07 correctly solved the single most damaging reliability gap (resilient provider orchestration after quota exhaustion). The system is grounded, cited, query-adaptive, and API-conscious. However, the audit found **one structural issue that materially undercuts the Phase 07 work in production**: the entire resilient MultiModelRouter path (health + fallback + quota) is **disabled by default** because `multimodel_enabled` defaults to `False`, so `run_query` uses the legacy single-provider router unless a deployment explicitly enables it.

Secondary findings (none P0, all P1/P2):
- Phase 04 verification + contradiction detection are **not wired into the default `/query` path** — a synthesized answer is returned with no automated post-hoc fact-check.
- The `reasoning` and `revision` call types are declared in the model policy but **never invoked** on the query path (dead configuration, not dead code).
- The evidence graph is built at ingestion but **not consulted during query answering** (multi-hop retrieval unused on the main path).
- No explicit cap on the number of evidence chunks fed to synthesis/assessment (context growth across iterations).
- All LLM calls in the orchestration loop are **strictly sequential** — no safe parallelization is used anywhere.

**Recommendation: `DO NOT START PHASE 08 YET`.** The correct next step is not another implementation phase. It is a small, tightly-scoped "Phase 07 completion/hardening" pass (make Phase 07 active by default and close the verification gap with non-default behavioral changes), followed by a clean, quota-free validation run of the *live* resilience path. See §16–§18.

---

## 2. Current ARGUS Architecture

Module map (all paths under `app/`):

```
api/           FastAPI layer: /query (run_query), /verify (verification engine),
               /retrieval, /telemetry, /health
llm_gateway/   LLMRouter (single-provider) + MultiModelRouter (Phase 07),
               ProviderHealthTracker (health.py), QuotaTracker (quota.py),
               complexity classifier, telemetry, capabilities, provider registry
orchestration/ LangGraph StateGraph (analyze→plan→retrieve→assess→stop→synthesize),
               active evidence seeking, adaptive stopping logic, multi-agent debate
retrieval/     Hybrid (BM25 + FAISS vector), retrieval policy router, adaptive gap seeking
reranking/     Cross-encoder reranker (local) with NoOp fallback
evidence/      EvidenceStore (+ BM25/FAISS index assignment)
graph/         EvidenceGraphStore + retrieval (ingestion-time; not on /query path)
verification/  verify_claim engine (Phase 04) + contradiction + gaps + confidence
memory/        Phase 08 memory (off by default; memory_enabled=False)
integrations/obsidian  (off by default)
ui/            Streamlit
benchmarks/    runner + metrics + ablation + 110-item dataset
runtime.py     lifecycle shutdown
configs/       providers.yaml, model_policy.yaml, retrieval_policy.yaml
```

**Default feature flags:** `True` by default — `retrieval_policy_enabled`, `active_evidence_seeking_enabled`, `stopping_logic_enabled`. `False` by default (must be explicitly enabled) — `multimodel_enabled`, `memory_enabled`, `multiagent_enabled`. Note `verification_enabled=True` but it is not active on the default `/query` path (see §8).

---

## 3. Phase 07 Outcome

Phase 07 delivered, and the code confirms it is correctly integrated into `MultiModelRouter`:
- `app/llm_gateway/health.py` — two-scope `ProviderHealthTracker` (provider-wide + model-scoped), cooldown windows, failure classification (auth/timeout/rate-limit/quota configured; `CALL_CEILING_EXCEEDED` deliberately excluded), success-reset recovery, wall-clock expiry. Thread-safe (single `Lock`).
- Router wiring (`multi_model_router.py`): health consult (`skip_reason`) before each candidate in both the call-type chain and provider-fallback loop; `record_failure` with scope derived from exception class; `record_success` clears provider+model scope; health surfaced in `get_available_models()`.
- `runtime.py`: health tracker closed on shutdown with quota tracker (idempotent, non-fatal).
- 12 new tests; full suite 448 passed / 20 skipped; `-W error::RuntimeWarning` clean.

**Verdict on Phase 07 integration: correct, minimal, and test-backed.**

---

## 4. Problems Solved

From Phase 06.6 (groq daily-quota exhaustion degrading DEFAULT runtime to raw-evidence recall 9/10):

1. **Resilient provider orchestration** — the router now fails over to a healthy provider instead of depending on a single provider. Verified live: Scenario B (quota-exhausted) and Scenario C (unavailable) both skip groq and synthesize a cited, grounded answer via Gemini.
2. **No repeated dead-provider waits** — health/cooldown skips a known-unhealthy provider at routing time (`provider_health_skip`), rather than paying a 15s-per-attempt timeout repeatedly. `attempt_ceiling_s=15` bounds each attempt.
3. **Failure classification** — timeout vs rate-limit vs quota vs auth are distinguished and mapped to different cooldown severities; model-scoped vs provider-scoped is preserved so intra-provider fallback stays reachable.
4. **Quota awareness** — `QuotaTracker.can_make_request` (provider + per-model) is checked before each candidate, avoiding calls that are certain to be quota-blocked.
5. **Observability** — per-routing-decision telemetry (provider, model, fallback, latency, tokens), call ceiling (16), health state in `get_available_models`.

---

## 5. Problems Remaining

1. **P1 — Phase 07 resilience is OFF by default.** `multimodel_enabled` defaults `False`; `get_router()` returns the single-provider `LLMRouter` in the default deployment. The exact failure mode Phase 06.6 documented can still occur on a stock ARGUS install. The MultiModelRouter code is correct; it is simply not the default.
2. **P1 — No automated verification on the query path.** `run_query` → synthesize returns a cited answer with **no** LLM verification or contradiction detection. Phase 04 (`verify_claim`) is reachable only via the standalone `/api/v1/verify` endpoint, the benchmark runner, and the (off-by-default) multi-agent debate. For a project whose headline is "evidence verification," this is a material gap in the core product path.
3. **P2 — `reasoning`/`revision` call types are defined but never invoked** on the query path. Stale policy entries (not harmful, but misleading for the "right model for right operation" goal).
4. **P2 — Evidence graph (Phase 03) is not used during answering.** Built at ingestion; multi-hop retrieval (`graph/retrieval.py`) is not consulted in `run_query`.
5. **P2 — No explicit cap on evidence context length** fed to the assess/synthesis prompts; it grows with iterations.
6. **P2 — All orchestration LLM calls are sequential**; no safe parallelism (e.g., parallel sub-question retrieval, parallel verification of independent claims).
7. **P3 — No live coverage in the default test run** — the 20 skipped tests are the *only* tests exercising the real provider/resilience path; they need `RUN_LIVE_LLM_TESTS=1`.

---

## 6. Runtime Efficiency Analysis

### Query-type execution path (from code: `graph.py`, `nodes.py`, `complexity.py`, `multi_model_router.py`)

The zero-LLM complexity classifier (`classify_complexity`) and the graph entry router decide fast vs deep path. No LLM cost is spent on routing.

| Query type | Example | Tier (`classify_complexity`) | Graph path | Sequential LLM calls |
|---|---|---|---|---|
| **A. Simple factual** | "What is the refund period?" (<60 chars) | FAST | retrieve→synthesize (fast path skips analyze/plan/assess) | **1** (synthesis) |
| **B. Normal doc question** | "How does the return process work?" (no strong signals, >60 chars) | BALANCED | analyze→plan→retrieve→assess→stop→synthesize | **4** (analyze, plan, assess, synthesize) |
| **C. Multi-document** | comparative/multi-part (>160 chars or rel. signals) | STRONG/BALANCED | deep + assess loop; up to `max_iterations=3` retrieve/assess passes | **4 + N×assess** (up to ~7) |
| **D. Complex research** | "Compare the impacts of X and Y ..." | STRONG | deep + iterate; STRONG tier → gpt-oss-120b for synthesis/analysis | **4 + N×assess** |
| **E. Conflict question** | conflicting evidence between sources | deep | no automated contradiction detection in default (off) | **4 + N×assess; NO verifier** |
| **F. Absent-information** | no evidence in corpus | deep | consecutive empty retrievals (≥2) → stop `NO_NEW_EVIDENCE`; degrades to evidence summary | **assess stops early; then synthesis or degraded summary** |

### Key efficiency findings

- **Fast path is genuinely cheap:** 1 LLM call for simple lookups. Excellent — matches "simple queries stay simple."
- **The deep path is 4 LLM calls minimum, all sequential** (`query_analysis`, `research_planning`, `evidence_extraction`, `synthesis`). No step is parallelized.
- **Retrieval/reranking/stopping/plan-fallback are deterministic** (0 LLM). Only assess/synthesis/analyze/plan cost tokens.
- **MultiModelRouter.complete()** makes 1 physical provider call normally (primary, when healthy + within quota); worst case it walks the full chain + provider fallbacks (up to ~9 attempts) but is hard-capped by the global 16-call run ceiling.
- **Unnecessary calls:** none structurally *required* beyond the pipeline. The main inefficiency is sequential assess-loop + full-context re-send each iteration.
- **Latency budget:** each LLM call is sequential; groq/gemini single calls are ~1–3s (live validation). A 4-call deep path has floor latency ≈ 4×(1–3s) + retrieval time ≈ 5–13s. Multi-iteration paths (C/D) grow linearly.

---

## 7. Retrieval Efficiency Analysis

- **Hybrid fusion** (BM25 + FAISS) is weight-normalized RRF-style; `mechanisms` set already lets the policy router skip vector (and its embedding round-trip) on lexical patterns (06.5.5) — good.
- **Top-K:** `orchestration_retrieval_top_k=8`, `retrieval_top_k` default. Reasonable.
- **Reranking:** cross-encoder, local (not an LLM call). NoOp fallback. Not redundant — applied once per retrieval pass.
- **Evidence accumulation:** each iterate appends up to `top_k` new refs; `_merge_evidence` dedups by `chunk_id`. Duplicate retrieval is avoided via `issued_subqueries` dedup of next-subquery strings. Good.
- **Retrieval depth/adaptive:** active evidence seeking (gap detector) queues targeted follow-up queries; stopping logic (`consecutive_empty_retrievals>=2`, budget, max_iterations) bounds the loop. Deterministic.
- **Weakness (P2):** no cap on the *total* number of evidence chunks forwarded to the assess/synthesis LLM. Over 3 iterations × 8 top_k, up to 24 chunks (truncated to 500 chars each ≈ up to ~12k chars / ~3k tokens) is re-sent whole on each assess AND the final synthesis. Correct but token-heavy for multi-hop questions.
- **Graph retrieval is NOT used on the query path** (0 calls). So no redundant graph retrieval — rather, an unused retrieval capability. `run_query` never touches `graph_store`.

---

## 8. Verification Efficiency Analysis

- **Selective verification exists but is debate-path-only.** `coordinator._should_skip_verification` (06.5.4) suppresses verifier/skeptic/alt-hyp debate when evidence is high-confidence/low-risk — good, but reachable only when `multiagent_enabled=True`.
- **In the default `/query` path there is NO verification.** `app/orchestration/` never imports or calls `verify_claim`. The `verification` call type (Gemini primary) is thus never exercised by the live query API.
- The benchmark runner *does* call `verify_claim` after synthesis — which is why the 06.6 benchmark reported `verification_status=supported`. That is a benchmark-harness behavior, not the API behavior.
- **Consequence (P1 for answer-quality/reliability):** a user hitting `/api/v1/query` gets a synthesized, cited answer with **no automated claim-level fact-check or contradiction detection**, even when evidence quality is low or conflicting. The `verification_enabled=True` setting exists but is not active on the query path.

---

## 9. Model Routing Analysis

The router genuinely respects health → quota → capability ordering before each candidate:
- **Capability matching:** `_validate_capabilities` (structured_output, tools, context) gates candidates; e.g., groq gpt-oss-120b is rejected for the 12-field `ResearchPlan` strict schema and Gemini serves it (documented in `model_policy.yaml`).
- **Task matching / specialization:** driven *only* by the explicit per-call-type chains in `model_policy.yaml` + the complexity `tier` (fast/strong) for `query_analysis`. There is **no** dynamic per-task model selection — by design (the policy comment states model choice is owner-controlled, not autonomous). This matches the project philosophy.
- **Provider health + quota:** both checked before each candidate. **Correct.**
- **Fallback ordering:** Groq→Gemini→Cerebras→Zen; Zen kept last-resort. Zen timeouts/429s bounded by `attempt_ceiling_s=15` and health cooldown. Reasonable.
- **Weaknesses:**
  - P1 — **Active only when `multimodel_enabled=True`.** Default deployment = single provider, no specialization, no fallback.
  - P2 — `reasoning`, `revision`, and `verification` call types are defined but unused on the query path, so that portion of the "right model for right operation" policy is dead.
  - P2 — No latency-based selection: all models in a chain are treated as equivalent aside from order; there is no "prefer the fastest healthy provider" signal beyond the static chain order.

---

## 10. Latency Analysis

Not relying on the contaminated 06.6 numbers. Code-path estimates (healthy groq/gemini, single ~1–3s calls):

| Contributor | Estimate | Notes |
|---|---|---|
| Retrieval (BM25+vector+rerank) | ~0.1–1s | local vectors/BM25/cross-encoder, index already built |
| Fast path | **~1–4s** | 1 synthesis call + retrieval |
| Deep path (single pass) | **~5–13s** | 4 sequential calls + retrieval (floor) |
| Multi-iteration (C/D) | linearly +assess | each assess+retrieve adds ~2–5s |
| Provider fallback | +attempt time per hop | bounded by attempt_ceiling_s=15 + health cooldown skip |
| Verification (if enabled) | + one Gemini call ~1–3s | currently absent on query path |

**Bottlenecks:** (i) sequential 4-call pipeline; (ii) multi-iteration re-retrieval re-sends *all* accumulated evidence each assess; (iii) no parallel sub-question handling. These are the dominant controllable latency drivers. Provider health fallback is now amortized (skip, not wait).

---

## 11. API / Token Efficiency Analysis

- **Good:** zero-LLM complexity routing; deterministic retrieval/rerank/stopping; single synthesis for simple queries; structured-output gates avoid bad-format retries; per-attempt ceiling bounds wasted spends; health/quota skip avoids pointless calls.
- **OSignificant token concern (P2):** every assess iteration re-sends the **entire accumulated evidence** (truncated 500 chars each) to the LLM, and synthesis does the same. For a 3-iteration query this is ~2–4× redundant token spend on the same evidence. No per-call evidence-budget cap.
- **Total-call ceiling (16)** bounds worst-case spend but is high relative to the ~4-call happy path — the ceiling is a safety net, not the efficiency lever.
- **No LLM calls wasted on pure deterministic work** (e.g., the fast-path plan is a hardcoded `_fast_path_plan`).

Net: already API-conscious; the main lever is context/token reuse across iterations and avoiding the sequential re-send.

---

## 12. Reliability Analysis (conceptual, mapped against code guards)

| Failure | Behavior | Status |
|---|---|---|
| Provider outage (groq down) | health records, router falls back to Gemini | ✅ (when multimodel enabled) |
| Quota exhaustion | health/quota skip → fallback | ✅ |
| Rate limit (429) | health RATE_LIMITED cooldown → skip | ✅ |
| Timeout | attempt_ceiling + health UNAVAILABLE → skip | ✅ |
| Malformed/empty response | normalized to retryable/unavailable (06.6 unblock) | ✅ |
| Model-specific failure | model-scoped health, intra-provider fallback kept | ✅ |
| Whole-provider auth failure | provider-scoped health block | ✅ |
| Partial retrieval failure | `except` → `results=[]`; loop stops efficiently (empty→NO_NEW_EVIDENCE) | ✅ |
| Synthesis LLM failure | degrades to deterministic evidence summary (still grounded, no collapse) | ✅ |
| Entire run timeout | `orchestration_timeout` (120s) wraps `graph.ainvoke` | ✅ |
| **Router never selected resiliently by default** | `multimodel_enabled=False` → single provider | ⚠️ **single point of failure remains in stock config** |

**Remaining SPOF in default config:** if `multimodel_enabled` is not explicitly `true`, ARGUS is back to a single provider with no health/fallback — the exact Phase 06.6 failure. This is the single most important reliability caveat.

---

## 13. Production Engineering Analysis

- **Clean:** runtime lifecycle is idempotent/non-fatal; health tracker closed on shutdown; telemetry persisted as run trace with JSONL option; structured logging throughout; prompt-injection posture honored (evidence wrapped as untrusted data).
- **Dead/stale config:** `reasoning` and `revision` call types in `model_policy.yaml` are never used on the query path. Not harmful, but violates the "right model for right operation is realized" promise and should be reconciled or wired.
- **Dead-ish modules:** `reasoning`/`revision` are stale; graph `retrieval.py` is unused on the query path; `app/observability/` is empty.
- **Fragile-but-accepted:** singletons (`evidence/store`, `graph/store`) are documented thread-unsafe for multi-worker; `_conn()` private method is reached by external modules. These are known/accepted MVP trade-offs, not Phase 07 regressions.
- **Test-only assumptions:** 20 skipped tests are the ONLY live-provider coverage; the default suite never proves the resilience path works against real APIs. The MockProvider-based fabric tests are excellent for invariants but can't catch provider-specific quirks (e.g., Zen empty-choices envelope, Groq ResearchPlan schema limitation).
- **Docs:** `model_policy.yaml` accurately documents the Zen demotion; the post-06.5 validation note is current. No architecture/doc mismatch of consequence. Workplan/state/handoff in `E:\ARGUS_VAULT` are current.

---

## 14. Priority Classification

**P0 — Must fix** (blocks "advanced practical RAG" claim in a stock deployment):
- **P0-1:** `multimodel_enabled` should default `True` (or the resilient path must otherwise be the default). Right now Phase 07's entire value is opt-in; a stock install keeps the Phase 06.6 failure mode. *(Data-point: the whole point of Phase 06.6 → 07 was to stop depending on a single provider; leaving it off by default negates that.)*

**P1 — Important** (materially affect answer quality / reliability):
- **P1-1:** Wire selective LLM verification / contradiction detection into the `/query` result path (at minimum for medium/high-risk or conflicting/low-confidence evidence), honoring `_should_skip_verification` so simple/low-risk answers are not verified.
- **P1-2:** Add live-provider validation of the *default* deploy path (a `RUN_LIVE_LLM_TESTS=1` run over the healthy/quota-exhausted/unavailable scenarios) so the resilience proof is part of CI rather than opt-in-and-skipped.

**P2 — Nice to have** (efficiency / polish):
- **P2-1:** Cap total evidence-token budget fed to assess/synthesis (e.g., keep the best-K most-relevant chunks or truncate to a fixed context window per call) to stop re-sending all accumulated evidence each iteration.
- **P2-2:** Safe parallelization of the assess-loop or independent claim verification (only where dependencies allow).
- **P2-3:** Reconcile `reasoning`/`revision` policy entries (wire or remove) so the policy reflects the actual path; consider consulting the evidence graph for multi-hop (only if it measurably improves quality without hurting latency).

**P3 — Do not touch:**
- Provider net-new, DB/graph/memory/agent expansion, UI framework change, model-policy reshuffle, benchmark harness rewrite, huge refactors. These add complexity without addressing the actual gaps.

---

## 15. Is ARGUS Already an Advanced Practical RAG?

**ALMOST — with one material caveat.**

Justification:
- ✅ **Grounded & cited:** answers cite numbered evidence; provenance preserved even on degradation (no hallucination; degrades to evidence summary, never fabricates).
- ✅ **Query-adaptive:** zero-LLM complexity routing; simple queries cost 1 call; deep queries get plan+assess+planning.
- ✅ **API-efficient:** no wasted calls on deterministic work; health/quota/capability gates; per-attempt ceiling.
- ✅ **Resilient (when enabled):** Phase 07 MultiModelRouter health/fallback is correct and live-validated.
- ⚠️ **Resilient mode is NOT the default** → a stock ARGUS still has a single-POF.
- ⚠️ **No automated verification on the live query path** → the "verification" that defines the project is opt-in, so answer *correctness confidence* is less assured in the default product.

Because P0-1 and P1-1 are both real, "advanced practical RAG" is *almost* true but not yet guaranteed out-of-the-box. The next step is to make the already-built resilience + verification the default and validate it — not to add more subsystems.

---

## 16. Phase 08 Recommendation

**DO NOT START PHASE 08 YET** as a new feature phase.

The reason: the system does not lack features; it lacks **default-on activation and validation of what already exists**. Starting a Phase 08 (memory, agents, more retrieval, etc.) would add surface area on top of (a) a resilience path that is currently opt-in and (b) a verification path that is disconnected from the primary API. That is "more features," not "better RAG," and violates the phase-boundary discipline.

The correct follow-on is a short, focused **Phase 07.19-style hardening+validation** (or "Phase 07b — Production Default") that:
1. Makes the resilient router the default.
2. Wires selective verification into the query result path.
3. Runs a clean, quota-minimal live validation of the default path (healthy / quota-exhausted / unavailable / absent-evidence).
4. Optionally tightens the evidence-context budget.

Only after those are proven should the owner decide whether a genuinely new capability phase is warranted — and that decision should be driven by measurement, not by a generic roadmap.

---

## 17. Proposed Phase 08 Architecture/Plan

*Phase 08 as a new feature phase is **not** yet justified. The proposed follow-on (Phase 07b — Production Default) is:*

**Objective:** Ship the already-built resilience + verification as the default runtime and prove it with low-quota live validation.

**Exact problems solved:** P0-1 (resilience off by default), P1-1 (no automated verification on `/query`), P1-2 (no live validation in CI), P2-1 (evidence-context growth).

**Subphases:**
- **07b.1 (Default-on)** — flip `multimodel_enabled` default to `True`; ensure `get_router()` and `/query` use MultiModelRouter; add a config-level test asserting the default is resilient. *(Behavioral/config change, no architecture change.)*
- **07b.2 (Selective verification)** — add a deterministic, zero-LLM gate in the graph that, for medium/high-risk or low-confidence/conflicting evidence, invokes the existing `verify_claim` on the synthesis claims (reusing the `verification` call type already in policy and `verification_enabled` setting). Simple/low-risk answers skip it (honoring 06.5.4's skip logic). Respect `call_ceiling`.
- **07b.3 (Evidence-context budget)** — cap the number/bytes of evidence forwarded to assess/synthesis (best-K by score) so iteration N doesn't re-send all accumulated chunks.
- **07b.4 (Live validation)** — run `tests/test_provider_contract_live.py` and a small orchestration smoke (healthy + simulated quota-exhaust + unavailable + absent-evidence) with `RUN_LIVE_LLM_TESTS=1` to prove the default path. Minimal quota.
- **07b.5 (Optionally wire graph multi-hop)** — only if a measured gain justifies it (guard against latency/context overhead).

**Tests required:** default-on config test; verification-on-query-path test (scripted provider); evidence-budget test; keep all 448 existing green.

**Measurable success criteria:** default `/query` returns a cited answer with automated verification for risky/conflicting cases; no raw-evidence collapse when primary is exhausted (default config); deep-path tokens no longer grow unboundedly with iterations; all existing tests stay green; live smoke passes with minimal quota.

**Scope boundaries (explicitly NOT this phase):** no new providers/databases/agents/memory/graph/multimodal; no model-policy change; no architecture rewrite; no UI change; no benchmark harness rewrite.

---

## 18. Acceptance Criteria (for the follow-on, if adopted)

1. `multimodel_enabled` default `True`; a stock `scheme`/`get_router()` returns the resilient `MultiModelRouter`.
2. `/api/v1/query` performs selective verification: verification runs when `risk ∈ {medium,high}` OR evidence avg-score < threshold OR conflicting evidence; skipped for simple/low-risk answers.
3. Evidence-context to assess/synthesis is bounded (best-K), and deep-path token estimate stops growing monotonically across iterations.
4. Live smoke (default path) passes for: healthy → cited answer; groq quota-exhausted → Gemini cited answer; groq unavailable → Gemini cited answer; absent evidence → grounded "no evidence" answer (no hallucination).
5. Full suite ≥ 448 passed / 20 skipped; no test weakened; `-W error::RuntimeWarning` clean.

---

## 19. What We Explicitly Will NOT Do

- **Not** add new providers, databases, memory systems, agent frameworks, graph stores, or multimodal systems.
- **Not** change the model policy (`model_policy.yaml`) or provider assignments.
- **Not** rewrite working components (router, health, retrieval, orchestration, verification).
- **Not** run broad/budget-heavy benchmarks or burn API quota.
- **Not** weaken any existing test or remove the documented Zen free-tier instability handling.
- **Not** make unrelated cleanup/refactor changes.
- **Not** auto-implement a speculative Phase 08.

---

## 20. Recommended Next Action

1. **Stop here** — do not begin a new feature phase.
2. Obtain **owner decision** on whether to run the small, focused **"Phase 07b — Production Default"** pass described in §17 (default-on resilience + selective verification + evidence budget + low-quota live validation), rather than a Phase 08.
3. If approved, execute 07b with the acceptance criteria in §18 and a clean recovery-point commit.
4. Only after 07b is proven should a genuine new capability phase be scoped — and it should be driven by measured gaps (e.g., retrieval recall on shared corpora, per the 06.6 observation), not by a generic roadmap.

---
**Reset:** `DO NOT START PHASE 08 YET.` The audit recommends a default-on hardening + validation pass, not a new feature phase.