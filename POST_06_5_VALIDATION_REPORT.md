# POST-06.5 REAL-WORLD VALIDATION REPORT

Date: 2026-09-03 · Phase: 06.5 (HARDEN-06) COMPLETE · Base: `e8e8e5c` → `8e73878`

## 1. Validation scope (per the 9-step brief)

Controlled, evidence-driven inspection of the actual ARGUS implementation — not the
intended architecture. No new features were added. Commit kept clean. The existing
06.5 work was already committed (4 commits: `b9fe522`, `cf1dd23`, `a055a4d`, `8e73878`);
working tree is clean at HEAD.

- Test suite: **436 passed / 20 skipped** (normal run).
- `tests/test_runtime.py -W error::RuntimeWarning` → **3 passed**, no coroutine warnings.

## 2. A. Runtime path verified (STEP 2) — what actually executes

`run_query()` → build graph → index→ retrieval → loop → synthesis. Real wiring per settings:

| Stage | Default (`multimodel_enabled=False`) | Opt-in (`multimodel_enabled=True`) |
|---|---|---|
| Router | `LLMRouter(GroqProvider)` single-provider | `MultiModelRouter` (zen primaries + groq/gemini/cerebras fallbacks) |
| Analysis | LLM, unless fast path | LLM, tier-aware |
| Planning | LLM | LLM, tier-aware |
| Retrieval policy router | **Active** (hybrid + vector; BM25-skip when covered) | **Active** |
| Active evidence seeking / stopping | **Active** | **Active** |
| Memory / Multi-agent | **OFF by default** (neither initialized) | OFF unless enabled |

Per-query-type actual path:
- **A simple** ("What is the refund period?"): `classify_complexity==FAST` → **fast path** → `retrieve → synthesize` (analyze/plan/assess skipped). 1 logical LLM call.
- **B multi-part**: full `analyze → plan → retrieve → assess → stop_check → synthesize`.
- **C complex comparative**: full path; strong-tier models; assess+stopping decide sufficiency.
- **D cross-document**: full path; can iterate (observed `iterations=2`) when evidence spans docs.
- **E weak evidence**: full path; correct honest decline (no hallucination).

## 3. B. Providers / models actually tested (STEP 3)

Credentials from `.env` (never printed). Tested **live via the real LLM fabric and the
06.5.7 contract suite** plus direct probes.

| Provider | Key | Simple completion | Structured output | Fallback | Result |
|---|---|---|---|---|---|
| groq (`openai/gpt-oss-120b`, `-20b`) | present | ✅ (adequate max_tokens) | ✅ | n/a (default single) | **WORKS** |
| gemini (`gemini-3.5-flash-lite`) | present | ✅ | ✅ | ✅ | **WORKS** |
| zen (`nemotron-3-ultra-free`, `big-pickle`, `mimo-v2.5-free`) | present | ❌ repeated timeout (15s ceiling); `big-pickle` 429 under free tier | ❌ timeout | ✅ falls back to groq | **UNRELIABLE (free-tier latency/rate-limit)** |
| cerebras | **absent** | — | — | — | **UNTESTED — CREDENTIAL UNAVAILABLE** |

> NOTE: the 06.5.7 live suite (`RUN_LIVE_LLM_TESTS=1`) reported groq/gemini/zen failures.
> Root cause investigated: the harness calls with `max_tokens=16/32`, which is too small
> for these providers' JSON-schema-enforced responses → truncated/invalid output (zen also
> times out independently). At realistic `max_tokens`, groq and gemini **pass** the full
> contract. This is a **test-harness calibration issue**, not a production code defect.

## 4. C/D. Real document → answer (STEP 4) & complexity behavior (STEP 5)

Controlled 5-document corpus (factual, technical ×2, cross-doc pair, conflicting). Real
production pipeline (ingestion → chunk→ index→ retrieve→ plan→ synthesize). All answers
**factually correct, grounded, and cited**.

### Default runtime (what a real user gets): FAST & correct

| Query | LLM calls | Retrievals | Citations | Verif. | Latency |
|---|---|---|---|---|---|
| A simple | **1** (synthesis) | 1 (vector+hybrid) | 3 | skipped (fast path) | **1.4s** |
| C complex | 4 (analyze/plan/assess/synth) | 1 | 2 | none (multi-agent off) | **7.0s** |
| E weak | 4 | 1 | 3 | none | **3.9s** |

- A: "Customers can request a full refund within 30 days [1]" ✓
- C: produces a full **comparison table** with per-row `[1]`/`[2]` citations, correctly
  attributing refund_exceptions vs refund_policy. Excellent grounded analytic RAG.
- E: "evidence does not contain any information about a policy regarding pets" — **no
  hallucination** (textbook weak-evidence behavior).

### MultiModelRouter mode (opt-in): equal quality, ~15–60× slower

| Query | LLM calls | Latency |
|---|---|---|
| A simple | 1 | **23s** |
| B multi | 4 | **93s** |
| C complex | 4 | **90s** |
| D cross-doc | 4 | **91s** |
| E weak | 4 | **90s** |

Root cause: configured **zen primaries time out (~15s each)**; the fallback correctly
excludes zen and serves via groq, but every logical call pays a ~15s dead-wait before
fallback. Per-call attempt ceiling (`llm_attempt_ceiling_s=15`) bounds the damage but
does not remove it.

## 5. E–I. Retrieval / citation / verification / efficiency / latency observations

- **Retrieval relevance:** high. Correct source per query (refund docs for C, telemetry docs
  for D). Fusion + dedup by chunk_id works. BM25-skip optimization observed
  (`policy_hybrid_limited mechanisms=['bm25']`).
- **Citation correctness:** markers mapped to the right source_path/chunk/document. Both
  `[n]` (groq) and `【n】` (refinement) styles extracted correctly.
- **Verification:** multi-agent **off by default**; the 06.5.4 selectivity gate and debate
  are opt-in. Not triggered in these runs (expected).
- **LLM-call efficiency:** fast path = **1 call** for simple queries — excellent
  (06.5.3 works). Complex = 4 calls; reasonable, no redundant calls observed.
- **Latency:** default mode 1.4–7s (good). Multi-model mode 20–90s (net of zen dead-wait).

## 6. J/K/L. Failure & fallback behavior (STEP 7) — verified

| Scenario | Observed | Severity |
|---|---|---|
| Provider fully unavailable | Graceful → "Synthesis was unavailable; returning top retrieved evidence [1]"; no crash, no endless retry, no hallucination | ✅ OK |
| Empty/malformed synthesis | Same safe degradation, explicitly flagged | ✅ OK |
| Weak/absent evidence | Honest "no evidence" answer | ✅ OK |
| Rate limit (groq 429) | Bounded retry-with-backoff, then `orchestration_llm_call_failed` | ✅ OK |
| Provider timeout (zen) | Classified provider-level, excluded, fallback to groq | ✅ OK (but see HIGH) |
| Cross-provider fallback | Works (confirmed end-to-end) | ✅ OK |

## 7. Problems discovered (severity)

1. **HIGH — MultiModelRouter latency under configured primaries.** All zen models are
   effectively non-serving on the current free tier (timeouts/rate-limits); fallback pays
   ~15s/call. Any user who enables `multimodel_enabled` gets 20–90s queries despite correct
   answers. Evidence: live probes + e2e report (`report4.json`: A=23s, B/C/D/E≈90s).
2. **MEDIUM — 06.5.7 live contract suite under-allocates `max_tokens` (16/32).** Produces
   false failures for groq/gemini structured output and completion. Test-harness calibration,
   not production bug. Evidence: identical calls at `max_tokens=256/512` pass.
3. **LOW — hard-coded zenith assumptions.** `D-014` primary = zen, but no health-based fast
   fail. Consider a provider-health signal or config demotion of dead models.
4. **INFO — telemetry indicates calls route to groq in multi-model mode** (all recorded
   `provider=groq`). Validates fallback; confirms zen never served successfully.

## 8. M. Recommended fixes (only where evidence supports)

1. **Demote/remove non-serving zen free-tier models from `configs/model_policy.yaml`
   primaries** (or shrink their attempt ceiling to ~3–5s) so the MultiModelRouter does not
   burn ~15s per call on a provider that never succeeds within quota. This is the single
   highest-leverage latency fix and is config-only (no code change).
2. **Bump the 06.5.7 live-suite `max_tokens`** to a realistic value (e.g. 256) and assert on
   `finish_reason != length` so the contract suite reflects true provider health.
3. **(Optional) default `multimodel_enabled`** already `False`; document that enabling it
   today requires healthy primaries (or apply fix #1).

## 9. Classification

- **VERIFIED WORKING:** test suite (436/20), ingestion→re-cited-answer, retrieval relevance,
  citation provenance, fast-path efficiency (1 call), adaptive tier routing, cross-provider
  fallback, graceful provider-failure degradation, weak-evidence honesty, runtime shutdown.
- **NEEDS FIX (evidence-backed):** zen-primary latency in opt-in MultiModelRouter.
- **NEEDS MORE DATA:** token cost (no telemetry surfaced in this run); verification-vs-quality
  tradeoff with multi-agent ON.
- **NOT TESTABLE:** cerebras (no credential).

## 10. FINAL VERDICT (STEP 9)

Three distinct statuses, honestly separated:

- **TEST SUITE HEALTH — STRONG.** 436 passed / 20 skipped; runtime shutdown clean under
  `-W error::RuntimeWarning`; all 06.5 features covered by regression + fast-path +
  cross-phase + opt-in live tests.

- **REAL-WORLD RAG QUALITY — HIGH (default runtime).** Out-of-the-box ARGUS is a working,
  efficient practical Agentic RAG:
  - simple queries answered correctly with **1 LLM call in ~1.4s**;
  - complex comparative questions produce correct, well-cited analytic answers in ~7s;
  - weak-evidence questions are answered honestly with **no hallucination**;
  - citations trace correctly to source/document/chunk;
  - provider/query failures degrade gracefully without crashing or fabricating.

- **PRODUCTION READINESS — CONDITIONAL.**
  - **In its default single-provider (groq) configuration:** ready and performs well.
  - **The adaptive MultiModelRouter (Phase 07 / 06.5.2) is off by default and, when enabled
    with the current zen-first policy, is not latency-suitable for production.** It produces
    equal-quality answers only after large fallback dead-waits. Enabling it for production
    requires the config fix in §8.1.

### Verdict
In its **default runtime**, ARGUS can honestly be described as **"a working advanced
practical Agentic RAG system"**: fast, grounded, correctly cited, honest on weak evidence,
and resilient to provider failure. The adaptive multi-model fabric exists and its *fallback
mechanism works*, but it is currently **not the served production path**, and its configured
primary is not healthy enough to be latency-competitive. This is a configuration-suitability
issue, not an architecture failure.