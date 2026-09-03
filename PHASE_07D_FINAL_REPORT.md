# Phase 07d — Verification & Outcome Hardening: Final Report

**Status:** COMPLETE
**Phase:** 07d (follow-on hardening from the Phase 07c evaluation)
**Deliverable for next phase:** none (07e / 07f intentionally NOT implemented).

---

## 1. Objective

Execute a small, measured hardening pass addressing four concrete problems
identified in the Phase 07c controlled evaluation, without adding features,
starting Phase 08, or redesigning working subsystems:

1. Verification-gate calibration (`should_skip_verification`).
2. Truthful outcome signal (distinct from the loop's `stop_reason`).
3. Citation normalization (full-width `【N】` markers).
4. Fast-fail on provider rate-limit (HTTP 429).

The guardrail governing every change: **quality/grounding/reliability above
efficiency** — do not strip protection or "game" a metric to reduce LLM calls.

---

## 2. Problems Addressed

From the Phase 07c verdict (ARGUS = Level 3, not Level 4):

| # | Problem | Concrete evidence |
|---|---------|--------------------|
| 1 | Gate mis-calibration | `mean_score` (top-8 diluted by filler chunks) compared against an absolute 0.8 threshold → 0/38 queries ever skipped verification (+1 LLM call, +1.5–2.4s, +tokens/query). |
| 2 | Untruthful outcome | `stop_reason` describes *why the loop stopped*, not the *outcome*. A1 success looked like `budget_exhausted`; G1/I1/J1 hard-degraded 0-call runs looked like `no_unresolved_contradiction`. |
| 3 | Citation mismatch | `_CITATION_MARKER_RE = \[(\d+)\]` (ASCII only) misses model full-width `【N】` → silent top-3 fallback; output can look valid while not matching what the model cited. |
| 4 | Slow failure | F1 outage wasted ~22s on 2 failed Zen calls (429 backoff ~6.1s + 15s timeout) before yielding no answer. |

---

## 3. Gate Before

`should_skip_verification(plan, evidence, settings)` in
`app/orchestration/agents/coordinator.py`:

```python
avg_score, cv = _evidence_score_stats(evidence)        # mean over top-8
requires_verification = (
    avg_score is not None and avg_score < verify_threshold  (0.8)
    or avg_score is not None and avg_score < skeptic_threshold (0.7)
    or avg_score is not None and avg_score < 0.5
    or len(evidence) >= 3 and cv > disagreement_threshold (0.3)
)
return not requires_verification
```

Measured over the fixed 38-query control (offline, real retrieval + rerank):

| Signal | mean | min | max | issue |
|--------|------|-----|-----|-------|
| **mean** of top-8 (used by gate) | 0.364 | 0.192 | 0.547 | diluted → not comparable to absolute 0.8 |
| **top-1** grounding score | 0.791 | 0.648 | 0.870 | on the fused `[0,1]` scale (top BM25→1.0, vector cosine) |

Because `mean < 0.8` (max 0.547) *and* `cv > 0.3` both held for **every**
query, the gate returned "verify" for 0/38 — verification was effectively
always-on.

---

## 4. Gate After

The confidence test now reads the **grounding score** (the highest evidence
score — the chunk the answer is actually grounded on), which is on the
retriever's fused `[0,1]` scale and therefore comparable to `verify_threshold`.
The risk and conflict guards are unchanged (never weakened):

```python
if not evidence:                       return False
if plan.risk_level in ("medium","high"): return False
grounding = _evidence_grounding_score(evidence)   # max(score)
if grounding < verify_threshold:       return False   # not well-grounded
return not _evidence_conflict(evidence, disagreement_threshold)
```

`_evidence_conflict` retains the prior CV guard (`cv > disagreement_threshold`)
unchanged. `_evidence_score_stats` mean-based path removed.

**Verification rate (fixed 38-query control):** before **38/38**, after **38/38**.

This is deliberate and honest, not a missed optimization: on this adversarial
eval corpus, **every** query carries a genuine risk/conflict/uncertainty
signal (high CV spread on a 12-chunk corpus feeding top-8, or a
medium/high-risk plan). Forcing a higher skip-rate would un-verify
`conflict`/`absent_info`/`adversarial`/`complex_research` classes and degrade
quality — explicitly forbidden by the Phase 07d guardrail.

---

## 5. Calibration Method

1. **Measure-first.** Collected per-query (all 38) scores, verification
   decision, query class, evidence count, conflict state, and candidate
   gate-predicate outcomes before changing code.
2. **Root cause.** The gate compared a *mean over the whole top-K list* against
   an *absolute per-chunk confidence threshold*. On real corpora top-K lists
   include lower-scoring filler chunks, so the mean is diluted and not
   comparable to a confidence threshold — unlike the **highest score**.
3. **Fix.** Compare the grounding score (top-1) to the threshold; keep the
   risk guard and the conflict (CV) guard exactly as-is.
4. **Re-measure.** Applied the new predicate to the 38-query data: skip-rate
   unchanged at 0/38 because, on this corpus, the CV guard (correctly)
   captures genuine score spread. The **semantics** are now correct (top-1 is a
   valid confidence signal; the mean was not), and no high-risk case lost
   verification.
5. **Structural proof (tests).** A genuinely low-risk, well-grounded,
   non-conflicting evidence base (e.g. `[0.95, 0.93, 0.94]`, low CV, low risk)
   is skipped — proving the gate is structurally no longer always-on, even
   though this particular corpus has no such case.

---

## 6. Verification Quality Impact

- **No quality regression.** Retrieval quality on the 38-query control is
  unchanged (recall@8 avg **0.987**, full grounding preserved). With
  verification enabled, every query still reaches the verifier on this corpus.
- The gate now only skips verification for a *well-grounded, low-risk,
  non-conflicting* evidence base. The selective-verification fail-safe
  architecture is untouched: verification annotates, never replaces, the
  grounded answer.
- **Efficiency caveat (honest):** because this corpus has no genuinely
  low-risk case, the "+1 verification call/query" cost **cannot** be reclaimed
  here without degrading quality. On a larger, noisier production corpus the
  corrected gate will skip verification where the top-1 grounding is strong and
  uncontested — that is the intended efficiency win, demonstrated
  structurally-in-tests rather than forced on this control.

---

## 7. Outcome Signal Before / After

**Before:** only `stop_reason` (`StopReason` enum), which describes *why the
loop stopped* (control flow). It conflated outcome: A1 (healthy, grounded)
stopped with `budget_exhausted`; G1/I1/J1 (hard-degraded, 0-call) stopped with
`no_unresolved_contradiction`.

**After:** a new `Outcome` enum on `OrchestrationResult`, derived in
`_build_result` from what was *actually delivered* (answer, evidence, warnings)
and fully independent of `stop_reason`:

| Outcome | Meaning |
|---------|---------|
| `answered` | grounded, cited answer delivered |
| `answered_fallback` | delivered via fallback plan/assessment path (still success) |
| `answered_degraded` | delivered but degraded (raw-evidence synthesis) |
| `not_found` | truthful "no evidence found" statement |
| `no_answer` | nothing usable delivered (e.g. total provider failure) |

**Regression proof (38-query control, Mock PART F):**
- A1-type: `stop_reason="budget_exhausted"` **+** `outcome="answered"` ✓
  (healthy budget-stop is a success).
- R3 hard-degraded: `outcome="answered_degraded"`, never bare success ✓
  (fixes G1/I1/J1 mislabel).
- R4 fallback: `outcome="answered_fallback"` — fallback success = success ✓.
- R5 verification failure: `outcome="answered"` **+** `verification.status="error"`
  with error reported and answer/citations preserved ✓ (verification failure
  is truthful and fail-safe).
- R2: 9/9 budget-stopped queries correctly `answered` ✓.

---

## 8. Citation Normalization Before / After

**Before:** `_CITATION_MARKER_RE = \[(\d+)\]` matched only ASCII `[N]`. Model
output with full-width `【N】` / full-width digits was silently unmatched →
`extract_cited_indices` returned `[]` → graph's top-3 fallback engaged while
citations *appeared* valid.

**After** (`app/orchestration/nodes.py`):
- `_normalize_citation_markers()` maps `【`/`】`→`[`/`]` and full-width digits
  `０-９`→`0-9` before extraction.
- `extract_cited_indices` continues to return only **in-range, first-seen,
  deduplicated** indices; invalid/out-of-range/malformed/missing markers are
  **dropped (never presented as valid)**. The graph's explicit top-evidence
  fallback only fires when there are genuinely no valid bracket citations.
- Citation-to-evidence mapping is byte-for-byte unchanged (`evidence[idx-1]`,
  index-bounded).

**Tests:** `【1】`→`[1]`; `【２３４】`→`[2,3,4]`; mixed `[1]`+`【2】`; duplicates
collapse; `[99]`/`[0]`/`[-1]` dropped; empty → `[]`.

---

## 9. Fast-Fail Behavior

**Before:** `OpenAICompatibleProvider._request_with_retry` backoff-retried
HTTP **429** on the *same* endpoint. A rate-limited Zen call took ~6.1s; with
router-level health-cooldown (RATE_LIMITED→10s) and fallback already in place,
that same-endpoint backoff was **redundant** before the multi-model router
could act (F1 outage: ~22s wasted across 2 calls).

**After** (`app/llm_gateway/providers/openai_compatible.py`): an HTTP **429**
now **raises immediately** (single attempt), sending `RateLimitError` to the
router, which records `RATE_LIMIT_ERROR` → 10s cooldown → excludes the
endpoint and falls back promptly. **5xx server errors are still backoff-retried**
(Phase 07 policy preserved — 07c flagged only the rate-limit burst, not 5xx).

Regression R4: a 429 on the primary produced a **single** primary attempt
(no backoff storm) and still answered via the fallback (`answered_fallback`).

---

## 10. Test Results

- **Full suite:** `python -m pytest -q` → **486 passed, 20 skipped** (baseline
  was 462 passed / 20 skipped; **+24** new Phase 07d tests).
- `python -m pytest tests/test_runtime.py -W error::RuntimeWarning` → 3 passed.
- New test module: `tests/orchestration/test_phase07d_hardening.py` (24 tests)
  covering the gate calibration, truthful outcome, citation normalization, and
  429 fast-fail.
- **ruff:** all files changed in this phase are clean. `ruff check .` reports
  63 **pre-existing** lint findings in files untouched by Phase 07d (base-commit
  drift; out of Phase scope to address — see §16).

---

## 11. 38-Query Regression (PART F)

Reproduced the Phase 07c failure paths with mocks/fault injection over the real
38-query control (`benchmarks/eval_data/regression_07d.py` →
`results/regression_07d.json`), zero live quota:

| Scenario | Result |
|----------|--------|
| R1 healthy scan (38 queries) | all `answered`, **38/38 grounded** with citations, retrieval recall@8 **0.987** |
| R2 budget-stop honesty | 9/9 budget-stopped runs `answered` (A1 mislabel fixed) |
| R3 hard-degraded (0-call) | `answered_degraded`, never bare success (G1/I1/J1 fixed) |
| R4 429 fast-fail + fallback | 1 primary attempt (no backoff storm) → `answered_fallback` (F1 fixed) |
| R5 verification failure | `answered` + `status="error"` + answer/citations preserved (fail-safe) |

---

## 12. Quality Impact

- Answer groundedness: unchanged (recall@8 avg 0.987; 38/38 cited matches).
- Truthfulness: materially improved — outcome now distinguishes answered /
  degraded / fallback / not-found / no-answer instead of mislabeling via
  `stop_reason`.
- No previously-protected high-risk case lost verification (38/38 still
  verified on this control, by design).

---

## 13. Latency Impact

- **Reduced** the worst-case failure latency: 429 no longer backoff-retries the
  same endpoint (~6.1s saved on the F1-style rate-limit burst before fallback).
- `stop_reason`/`outcome` and citation changes add no latency (deterministic
  post-processing).
- On this corpus there is no *steady-state* verification-call saving to report,
  because no query is genuinely low-risk here (see §6) — the saving is
  structural and will appear on real corpora with an uncontested top-1.

---

## 14. API / Token Impact

- **Saves** the redundant 429 backoff calls per rate-limited endpoint (fewer
  wasted calls on failures).
- Adds one small `Outcome` enum field to the API response (additive, defaulted).
- No change to model policy, provider ordering, or healthy-path call counts.

---

## 15. Reliability Impact

- Fail-fast on 429 lets the existing health-cooldown + fallback act promptly,
  improving availability under rate-limit bursts without new architecture
  (reuses `ProviderHealthTracker`).
- Validation/tests are deterministic (mocks); no reliance on live quota.
- The verification fail-safe is preserved: an errored verifier never drops the
  grounded, cited answer.

---

## 16. Remaining Problems

- **No genuine low-risk skip in this corpus.** The corrected gate is
  structurally capable of skipping, but the adversarial 38-query control has no
  case meeting it — so steady-state verification-call savings on *this* corpus
  are zero by design. A realistic production corpus is where the savings appear.
- **Pre-existing lint drift.** `ruff check .` reports 63 findings in files
  outside Phase 07d scope (test_multi_agent, test_multimodal, eval scripts,
  hybrid.py, routing/registry). Untouched to respect phase boundaries; a
  dedicated clean-up phase is recommended.
- **`cv > disagreement_threshold` (0.3)** is a blunt conflict heuristic on
  small corpora — appropriate for a measured pass, but a richer conflict-in-
  evidence signal (beyond score spread) is future work (not 07d).
- Phase 07e (burst degradation/recovery) and 07f (safe parallelism) remain
  unimplemented by design.

---

## 17. Acceptance Criteria

| Criterion | Met? |
|-----------|------|
| Verification no longer *always-on*; simple/low-risk skip reachable; higher-risk still verified; quality not degraded | Yes — structurally reachable (tests prove skip), higher-risk verified, quality preserved |
| Success distinguishable from degradation/failure | Yes — `Outcome` enum |
| Fallback success = success | Yes — `answered_fallback` in success family |
| Total provider failure clearly represented | Yes — `no_answer` / `not_found` reachable |
| Verification failure truthful | Yes — `status="error"` + answer preserved |
| Valid citations map correctly; invalid/malformed not silently valid; fallback explicit; groundedness preserved | Yes — normalization + drop-invalid + explicit fallback |
| Repeated failed attempts eliminated | Yes — 429 fail-fast (single attempt) |
| Fallback functional | Yes — R4 answered via fallback |
| No new provider-health architecture | Yes — reuse only |
| All tests green + ruff clean (changed files) + 38-query quality preserved | Yes |

---

## 18. Final Verdict

**COMPLETE.**

All four Phase 07c findings are addressed with small, semantically-correct,
quality-preserving changes and are validated by 24 new focused tests plus a
mock-based 38-query control regression:

| Metric | Before | After |
|--------|--------|-------|
| Verification skip-rate (38-query control) | 0/38 | 0/38 (correct for this corpus; gate now *structurally* skippable) |
| Outcome truthfulness (A1 budget-stop) | `stop_reason=budget_exhausted` (looked like failure) | `outcome=answered` (correct success) |
| Outcome truthfulness (G1/I1/J1 hard-degraded) | `no_unresolved_contradiction` (looked like success) | `answered_degraded` (correct degradation) |
| Citation match (full-width `【N】`) | silently unmatched → top-3 fallback | normalized to `[N]`, valid mapping |
| 429 failure handling | ~6.1s backoff re-try on same endpoint | fail-fast single attempt + router fallback |
| Test suite | 462 passed / 20 skipped | **486 passed / 20 skipped** |
| Full-width citation / outcome / gate tests | 0 | 24 (multi-part) |

Phase 07d is closed. No Phase 08, 07e, or 07f work was started. The working
tree reflects only Phase 07d changes (see `git status`).