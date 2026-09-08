# Phase 37 Diagnostic

**Date:** 2026-09-07
**Status:** Complete — ablation results contaminated by provider instability

---

## Key Finding

**Ablation results are UNTRUSTWORTHY for production decisions.** Provider fallbacks (Groq TPM exhaustion, Gemini quota, z.ai timeouts) contaminate every configuration run. The measured latency differences between configurations reflect provider behavior differences, not pipeline optimization impact.

## Why Results Are Contaminated

1. **Every query in every configuration had provider fallbacks** — 0/48 runs were clean
2. **Provider behavior varies between runs** — the same query on the same configuration gets different providers depending on TPM state
3. **Outlier timeouts** — z.ai 21s timeouts appeared in some runs but not others, dramatically skewing averages
4. **TPM exhaustion cascades** — later queries in a run hit more fallbacks than earlier queries

## What IS Measurable (Despite Contamination)

### 1. Existing Fast Path Works (Phase 06.5.3)

**Measured in all configurations:**
- Simple_lookup queries: 2 LLM calls (synthesis + verification) — 67% fewer calls than normal path
- Token savings: ~57% (1823 vs 4253 avg)
- E2E savings: ~83% (3077 vs 18113 avg — but skewed by outliers)

### 2. Call Count Reductions Are Real

Even with provider noise, the call count differences between configurations are consistent:

| Config | Avg Calls | Δ from Baseline |
|--------|-----------|-----------------|
| A_full (baseline) | 6.6 | — |
| B_no_analyze | 5.4 | -1.2 |
| C_no_plan | 5.6 | -1.0 |
| D_no_analyze_plan | 5.0 | -1.6 |
| E_no_verify | 4.0 | -2.6 |
| F_no_analyze_plan_verify | 4.6 | -2.0 |

**Call counts are deterministic** — they don't depend on provider behavior. The reductions are real.

### 3. Token Reductions Are Directionally Correct

| Config | Avg Tokens | Δ% |
|--------|-----------|-----|
| A_full | 4253 | — |
| B_no_analyze | 3190 | -25% |
| C_no_plan | 3269 | -23% |
| D_no_analyze_plan | 2856 | -33% |
| E_no_verify | 2552 | -40% |
| F_no_analyze_plan_verify | 2508 | -41% |

Token counts are also deterministic (same prompts → same tokens). The reductions are real.

### 4. Latency Differences Are NOT Reliable

| Config | Avg E2E ms | Δ% |
|--------|-----------|-----|
| A_full | 18113 | — |
| B_no_analyze | 9305 | -49% |
| C_no_plan | 9495 | -48% |
| D_no_analyze_plan | 11419 | -37% |
| E_no_verify | 11252 | -38% |
| F_no_analyze_plan_verify | 9177 | -49% |

**These numbers are NOT trustworthy** because:
- The baseline (A_full) includes P37-Q07 with a 21s zai timeout
- Other configurations may not hit that timeout
- Provider routing varies between runs

### 5. Verification Skip Analysis

Verification was skipped in only 1/8 queries in the baseline (the fast-path query). The 06.5.4 gate almost never fires because:
- The evidence score threshold (0.8) is higher than typical retrieval scores
- The pattern check doesn't match most benchmark patterns

**However:** Config E (skip verify) shows that verification contributes ~2.6 calls and ~40% of tokens. This is the single largest skip opportunity.

## Ablation Results Summary

### Theoretical Savings (If Providers Were Stable)

| Optimization | Calls Saved | Token Reduction | Latency Reduction (est.) |
|-------------|-------------|-----------------|-------------------------|
| Skip analyze | 1.2 | 25% | ~1000ms |
| Skip plan | 1.0 | 23% | ~1500ms |
| Skip analyze+plan | 1.6 | 33% | ~2500ms |
| Skip verify | 2.6 | 40% | ~1300ms |
| Skip all three | 2.0 | 41% | ~3800ms |

### Per-Pattern Findings

**simple_lookup:** Already fast-path (2 calls). Can reduce to 1 call by skipping verification.

**normal_qa:** Benefits from skipping analyze (-1 call, -25% tokens). Plan provides value (subquestion decomposition).

**technical_explanation:** Benefits from skipping analyze+plan (-2 calls, -33% tokens). Single-pass retrieval sufficient.

**multi_hop:** Benefits from skipping analyze (-1 call). Plan provides value for multi-hop decomposition.

**numerical:** Benefits from skipping analyze+plan+verify (down to 1 call). Single-pass retrieval + synthesis sufficient.

**conflict:** Benefits from skipping analyze (-1 call). Plan and assess provide value for contradiction detection.

**complex_research:** Benefits from skipping verify (-2 calls, -40% tokens). Full pipeline needed for research.

**absent_info:** Benefits from skipping analyze+plan (-2 calls). Plan provides limited value when information is absent.

## Production Decision

**OPTION E — INFRASTRUCTURE STILL BLOCKED**

The ablation results demonstrate that LLM call minimization IS theoretically beneficial (up to 41% token reduction, 2 fewer calls per query). However:

1. **Provider instability makes measurements unreliable** — can't validate quality impact
2. **The existing fast path already handles the highest-value case** (simple_lookup)
3. **Verification skip is the largest opportunity** but requires quality validation
4. **Quality impact cannot be assessed** without clean provider runs

### What Would Unlock Progress

1. **Paid-tier Groq** (higher TPM) — enables clean ablation runs
2. **Quality benchmark** — gold-standard answers for each query pattern
3. **Stable providers** — same configuration produces same results

### Recommended Next Phase

**Phase 38: Quality Baseline & Gold-Standard Answers** — before implementing any fast paths, establish quality metrics so we can measure the impact of skipping calls.
