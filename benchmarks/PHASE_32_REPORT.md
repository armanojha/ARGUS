# Phase 32 Report: Claim Generation Optimization & Inference Cost Reduction

**Date:** 2026-09-07
**Status:** INCONCLUSIVE — Rate limits corrupted benchmark results

---

## Executive Summary

Phase 32 tested 6 strategies for optimizing Pass 1 (claim generation) latency and cost. The benchmark was **invalidated** by simultaneous rate limits on all 4 LLM providers (Groq, Gemini, Z.ai, Zen), forcing fallback to slow/malformed providers for most queries.

**Production Decision: OPTION A — Re-run benchmark when rate limits reset**

No code changes to production. The existing Phase 30 Strategy C (restricted Pass 2) remains the production configuration.

---

## Strategies Tested

| Strategy | Description | Claim Support | Citation Precision | Avg Latency | Pass1 Latency |
|----------|-------------|---------------|-------------------|-------------|---------------|
| A (baseline) | 8 chunks, full prompt | 15.9% | 59.5% | 4589ms | 3928ms |
| B (4 chunks) | Reduced evidence | 8.7% | 3.0% | 5868ms | 4927ms |
| C (6 chunks) | Reduced evidence | 2.4% | 0.0% | 4720ms | 4246ms |
| D (minimal) | Stripped prompt | 0.0% | 0.0% | 2303ms | 2303ms |
| E (cap 8) | Max 8 claims | 1.6% | 4.8% | 9847ms | 9008ms |
| F (combined) | 6+minimal+cap | 0.0% | 0.0% | 4929ms | 4538ms |

**⚠️ All metrics are unreliable due to rate limit corruption.**

---

## Key Findings (with caveats)

1. **Baseline performed best even under duress** — Suggests 8 chunks + full prompt is robust
2. **Minimal prompt completely failed** — All 21 queries hit pass1_error (provider exhaustion, not prompt quality)
3. **Reduced evidence degraded quality** — Even if rate-limited, 4/6 chunks consistently produced worse answers
4. **Strategy F had best absent-info detection** — 3/4 vs baseline 2/4, suggesting the claim cap helps with edge cases
5. **Latency measurements are meaningless** — Provider fallback added 5-30s per query

---

## Production Decision: OPTION A — Re-run Required

**Rationale:**
- Results are unreliable due to rate limits
- No strategy conclusively beat baseline
- The safety invariant (absent-info detection) was partially met by Strategy F
- Need clean data before making production changes

**Next Steps:**
1. Wait for rate limit reset (Groq: daily, Gemini: daily)
2. Re-run with 10-query subset for faster results
3. If baseline still wins, conclude Phase 32 with "no optimization needed"
4. If a strategy wins, implement behind feature flag

---

## Files Modified
- `benchmarks/phase32_ablation.py` — Fixed tuple-length bug (14→13 elements), fixed double evidence slicing
- `benchmarks/results/phase32_ablation.json` — Raw results (corrupted by rate limits)
- `benchmarks/PHASE_32_DIAGNOSTIC.md` — 12 diagnostic answers
