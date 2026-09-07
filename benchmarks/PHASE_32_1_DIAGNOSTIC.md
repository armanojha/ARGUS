# Phase 32.1 Diagnostic

**Date:** 2026-09-07
**Status:** BENCHMARK DEFERRED — Stable provider capacity unavailable

---

## D1: Was the benchmark infrastructure correct?
**YES (after fixes).** Three bugs were fixed before this run:
1. Tuple-length bug: strategies returned 14 elements, unpacking expected 13
2. Double evidence slicing in strategies B/C
3. Strategy E used custom prompt instead of baseline prompt (tested 2 variables)

## D2: Was provider capacity stable?
**NO.** Groq 200K TPD was exhausted by query 8-9 of 11, despite interleaved execution and reduced query set. The free tier cannot support 6 strategies × 11 queries × 2 LLM calls.

## D3: Were any queries rate-limited?
**YES.** 10-11 out of 11 queries per strategy were rate-limited (provider failures).

## D4: Were any retries/fallbacks used?
**YES.** Router attempted Groq → Gemini → Z.ai → Zen fallback chain. Most providers were also rate-limited. Some queries succeeded via Z.ai or Zen but with different models, contaminating the comparison.

## D5: Did the baseline still outperform?
**UNDETERMINED.** Baseline (A) had 0/11 quality results (all provider failures). No strategy had enough quality results to compare.

## D6: Which strategy produced the best quality?
**UNDETERMINABLE.** Maximum 1 quality result per strategy. Insufficient data.

## D7: Which strategy produced the best latency?
**UNDETERMINABLE.** Latencies reflect provider fallback delays (5-30s), not strategy performance.

## D8: Which strategy produced the best quality/latency tradeoff?
**UNDETERMINABLE.**

## D9: Did any optimization reduce safety?
**UNDETERMINABLE.** Too few quality results to assess.

## D10: Is there enough evidence to make a production decision?
**NO.** Zero strategies had enough quality results for statistical comparison.

---

## Root Cause

The Groq free tier (200K TPD) cannot support the Phase 32.1 benchmark:
- 6 strategies × 11 queries × 2 LLM calls = 132 calls
- Average ~1500 tokens/call = ~198K tokens (at limit)
- Plus health check requests and previous day's usage
- Fallback providers (Gemini 500/day, Z.ai rate-limited) also exhausted

## Recommendation

**Benchmark deferred because stable provider capacity is unavailable.**

Options for future runs:
1. **Wait for fresh Groq quota** (resets daily) AND run only 3 strategies (A, one optimization, F) with 8 queries = 48 calls × 1500 tokens = 72K tokens (fits)
2. **Use paid tier** for higher TPD
3. **Accept OPTION A** (no changes) based on architectural analysis rather than benchmark data
