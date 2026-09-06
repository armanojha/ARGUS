# Phase 32 Diagnostic: Claim Generation Optimization

**Date:** 2026-09-07
**Status:** CORRUPTED — Rate limits invalidated results. Re-run needed.

---

## ⚠️ Rate Limit Corruption

All 6 LLM providers were exhausted during this benchmark run:
- **Groq** (primary): 200K TPD limit, ~199K used — `RATE_LIMIT_ERROR` on every request
- **Gemini**: 500 requests/day free tier — `RESOURCE_EXHAUSTED`
- **Z.ai** (glm-4.5-flash): `RATE_LIMIT_ERROR` on requests
- **Zen** (nemotron): Available but slow (5-30s), sometimes returns malformed responses

Fallback chain forced most queries to Zen, which is ~5-10x slower than Groq and less capable. This corrupts both latency and quality measurements.

---

## 12 Diagnostic Answers

### D1: Why does Pass 1 consume 69% of latency?
**Partially answered (from Phase 31).** Pass 1 sends ~2500 input tokens (8 chunks × ~800 chars + ~500 token system prompt). With Groq GPT-OSS-120B, this takes ~1.5-3s. With rate-limited fallbacks, it takes 5-30s.

### D2: Does reducing evidence to 4 chunks help?
**INCONCLUSIVE.** Strategy B (4 chunks) showed worse quality (8.7% support vs 15.9% baseline) and higher latency (5868ms vs 4589ms). But this was corrupted by rate limits — B ran during peak exhaustion.

### D3: Does reducing evidence to 6 chunks help?
**INCONCLUSIVE.** Strategy C (6 chunks) showed 2.4% support vs 15.9% baseline. Rate limit corruption.

### D4: Does a minimal prompt help?
**NO (with caveats).** Strategy D (minimal prompt) returned 0% on all metrics — all queries hit pass1_error from provider exhaustion. The minimal prompt itself may be fine, but the benchmark couldn't test it.

### D5: Does capping claims at 8 help?
**INCONCLUSIVE.** Strategy E showed 1.6% support, 33.3% gold coverage. Rate limit corruption.

### D6: Does the combined strategy (6 chunks + minimal + cap 8) help?
**PARTIALLY.** Strategy F showed 3/4 absent-info detection (best of any strategy) but 0% support. Rate limit corruption for quality metrics.

### D7: What is the actual Pass 1 token count?
From baseline (A): 21,256 input tokens / 21 queries = ~1,012 tokens/query average. This is lower than expected (~2500), suggesting some queries got smaller evidence sets or truncated prompts.

### D8: Is there a faster model for Pass 1?
Not testable in this run due to rate limits. Z.ai (glm-4.5-flash) was rate-limited. Would need fresh quotas.

### D9: Can we use structured output to reduce tokens?
Not testable — all strategies used `response_format=ClaimGenerationOutput`. The structured output constraint was constant.

### D10: Does pattern-specific Pass 1 routing help?
Not tested in this phase (was Phase 31 scope). Phase 31 found ~400ms average savings — marginal.

### D11: Can simple lookups skip Pass 1 entirely?
Not tested. Would require a confidence-based early exit before claim generation.

### D12: What is the minimum viable evidence for good claims?
**INCONCLUSIVE.** 4 chunks degraded quality. 6 chunks was also corrupted. Need clean re-run.

---

## Recommendations

1. **Re-run when rate limits reset** (Groq resets daily, Gemini resets daily)
2. **Reduce benchmark to 10 queries** for faster, cheaper runs
3. **Consider using only Groq** (when available) to avoid fallback noise
4. **The baseline (8 chunks, full prompt) appears robust** — even under rate limits, it performed best
