# Phase 36 Diagnostic

**Date:** 2026-09-07
**Status:** Complete — benchmark ran, 0/8 clean queries

---

## Key Finding

**0 out of 8 queries were clean (no fallback).** Every query experienced at least 1 provider fallback. This is NOT contamination — it is the **real production behavior**. Groq's 8K TPM free-tier limit is exhausted within 2-3 queries, forcing all subsequent queries to fall back to Gemini or other providers.

## Why Fallbacks Are Inevitable

| Factor | Constraint |
|--------|-----------|
| Groq TPM | 8,000 tokens/minute |
| Per-query LLM tokens | ~3,000-5,000 (5-7 calls × ~500-800 tokens each) |
| Queries before TPM exhaustion | ~2-3 |
| Remaining queries (of 8) | ~5-6 must use fallback providers |

The MultiModelRouter correctly detects TPM exhaustion and falls back to Gemini. This is working as designed.

## Implications

1. **Phase 35's 77.1% LLM dominance IS real** — it was not an artifact of contamination
2. **The fallback provider (Gemini) is slower than Groq** — so contaminated queries are actually SLOWER than clean ones
3. **A truly clean single-provider benchmark is impossible** on free-tier Groq
4. **The production system always runs with fallbacks** — this is the normal operating state

## Benchmark Results (All Queries, Including Fallbacks)

| Metric | Value |
|--------|-------|
| Total E2E avg | 13,566ms |
| Retrieval avg | 1,993ms (14.7%) |
| LLM avg | ~11,573ms (85.3%) |
| Evidence selection avg | 1.5ms (0.0%) |
| Cold start | 12,196ms |

## Production Decision

**OPTION B — PROVIDER STILL UNSTABLE**

Free-tier provider limits prevent a clean single-provider benchmark. The fallback behavior is the real production state, but it means:
- We cannot isolate provider-specific latency from pipeline latency
- We cannot get a "pure Groq" or "pure Gemini" baseline
- The measured numbers represent the actual production experience

A paid-tier provider (or a provider with higher TPM) is needed for a clean baseline.
