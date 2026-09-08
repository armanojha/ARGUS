# Phase 38 — Diagnostic

**Status:** COMPLETE  
**Date:** 2026-09-07  

---

## What Was Measured

Ran 38 gold-standard queries through the full production pipeline and evaluated answer quality using deterministic metrics (claim support, citation precision, gold fact coverage, hallucination rate, abstention correctness, conflict handling).

## Key Findings

### 1. Provider Instability Invalidates Precision
- 95% of runs had provider fallbacks
- Only 2/38 queries were clean
- Quality metrics are directionally correct but not precisely trustworthy
- Different providers produce different answer quality for the same query

### 2. Core Quality Gaps
- **Claim support:** 32.8% — only 1/3 of claims are fully supported
- **Gold fact coverage:** 51.1% — half of expected facts are missing
- **Conflict handling:** 0% — never acknowledges contradictions
- **Abstention:** 50% — correctly abstains half, fabricates half

### 3. Category Strengths
- Simple lookups: 58% CS, 83% GFC — reliable
- Multi-doc synthesis: 47% CS, 100% GFC — strong
- Normal QA: 48% CS, 100% GFC — reliable

### 4. Category Weaknesses
- Technical explanations: 14% CS, 0% GFC, 14% hallucination
- Conflict: 0/3 acknowledge contradictions
- Absent info: 50% fabrication rate
- Complex research: 22s E2E, 36% CS — slow and low quality

### 5. Hallucination is Low but Real
- 3.7% overall hallucination rate
- Worst in technical explanations (14%)
- All hallucinations are "synthesis unavailable" text from degraded paths

## Root Causes

1. **Provider instability** — Groq TPM exhaustion cascades through all calls; Gemini fallback answers are lower quality
2. **Conflict detection not wired into synthesis** — The pipeline detects conflicts in evidence but synthesis doesn't acknowledge them
3. **Absent-info detection is weak** — No explicit abstention signal; synthesis fabricates rather than saying "not found"
4. **Gold fact matching is strict** — Exact string match misses semantically equivalent answers

## Recommendations

1. **Do NOT optimize based on these metrics** — provider contamination makes ablation unreliable
2. **Fix conflict handling** (0% → target 100%) — highest-impact quality gap
3. **Fix absent-info abstention** (50% → target 80%) — second-highest-impact
4. **Fix degraded synthesis path** — "Synthesis is temporarily unavailable" hallucination appears in 11/38 runs
5. **Establish clean provider baseline** before any optimization decisions
