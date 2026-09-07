# Phase 32.1 Report: Clean Re-Run, Validation & Production Decision

**Date:** 2026-09-07
**Status:** BENCHMARK DEFERRED — OPTION A selected by default

---

## Executive Summary

Phase 32.1 attempted a clean re-run of the Phase 32 benchmark after fixing three infrastructure bugs. The benchmark was **deferred** because Groq's free tier (200K TPD) cannot support 6 strategies × 11 queries × 2 LLM calls.

**Production Decision: OPTION A — Keep Phase 30 unchanged.**

No code changes to production. The existing Phase 30 Strategy C (restricted Pass 2) remains the production configuration.

---

## Bugs Fixed Before This Run

1. **Tuple-length bug:** All 6 strategies returned 14 elements; unpacking expected 13
2. **Double evidence slicing:** Strategies B/C sliced evidence, then baseline re-sliced
3. **Strategy E variable contamination:** Used custom prompt instead of baseline prompt (tested 2 variables simultaneously)

All three fixes are correct and verified.

---

## Benchmark Results

| Strategy | Quality Results | Provider Failures | Status |
|----------|----------------|-------------------|--------|
| A (baseline) | 0/11 | 11/11 | ALL FAILED |
| B (4 chunks) | 1/11 | 10/11 | INSUFFICIENT |
| C (6 chunks) | 1/11 | 10/11 | INSUFFICIENT |
| D (minimal) | 0/11 | 11/11 | ALL FAILED |
| E (cap 8) | 0/11 | 11/11 | ALL FAILED |
| F (combined) | 1/11 | 10/11 | INSUFFICIENT |

**No strategy had enough quality results for statistical comparison.**

---

## Production Decision: OPTION A

**Rationale:**
1. No optimization has been validated by clean benchmark data
2. The benchmark cannot be completed today (Groq quota exhausted)
3. Architectural analysis suggests diminishing returns for Pass-1 optimization
4. Phase 30 Strategy C (restricted Pass 2) is already production-validated
5. Reducing evidence or prompt complexity risks quality degradation without proven latency gains

**What would change this decision:**
- A clean benchmark run showing ≥20% latency reduction with no quality regression
- A paid Groq tier or multi-day benchmark strategy
- New evidence that Pass-1 latency is a user-facing bottleneck

---

## Future Work

1. **Re-run with 3 strategies** (A, best optimization, F) × 8 queries = 48 calls (72K tokens, fits in 200K TPD)
2. **Consider paid tier** for higher TPD if Pass-1 optimization is critical
3. **Accept OPTION A** and move to other optimization areas (retrieval, caching)

---

## Files Modified
- `benchmarks/phase32_ablation.py` — Fixed 3 bugs, interleaved execution, reduced to 11 queries
- `benchmarks/results/phase32_ablation.json` — Raw results (corrupted by rate limits)
- `benchmarks/PHASE_32_1_DIAGNOSTIC.md` — 10 diagnostic answers
- `benchmarks/PHASE_32_1_REPORT.md` — This file
