# Phase 38 — Quality Baseline & Gold-Standard Evaluation

**Status:** COMPLETE  
**Date:** 2026-09-07  
**Decision:** BASELINE ESTABLISHED — no production changes  

---

## Summary

Established the first trustworthy quality baseline for ARGUS answer quality. Measured 38 gold-standard queries across 10 categories using deterministic metrics.

## Results

### Overall Quality
- **Claim Support Rate:** 32.8% (only 1/3 of claims fully supported)
- **Gold Fact Coverage:** 51.1% (half of expected facts found)
- **Citation Precision:** 87.1% (when cited, mostly correct)
- **Hallucination Rate:** 3.7% (low but real)
- **Query Relevance:** 80.9% (answers generally address the question)

### Critical Gaps
- **Conflict handling:** 0/3 (0%) — never acknowledges contradictions
- **Absent info abstention:** 2/4 (50%) — fabricates half the time
- **Technical explanations:** 14% CS, 0% GFC, 14% hallucination — weakest category
- **Degraded synthesis:** 11/38 runs produce "Synthesis is temporarily unavailable" text

### Category Performance
| Category | CS | GFC | HR | Quality |
|----------|-----|-----|-----|---------|
| simple_lookup | 0.58 | 0.83 | 0.00 | STRONG |
| normal_qa | 0.48 | 1.00 | 0.00 | STRONG |
| multi_doc_synthesis | 0.47 | 1.00 | 0.00 | STRONG |
| complex_research | 0.36 | 0.20 | 0.04 | MODERATE |
| adversarial | 0.31 | 0.17 | 0.04 | MODERATE |
| multi_hop | 0.22 | 0.33 | 0.07 | WEAK |
| numerical | 0.23 | 0.67 | 0.06 | WEAK |
| conflict | 0.18 | 0.83 | 0.04 | WEAK |
| technical_explanation | 0.14 | 0.00 | 0.14 | CRITICAL |
| absent_info | 0.12 | 0.08 | 0.05 | CRITICAL |

### Provider Contamination
- 95% of runs had provider fallbacks
- Only 2/38 clean runs
- Quality metrics are **directionally correct but not precisely trustworthy**

### Latency & Cost
- Avg E2E: 8,744ms (LLM 82%, Retrieval 16%)
- Avg 4.1 LLM calls/query, 2,340 tokens/query
- Total: 155 LLM calls, 88,919 tokens across 38 queries

## Files Generated

- `benchmarks/phase38_quality_baseline.py` — benchmark script
- `benchmarks/results/phase38_quality_baseline.json` — raw results
- `benchmarks/PHASE_38_QUALITY_BASELINE.md` — detailed quality metrics
- `benchmarks/PHASE_38_GOLD_STANDARD.md` — dataset audit
- `benchmarks/PHASE_38_DIAGNOSTIC.md` — root cause analysis
- `benchmarks/PHASE_38_REPORT.md` — this file

## Next Steps

1. **Fix conflict handling** — 0% → target 100% (highest-impact)
2. **Fix absent-info abstention** — 50% → target 80%
3. **Fix degraded synthesis path** — eliminate "Synthesis is temporarily unavailable"
4. **Establish clean provider baseline** — cannot trust quality comparison until provider contamination < 20%
5. **Determine backend freeze readiness** — after quality gaps addressed
