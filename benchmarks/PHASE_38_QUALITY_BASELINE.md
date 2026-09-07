# Phase 38 — Quality Baseline Report

**Status:** COMPLETE  
**Date:** 2026-09-07  
**Decision required:** No — baseline measurement only  

---

## Executive Summary

Measured production answer quality on 38 gold-standard queries across 10 categories. Found that overall quality is **low** (32.8% claim support, 51.1% gold fact coverage), but this is heavily contaminated by provider instability (95% of runs had fallbacks). Hallucination rate is low (3.7%), citation precision is high (87.1%), and the system correctly abstains on absent information (50%). Conflict handling is broken (0/3 correct).

## Overall Quality Metrics

| Metric | Value | N | Interpretation |
|--------|-------|---|----------------|
| Claim Support Rate | 0.328 | 38 | Only 1/3 of claims fully supported |
| Unsupported Claim Rate | 0.062 | 38 | 6% of claims have no backing |
| Partially Supported Rate | 0.298 | 38 | 30% partially supported |
| Contradicted Claim Rate | 0.000 | 38 | No contradictions found |
| Citation Presence Rate | 0.664 | 38 | 66% of claims cite evidence |
| Citation Precision | 0.871 | 38 | When cited, 87% correct |
| Gold Fact Coverage | 0.511 | 38 | Half of gold facts found |
| Numerical Consistency | 0.593 | 27 | 59% of numbers match |
| Query Relevance | 0.809 | 38 | 81% address the question |
| Hallucination Rate | 0.037 | 38 | 3.7% hallucinated claims |
| Abstention Correctness | 2/4 (50%) | 4 | Correctly abstains half the time |
| Conflict Handling | 0/3 (0%) | 3 | Never acknowledges conflicts |

## Per-Category Breakdown

| Category | N | CS | GFC | HR | Abst | Conf | E2E ms |
|----------|---|-----|-----|-----|------|------|--------|
| simple_lookup | 6 | 0.58 | 0.83 | 0.00 | N/A | N/A | 3,695 |
| normal_qa | 4 | 0.48 | 1.00 | 0.00 | N/A | N/A | 6,435 |
| multi_hop | 3 | 0.22 | 0.33 | 0.07 | N/A | N/A | 12,989 |
| numerical | 3 | 0.23 | 0.67 | 0.06 | N/A | N/A | 5,097 |
| conflict | 3 | 0.18 | 0.83 | 0.04 | N/A | 0/3 | 5,932 |
| absent_info | 4 | 0.12 | 0.08 | 0.05 | 2/4 | N/A | 10,889 |
| technical_explanation | 3 | 0.14 | 0.00 | 0.14 | N/A | N/A | 11,295 |
| multi_doc_synthesis | 3 | 0.47 | 1.00 | 0.00 | N/A | N/A | 6,416 |
| complex_research | 3 | 0.36 | 0.20 | 0.04 | N/A | N/A | 22,351 |
| adversarial | 6 | 0.31 | 0.17 | 0.04 | N/A | N/A | 8,095 |

**Key findings:**
- **Simple lookups and normal QA** are strong (58% CS, 83-100% GFC)
- **Multi-doc synthesis** is strong (47% CS, 100% GFC, 0% hallucination)
- **Technical explanations** are weakest (14% CS, 0% GFC, 14% hallucination)
- **Conflict handling** is completely broken (0/3 acknowledge conflicts)
- **Absent info** correctly abstains 50%, but fabricates the other 50%
- **Complex research** is slow (22s) and low quality (36% CS)

## Provider Contamination

| Metric | Value |
|--------|-------|
| Clean runs | 2/38 (5%) |
| Contaminated | 36/38 (95%) |

95% of runs had provider fallbacks. Only 2 queries ran cleanly (G4, H1 — both simple lookups that hit Gemini directly). This means quality metrics are heavily influenced by which provider answered each call, not by ARGUS's actual capability.

**Impact on trustworthiness:** Quality metrics are **directionally correct** but **not precisely trustworthy**. Provider fallback introduces variance that masks true quality differences between query types.

## LLM Call Analysis

| Call Type | Count | Avg ms | Tokens |
|-----------|-------|--------|--------|
| query_analysis | 27 | 1,022 | 12,795 |
| research_planning | 32 | 1,973 | 8,789 |
| evidence_extraction | 42 | 1,383 | 25,327 |
| synthesis | 43 | 2,541 | 31,300 |
| verification | 11 | 1,446 | 10,708 |
| **TOTAL** | **155** | **1,768** | **88,919** |

- Avg 4.1 calls/query, 2,340 tokens/query
- Synthesis is the bottleneck: 73% of LLM time
- Verification was skipped on 27/38 queries (low-risk fast path)

## Latency Analysis

| Metric | Value |
|--------|-------|
| Avg E2E | 8,744ms |
| Avg LLM | 7,211ms (82%) |
| Avg Retrieval | 1,429ms (16%) |
| Cold start | 7,572ms |

## Error Taxonomy

| Error Type | Count | % |
|------------|-------|---|
| PROVIDER_FAILURE | 36 | 95% |
| LOW_CLAIM_SUPPORT | 26 | 68% |
| LOW_GOLD_FACT_COVERAGE | 18 | 47% |
| NUMERICAL_ERROR | 9 | 24% |
| HIGH_UNSUPPORTED_CLAIMS | 4 | 11% |
| CONFLICT_MISSED | 3 | 8% |
| ABSENT_INFO_FABRICATION | 2 | 5% |
| CITATION_ERROR | 1 | 3% |

## Files Generated

- `benchmarks/phase38_quality_baseline.py` — benchmark script
- `benchmarks/results/phase38_quality_baseline.json` — raw results (38 queries)
