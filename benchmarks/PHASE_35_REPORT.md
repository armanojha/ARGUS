# Phase 35 Report: End-to-End Performance & Quality Bottleneck Audit

**Date:** 2026-09-07
**Status:** Complete — no production changes

---

## 1. Executive Summary

Current E2E latency: **12,094ms** (warm avg), **10,071ms** (P50), **27,302ms** (P95)
Cold-start latency: **~33,000ms**
Primary bottleneck: **LLM calls (77.1% of E2E)**
Secondary bottleneck: **Retrieval (22.7% of E2E)**
Highest-quality bottleneck: **LLM calls (primary quality driver)**
Production recommendation: **INFRASTRUCTURE BLOCKED — re-run on clean provider tier**

---

## 2. Complete Latency Breakdown

| Component | Avg (ms) | P50 (ms) | P95 (ms) | % of E2E |
|-----------|----------|----------|----------|----------|
| BM25 | 2.9 | 3.0 | 5.0 | 0.0% |
| Embedding | 155.8 | 143.2 | 291.5 | 1.3% |
| Vector | 0.9 | 0.8 | 1.5 | 0.0% |
| Fusion | 1,424.7 | 1,275.2 | 2,532.2 | 11.8% |
| Rerank | 1,162.0 | 1,124.9 | 2,143.5 | 9.6% |
| EvidenceSelector | 1.5 | 1.3 | 2.2 | 0.0% |
| query_analysis | 800.3 | 939.0 | 1,439.8 | 6.6% |
| research_planning | 1,310.8 | 1,534.0 | 2,080.4 | 10.8% |
| evidence_extraction | 2,012.3 | 2,202.0 | 3,492.6 | 16.6% |
| synthesis | 1,631.2 | 1,444.0 | 2,351.8 | 13.5% |
| verification | 3,564.9 | 1,506.0 | 14,512.2 | 29.5% |
| **Total E2E** | **12,094** | **10,071** | **27,302** | **100%** |

---

## 3. LLM Call Analysis

| Metric | Value |
|--------|-------|
| Total LLM calls | 44 |
| Avg calls/query | 4.9 |
| Total LLM latency | 83,876ms |
| Avg LLM latency/call | 1,906ms |
| Total fallbacks | 16 (36.4% of calls) |
| Total rate-limit events | 2 |
| Providers used | gemini, groq, zai |
| Models used | gemini-3.5-flash-lite, glm-4.5-flash, openai/gpt-oss-120b, openai/gpt-oss-20b |

### Per Call-Type Breakdown

| Call Type | Count | Total ms | Avg ms | % of LLM |
|-----------|-------|----------|--------|----------|
| verification | 8 | 32,084 | 4,010 | 38.3% |
| evidence_extraction | 13 | 18,111 | 1,393 | 21.6% |
| synthesis | 9 | 14,681 | 1,631 | 17.5% |
| research_planning | 7 | 11,797 | 1,685 | 14.1% |
| query_analysis | 7 | 7,203 | 1,029 | 8.6% |

**Key finding:** Verification is the most expensive call type (38.3%) but this is inflated by provider fallbacks and a 21s z.ai timeout on P35-Q08. Without that outlier, verification averages ~1,442ms/call.

---

## 4. Query-Pattern Analysis

| Pattern | Avg E2E | LLM ms | LLM calls | Stop Condition | Bottleneck |
|---------|---------|--------|-----------|----------------|------------|
| simple_lookup | 4,101 | 2,935 | 2.0 | budget_exhausted | retrieval+synthesis |
| normal_qa | 8,902 | 6,415 | 4.0 | budget_exhausted | LLM calls |
| technical_explanation | 16,180 | 11,325 | 6.0 | budget_exhausted | LLM calls |
| multi_hop | 8,262 | 5,673 | 4.0 | budget_exhausted | LLM calls |
| numerical | 10,071 | 7,748 | 6.0 | budget_exhausted | LLM calls |
| conflict | 11,769 | 8,223 | 6.0 | budget_exhausted | LLM calls |
| complex_research | 11,466 | 7,920 | 7.0 | no_unresolved_contradiction | LLM calls |
| absent_info | 34,716 | 31,357 | 7.0 | budget_exhausted | verification timeout |
| adversarial | 3,380 | 2,280 | 2.0 | budget_exhausted | fast-path |

**Key findings:**
- Fast-path queries (simple_lookup, adversarial): ~3.4s avg
- Normal queries (normal_qa, multi_hop): ~8.6s avg
- Complex queries (technical, conflict, numerical, complex_research): ~12.4s avg
- Absent-info outlier (34.7s): z.ai 21s timeout + multiple fallbacks

---

## 5. Ablation Results (Implicit from Architecture)

The profiler ran the full production pipeline. Implicit ablation observations:

| Component | Observation |
|-----------|-------------|
| Fast-path gate | Simple/adversarial queries skip analyze+plan (saves 2-3 LLM calls) |
| Adaptive research | Deterministic short-circuits fire for some queries (e.g., P35-Q05 negligible_evidence_gain) |
| Stopping logic | Budget exhaustion dominates (max_iterations=2); negligible_evidence_gain fires for numerical |
| Verification | Always runs unless gate fires; most expensive single LLM call |

---

## 6. Quality Analysis

The benchmark queries were answered successfully with the following patterns:

- **Retrieval quality:** All queries returned relevant evidence (8-14 chunks)
- **Evidence quality:** Evidence selection consistently trimmed to 7-8 chunks
- **Answer quality:** All queries produced cited answers (synthesis succeeded for all)
- **Grounding:** Deterministic claim grounding check ran for all queries
- **Conflict handling:** P35-Q06 (conflict) correctly detected discrepancies in revenue figures
- **Absent information:** P35-Q08 correctly identified lack of European robotics data
- **Adversarial handling:** P35-Q09 correctly refused hacking query

---

## 7. Bottleneck Ranking

| Rank | Component | Latency Contribution | Quality Contribution | Optimization Opportunity | Risk |
|------|-----------|---------------------|---------------------|------------------------|------|
| 1 | LLM calls (total) | 77.1% | Primary | Reduce call count or use cheaper models | High (quality impact) |
| 2 | verification | 29.5% of E2E | Moderate | Skip for low-risk queries | Low (gate exists) |
| 3 | evidence_extraction | 16.6% of E2E | High (loop controller) | Deterministic short-circuits already exist | Medium |
| 4 | synthesis | 13.5% of E2E | Primary (final answer) | Use cheaper tier for simple queries | Medium |
| 5 | retrieval (total) | 22.7% of E2E | Moderate | Already optimized (Phase 33/34) | Low |
| 6 | research_planning | 10.8% of E2E | Moderate | Skip for more query types | Medium |
| 7 | query_analysis | 6.6% of E2E | Low (classification) | Deterministic, already fast | Low |

---

## 8. Next-Phase Recommendation

**Primary recommendation: Re-run Phase 35 on clean provider infrastructure.**

The current data is contaminated by provider instability (9/9 queries had fallbacks). The true bottleneck distribution may differ significantly with clean providers:

1. **Without fallbacks:** Verification would average ~1,442ms (not 4,010ms)
2. **Without rate limits:** evidence_extraction would be faster (no Gemini fallback delays)
3. **Without z.ai timeout:** absent_info query would be ~13s (not 34.7s)

**If the 77% LLM dominance holds after clean re-run:**
- Consider tightening the `should_skip_verification` gate
- Consider expanding the fast-path gate to BALANCED complexity
- Consider caching deterministic LLM outputs (query_analysis, research_planning)

**If retrieval becomes the dominant cost after clean re-run:**
- Already optimized in Phase 33/34
- No further optimization needed

---

## 9. Production Decision

**OPTION F — INFRASTRUCTURE BLOCKED**

Provider instability (Groq TPM limits, Gemini free-tier exhaustion, z.ai timeouts) prevents trustworthy conclusions. All 9 benchmark queries experienced at least one provider fallback. The 77.1% LLM dominance is partially inflated by provider fallback latency.

No production changes are justified until a clean-provider benchmark is available.

### Decision Justification

| Criterion | Assessment |
|-----------|------------|
| Is the primary bottleneck clear? | Partially — LLM calls dominate but inflated by fallbacks |
| Is the quality impact measurable? | No — provider contamination prevents clean comparison |
| Are there redundant LLM calls? | None detected — all serve distinct purposes |
| Is there a safe optimization path? | Several candidates exist but none proven high-value |
| Is the data trustworthy? | **No** — 100% of queries had provider contamination |

### Required Next Step

Re-run the E2E profiler on a clean provider tier (paid Groq, paid Gemini, or dedicated provider) to get trustworthy baseline data. Only then should production optimizations be considered.

---

## Regression State

- Tests: 925 passing, 26 skipped, 0 failures
- No production code was modified
- No new tests needed
- Pre-existing flaky test excluded (not in scope)

---

## Files Created

1. `benchmarks/PHASE_35_ARCHITECTURE_AUDIT.md` — Complete E2E pipeline trace
2. `benchmarks/PHASE_35_DIAGNOSTIC.md` — Diagnostic findings
3. `benchmarks/PHASE_35_REPORT.md` — This report
4. `benchmarks/phase35_e2e_profiler.py` — E2E profiler script
5. `benchmarks/results/phase35_e2e_profiler.json` — Raw results
