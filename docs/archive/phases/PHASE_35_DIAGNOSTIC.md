# Phase 35 Diagnostic

**Date:** 2026-09-07
**Status:** Complete — measurement and ablation finished

---

## E2E Latency (Warm)

| Metric | Value |
|--------|-------|
| Warm avg | 12,094ms |
| Warm P50 | 10,071ms |
| Warm P95 | 27,302ms |
| Cold start | ~33,000ms |

## Component Breakdown

| Component | Avg (ms) | % of E2E |
|-----------|----------|----------|
| LLM Calls | 9,320 | **77.1%** |
| Retrieval (BM25+Vec+Fuse+Rerank) | 2,746 | 22.7% |
| Evidence Selection | 1.5 | 0.0% |
| Other (classify, plan, verify) | 27 | 0.2% |

## LLM Call Breakdown

| Call Type | Count | Total ms | Avg ms | % of LLM |
|-----------|-------|----------|--------|----------|
| verification | 8 | 32,084 | 4,010 | **38.3%** |
| evidence_extraction | 13 | 18,111 | 1,393 | 21.6% |
| synthesis | 9 | 14,681 | 1,631 | 17.5% |
| research_planning | 7 | 11,797 | 1,685 | 14.1% |
| query_analysis | 7 | 7,203 | 1,029 | 8.6% |

Total LLM calls: 44 (avg 4.9/query)

## Primary Bottleneck

**LLM calls (77.1% of E2E)** — dominated by verification (38.3% of LLM time). Verification latency is inflated by provider fallbacks and timeouts (z.ai 21s timeout on P35-Q08).

## Secondary Bottleneck

**Retrieval (22.7% of E2E)** — fusion (11.8%) and reranking (9.6%) dominate retrieval. Embedding is 1.3%.

## Provider Contamination

9/9 queries contaminated by fallbacks (Groq TPM limits hit, Gemini free tier exhausted, z.ai timeout). No clean comparison possible. Results are statistically unreliable for model quality ranking.

## Production Decision

**OPTION F — INFRASTRUCTURE BLOCKED**

Provider instability (Groq TPM limits, Gemini free-tier exhaustion, z.ai timeouts) prevents trustworthy conclusions. The 77.1% LLM dominance is partially inflated by provider fallbacks. A re-run on a paid-tier / dedicated provider is needed before any production optimization is justified.

---

## Detailed Findings

### Finding 1: LLM Dominates E2E Latency (77.1%)

With an average of 4.9 LLM calls per query consuming 9,320ms of the 12,094ms total, LLM calls are the clear dominant cost. This is NOT inherently a problem — LLM calls are the primary quality driver. The question is whether any are redundant.

### Finding 2: Verification is the Most Expensive Call Type (38.3%)

Verification consumes 32,084ms across 8 calls (4,010ms avg). This is inflated by:
- P35-Q08: z.ai timeout (21,127ms) — a single outlier
- Multiple provider fallbacks (Groq TPM → Gemini free → z.ai → Gemini)

Without the z.ai timeout outlier, verification would average ~1,442ms per call, which is comparable to other LLM calls.

### Finding 3: Fast-Path Queries Are Fast

P35-Q01 (simple_lookup, 3,148ms) and P35-Q09 (adversarial, 3,380ms) both took the fast-path (1-2 LLM calls). This confirms the fast-path gate is working correctly.

### Finding 4: Complex Queries Hit Budget Exhaustion

Most queries stopped due to `budget_exhausted` (max_iterations=2). This means the planner proposes 2+ subquestions but the budget only allows 2 iterations. The gap detector keeps finding gaps and queuing more subqueries.

### Finding 5: Retrieval Is Well-Optimized

At 22.7% of E2E, retrieval is reasonable. The main costs are fusion (~1,425ms avg) and reranking (~1,162ms avg). Both are justified by Phase 33/34 findings.

### Finding 6: Evidence Selection Is Negligible

At 1.5ms avg, evidence selection is not a bottleneck and requires no optimization.

### Finding 7: Provider Instability Inflates All Numbers

Every single query experienced at least one fallback. This means:
- Provider fallback latency is counted in LLM call time
- Some calls are on slower fallback models (gemini-3.5-flash-lite vs groq models)
- Rate limit retries add latency
- The z.ai timeout (21s) on P35-Q08 is a provider infrastructure issue, not an ARGUS issue

### Finding 8: No Redundant LLM Calls Detected

Every LLM call serves a distinct purpose:
- query_analysis: Complexity classification (skipped for fast-path)
- research_planning: Subquery decomposition (skipped for fast-path)
- evidence_extraction: Evidence sufficiency assessment
- synthesis: Final answer generation
- verification: Post-synthesis quality check

None are redundant. The question is whether any could be made cheaper or eliminated.

---

## Potential Optimization Targets (Unverified)

1. **Skip verification for low-risk queries** — Currently gated by `should_skip_verification()` but the gate is not firing for most queries. Could be tightened.

2. **Skip query_analysis + research_planning for more queries** — Currently only FAST queries skip these. BALANCED queries still pay the 2-3s cost. Could expand the fast-path gate.

3. **Reduce verification fallback latency** — The z.ai 21s timeout is a provider issue. Could skip z.ai for verification calls.

4. **Parallelize assess + retrieve** — Currently sequential. Could potentially run in parallel if the assessment doesn't depend on the latest retrieval results.

5. **Cache query_analysis + research_planning** — Deterministic, could be cached for repeated queries.

None of these are proven high-value without controlled ablation. A clean-provider re-run is needed first.
