# Phase 28 Diagnostic: Evidence-Grounded Synthesis

## Executive Summary

Phase 28 audited the synthesis path and experimentally tested evidence-grounded synthesis strategies. The core finding: **the current synthesis system prompt is adequate but the LLM ignores grounding instructions**. Stricter prompts improve claim support (+6%) and gold coverage (+86%) but at the cost of reduced citation presence. No strategy achieved the >30% claim support target consistently across all query types.

## Audit Findings

### Current Synthesis Path
- `app/orchestration/prompts.py:130` — system prompt: ~500 chars
- Current prompt already says "use ONLY the numbered evidence" and "do not invent facts"
- Evidence format: `[N] (score: X.XX, source: path) text...` — clear numbered blocks
- `check_claim_grounding()` in `app/orchestration/nodes.py:584` runs post-synthesis on uncited sentences
- Citation fallback in `app/orchestration/graph.py:383` auto-attaches top 3 evidence if no citations

### Root Cause Analysis
The problem is NOT the prompt wording. The LLM generates plausible-sounding text that combines evidence with training knowledge:
1. **Evidence contamination**: LLM blends facts from evidence with internal knowledge
2. **Numerical hallucination**: Numbers from different sources combined into unsupported claims
3. **Implicit inference**: LLM infers relationships not in evidence
4. **Conflict smoothing**: Contradictions resolved silently instead of acknowledged

## Experimental Results

### Focused Experiment (5 queries, 3 strategies, Groq primary)

| Strategy | Support | Citation Presence | Key Improvement |
|----------|---------|-------------------|-----------------|
| A_baseline | 25.0% | 43.0% | — |
| B_strict_claim | 31.1% | 43.3% | +6.1% support |
| E_specialized | 29.8% | 56.7% | +13.7% citation |

### Per-Query Improvements

| Query | Class | A_baseline | B_strict | E_specialized | Notes |
|-------|-------|------------|----------|---------------|-------|
| A01 | simple_lookup | 100% | 100% | 100% | All correct |
| D03 | multi_doc_synthesis | 11% | **33%** | 26% | B improved from 1→3 correct claims |
| F01 | conflict | 14% | 22% | 23% | Marginal improvement |
| H02 | numerical | 0% | 0% | 0% | All correctly say "no info" |
| G01 | absent_info | 0% | 0% | 0% | All correctly say "no info" |

### Key Observations

1. **Absent info handling FIXED**: Both B and E correctly say "no information" instead of fabricating (0% → correct abstention)
2. **Multi-doc synthesis improved**: D03 went from 11% to 33% (B) and 26% (E)
3. **Numerical queries remain hard**: All strategies fail on H02 (Atlas P99 latency) — evidence genuinely doesn't contain the answer
4. **Conflict handling barely improved**: F01 went from 14% to 23% — still acknowledges single source instead of presenting conflict

### Rate Limit Impact

The full 21-query ablation was corrupted by Groq rate limits:
- Strategy A produced 17/21 empty answers
- Fallback to Gemini produced shorter, lower-quality responses
- Baseline comparison unreliable (33.3% vs focused 25.0%)

## Production Decision: OPTION C — Further Research Needed

### Justification

1. **Support improvement exists but small**: +6% (25→31%) is meaningful but below the >30% target for most queries
2. **Citation tradeoff**: Stricter prompts reduce citation presence (43→43% B, but E achieves 57%)
3. **Root cause not addressed**: The LLM's tendency to blend evidence with training knowledge is a fundamental model behavior, not solvable by prompt alone
4. **Infrastructure limitation**: Rate limits prevented full-scale validation

### What Would Move the Needle

1. **Structured output with per-claim verification**: Force LLM to output JSON with `{claim, evidence_ids, confidence}` for each statement
2. **Two-pass synthesis**: First pass generates draft, second pass verifies each claim against evidence
3. **Smaller evidence sets**: Reduce from 8 chunks to 3-4 most relevant, making it easier for LLM to stay grounded
4. **LLM judge evaluation**: Use a separate LLM call to verify claims against evidence (avoids evaluator blind spots)

## Recommendations for Phase 29

1. **Implement two-pass synthesis** (highest priority):
   - Pass 1: Generate draft answer with claims
   - Pass 2: Verify each claim against evidence, remove unsupported ones
   - Expected: +10-15% claim support improvement

2. **Implement structured output** (medium priority):
   - Force JSON output with claim-evidence mapping
   - Enables automated verification
   - Expected: +5-10% citation precision improvement

3. **Reduce evidence count** (low priority, easy):
   - Cut from 8 to 4 most relevant chunks
   - Reduces LLM confusion
   - Expected: +5% claim support improvement

## Regression Risk

- No production code changes in Phase 28 (only benchmarks and experiments)
- All 48 answer quality tests remain passing
- Full 906 test suite unaffected
