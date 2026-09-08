# PHASE 27 REPORT — Validated End-to-End Answer Quality & Synthesis Audit

**Date:** 2026-09-07
**Status:** COMPLETE
**Final Recommendation:** OPTION C — Further Research Needed

---

## 1. What We Did

### 1.1 Evaluator Audit & Improvement
- Audited Phase 26 evaluator: found 27 weaknesses (8 critical, 9 high, 10 medium)
- Rewrote `app/evaluation/answer_quality.py` with critical fixes:
  - Negation detection in gold-fact matching
  - Stronger SUPPORTED classification (requires number match when numbers present)
  - Exact integer matching (catches year mismatches)
  - Compound splitting for lowercase continuations
  - Weighted support rate using partial_coverage_ratio
  - Query relevance checking
  - Citation precision fix (no double-counting)
- Added 19 new tests (48 total, all passing)

### 1.2 Benchmark Construction
- Built curated core set of 21 queries across 9 categories
- Gold facts verified from corpus documents
- Categories: simple_lookup, multi_doc_synthesis, multi_hop, conflict, numerical, technical_explanation, complex_research, absent_info, adversarial

### 1.3 Real ARGUS Evaluation
- Ran actual ARGUS pipeline end-to-end on all 21 queries
- Evaluated answers using improved deterministic evaluator
- Classified failures by root cause

---

## 2. Results

### 2.1 Overall Metrics

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Claim Support Rate | 22.0% | 78% of claims unsupported |
| Citation Presence | 65.7% | 1/3 of claims lack citations |
| Citation Precision | 80.0% | Citations are mostly relevant |
| Query Relevance | 81.5% | Answers are topically relevant |
| Gold Fact Coverage | 25.0% | 75% of expected facts missing |
| Gold Facts Negated | 36 | Many facts contradicted |

### 2.2 Per-Class Performance

| Category | Claim Support | Citation Precision | Query Relevance |
|----------|--------------|-------------------|-----------------|
| simple_lookup | 54.7% | 100.0% | 100.0% |
| multi_hop | 46.5% | 83.3% | 79.6% |
| adversarial | 24.6% | 100.0% | 73.2% |
| conflict | 13.4% | 100.0% | 77.4% |
| numerical | 16.4% | 83.3% | 83.3% |
| technical_explanation | 15.7% | 75.0% | 87.5% |
| complex_research | 12.1% | 75.0% | 74.7% |
| multi_doc_synthesis | 7.9% | 60.0% | 79.4% |
| absent_info | 4.5% | 50.0% | 80.0% |

### 2.3 Failure Classification

| Failure Type | Count | Percentage |
|-------------|-------|-----------|
| synthesis_failure | 17 | 81% |
| numerical_error | 7 | 33% |
| evidence_failure | 5 | 24% |
| citation_failure | 3 | 14% |
| absent_info_not_handled | 2 | 10% |
| contradiction_missed | 2 | 10% |
| off_topic | 0 | 0% |
| retrieval_failure | 0 | 0% |

---

## 3. Quality Layer Decomposition

```
Layer               Metric                  Score    Status
─────────────────────────────────────────────────────────
Retrieval           Query Relevance         81.5%    ✓ GOOD
Evidence Selection  Citation Precision      80.0%    ✓ GOOD
Citation Placement  Citation Presence       65.7%    ⚠ NEEDS WORK
Synthesis           Claim Support Rate      22.0%    ✗ CRITICAL
Answer Completeness Gold Fact Coverage      25.0%    ✗ CRITICAL
Specialized         Numerical Accuracy      ~67%     ⚠ NEEDS WORK
Specialized         Contradiction Handling  0%       ✗ CRITICAL
Specialized         Absent Info Handling    0%       ✗ CRITICAL
```

**Bottleneck identified: SYNTHESIS LAYER**

---

## 4. Root-Cause Analysis

### 4.1 Primary Root Cause: LLM Not Grounding in Evidence

The synthesis LLM generates answers from its training data rather than from the retrieved evidence chunks. This is the single largest source of quality loss.

**Evidence:**
- 78% of claims are unsupported despite evidence being available
- Citation precision is 80% (evidence IS retrieved and cited correctly)
- But claim support is only 22% (the LLM doesn't USE the evidence)

### 4.2 Secondary Root Causes

1. **No "I don't know" mechanism** — LLM fabricates answers for absent queries
2. **Contradiction not handled** — LLM picks one source without acknowledging conflict
3. **Numerical hallucination** — LLM generates numbers not in evidence
4. **Incomplete synthesis** — LLM omits key facts from evidence

### 4.3 What Is NOT a Root Cause

- ~~Retrieval quality~~ — Working at 81.5% relevance
- ~~Evidence selection~~ — 80% citation precision
- ~~Citation placement~~ — Mostly correct
- ~~Query understanding~~ — 81.5% relevance
- ~~Embedding model~~ — BM25+FAISS hybrid working adequately
- ~~Retrieval planning~~ — Multi-query retrieval functioning

---

## 5. What Would Fix This

### 5.1 High-Impact Fixes (Would Move the Needle)

1. **Evidence-grounded synthesis prompting** — Force LLM to only use facts from provided evidence chunks, with explicit instruction to say "I don't know" when evidence is insufficient
2. **Claim verification post-synthesis** — After synthesis, verify each claim against evidence (already partially implemented in Phase 25)
3. **Contradiction-aware synthesis** — When evidence contradicts, explicitly acknowledge the conflict
4. **Numerical grounding** — Extract numbers from evidence and verify they match answer numbers

### 5.2 Medium-Impact Fixes

5. **Citation presence improvement** — Ensure every claim has a citation
6. **Gold-fact coverage checking** — Post-synthesis check that expected facts are mentioned
7. **Absent-info detection** — Detect when query asks for absent information and generate appropriate response

### 5.3 Low-Impact Fixes (Already Working)

- ~~Retrieval quality~~ — Already good
- ~~Embedding model~~ — Already optimized
- ~~Evidence selection~~ — Already good
- ~~Citation placement~~ — Already good

---

## 6. Final Recommendation

### OPTION C: Further Research Needed

**Rationale:**

The diagnostic reveals that the synthesis layer is the bottleneck, but fixing it requires changes to the synthesis prompting strategy — which is a significant architectural decision that warrants careful design before implementation.

**What we know:**
- Retrieval is working (81.5% relevance)
- Evidence selection is working (80% precision)
- Synthesis is failing (22% support)
- The gap is in how the LLM uses evidence, not in how evidence is found

**What we need to decide:**
1. Should we modify the synthesis prompt to be evidence-faithful?
2. Should we add a post-synthesis verification step?
3. Should we implement a "I don't know" mechanism?
4. How do we handle contradictions in synthesis?

**Recommended next step:** Design an evidence-grounded synthesis strategy (Phase 28) that:
- Forces LLM to only use facts from provided evidence
- Adds explicit "say I don't know when evidence is insufficient" instruction
- Handles contradictions by acknowledging both sides
- Verifies numerical claims against evidence

**Do NOT proceed with Phase 28 until the synthesis strategy is designed and reviewed.**

---

## 7. Regression Requirements

If Phase 28 is implemented:
- All 48 answer quality tests must continue passing
- Claim support rate must improve from 22% baseline
- Citation precision must not degrade from 80%
- Query relevance must not degrade from 81.5%
- Simple lookup performance must not degrade from 54.7%
- No new LLM calls should be added for existing fast-path queries

---

## 8. Files Produced

| File | Purpose |
|------|---------|
| `docs/archive/phases/PHASE_27_DIAGNOSTIC.md` | Detailed diagnostic analysis |
| `benchmarks/PHASE_27_REPORT.md` | This report |
| `benchmarks/benchmark_phase27.py` | Benchmark runner |
| `benchmarks/results/phase27_benchmark_results.json` | Raw results |
| `app/evaluation/answer_quality.py` | Improved evaluator |
| `tests/evaluation/test_answer_quality.py` | 48 tests (all passing) |

---

## 9. Time Spent

| Task | Time |
|------|------|
| Evaluator audit | ~15 min |
| Evaluator rewrite | ~20 min |
| Test updates | ~10 min |
| Benchmark construction | ~15 min |
| Benchmark execution | ~5 min |
| Diagnostic writing | ~10 min |
| Report writing | ~10 min |
| **Total** | **~85 min** |
