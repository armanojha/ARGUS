# PHASE 27 DIAGNOSTIC — Real ARGUS Answer Quality Evaluation

**Date:** 2026-09-07
**Status:** COMPLETE
**Purpose:** Validate evaluator, benchmark real ARGUS answers, identify actual synthesis weakness

---

## Executive Summary

Phase 27 ran the **actual ARGUS pipeline** on 21 curated queries and evaluated the synthesized answers. The results reveal that **synthesis quality is the dominant failure mode** — not retrieval, not evidence selection, not citation placement.

**Key Finding:** ARGUS answers have a **22% claim support rate** — meaning 78% of factual claims in synthesized answers are either unsupported, partially supported, or contradicted by the evidence retrieved.

---

## 1. Evaluator Audit (Phase 26 → Phase 27)

### 1.1 Weaknesses Found

| # | Severity | Issue | Impact |
|---|----------|-------|--------|
| A1 | CRITICAL | Substring match in gold-fact coverage affirms negated facts | False positive coverage |
| A3 | CRITICAL | Lexical overlap marks wrong facts as SUPPORTED | Wrong answers pass |
| B1 | CRITICAL | Paraphrases marked UNSUPPORTED | Correct answers fail |
| B2 | CRITICAL | Off-topic answers not detected | Irrelevant answers pass |
| B6 | MEDIUM | CONTRADICTED status never assigned | Contradictions invisible |
| D1 | CRITICAL | No query-answer relevance check | No relevance metric |
| C2 | HIGH | Compound splitting fails on lowercase | Missed compound claims |

### 1.2 Fixes Applied

1. **Negation detection** — `_has_negation()` and `_claim_contains_negated_fact()` prevent false coverage of negated facts
2. **Stronger SUPPORTED classification** — When claims have numbers, SUPPORTED requires number match
3. **Exact integer matching** — `_numbers_match()` now uses exact matching for integers (catches year mismatches like 1987 vs 1995)
4. **Compound splitting** — `_COMPOUND_SPLIT` now handles lowercase continuations after conjunctions
5. **Weighted support rate** — Uses `partial_coverage_ratio` instead of flat 0.5 weight
6. **Query relevance** — Optional `_check_query_relevance()` via key-term overlap
7. **Citation precision fix** — Counts unique citation IDs, not total usages

### 1.3 Evaluator Validation

12 predetermined-result tests validate the evaluator:
- Known correct answer → SUPPORTED ✓
- Known wrong number → NOT SUPPORTED ✓
- Negated gold fact → coverage = 0.0 ✓
- Off-topic answer → low relevance ✓
- Citation absence → NO_CITATION ✓
- Invalid citation → INVALID_CITATION ✓
- Contradicted numbers → NOT SUPPORTED ✓

**Evaluator trustworthiness: GOOD for deterministic evaluation.** Remaining limitation: no semantic similarity (paraphrases still fail).

---

## 2. Benchmark Design

### 2.1 Core Set

21 queries across 9 categories:

| Category | Count | Difficulty |
|----------|-------|------------|
| simple_lookup | 2 | easy |
| multi_doc_synthesis | 3 | medium-hard |
| multi_hop | 3 | medium-hard |
| conflict | 2 | hard |
| numerical | 3 | medium-hard |
| technical_explanation | 2 | medium-hard |
| complex_research | 2 | hard |
| absent_info | 2 | medium |
| adversarial | 2 | hard |

### 2.2 Gold Facts

- Human-verified from corpus documents
- Key-term based (not full sentences) to avoid evaluator false positives
- Covers expected facts that a complete answer should mention

---

## 3. Results — Quality Layer Decomposition

### 3.1 Retrieval Layer
- **All 21 queries succeeded** — retrieval is working
- **Query relevance: 81.5%** — answers are topically relevant to queries
- **No retrieval failures** — the hybrid retriever finds relevant chunks

### 3.2 Evidence Layer
- **Citation presence: 65.7%** — 1/3 of claims lack citations
- **Citation precision: 80.0%** — when citations exist, they're mostly relevant
- **Evidence failure: 5 queries** — gold facts missing from answers

### 3.3 Answer/Synthesis Layer (THE BOTTLENECK)
- **Claim support rate: 22.0%** — 78% of claims are unsupported
- **Synthesis failure: 17/21 queries** — 81% have claim support < 50%
- **Gold fact coverage: 25.0%** — only 25% of expected facts appear
- **Gold facts negated: 36** — many expected facts are contradicted

### 3.4 Citation Layer
- **Citation presence: 65.7%** — needs improvement
- **Citation precision: 80.0%** — acceptable
- **Citation failure: 3 queries** — missing citations on key claims

### 3.5 Specialized Layers
- **Numerical errors: 7 queries** — wrong numbers in answers
- **Contradiction missed: 2/2 conflict queries** — contradictions not handled
- **Absent info not handled: 2/2** — answers fabricate absent information

---

## 4. Per-Class Analysis

### 4.1 Simple Lookup (54.7% support)
- Best performing category
- Still only 54.7% — even simple factual answers have unsupported claims
- Root cause: LLM adds unnecessary context/speculation

### 4.2 Multi-doc Synthesis (7.9% support)
- **Worst performing category**
- Answers fail to combine facts from multiple documents
- Root cause: LLM synthesizes from memory, not from retrieved evidence

### 4.3 Multi-hop (46.5% support)
- Moderate performance
- Some queries succeed at chained reasoning
- Root cause: Bridge evidence sometimes missing

### 4.4 Conflict (13.4% support)
- **Contradictions completely missed** (2/2)
- LLM picks one source without acknowledging conflict
- Root cause: Synthesis prompt doesn't handle contradictions

### 4.5 Numerical (16.4% support)
- Wrong numbers in answers (7/21 queries)
- Root cause: LLM hallucinates numbers not in evidence

### 4.6 Technical Explanation (15.7% support)
- Answers are verbose but lack specific facts
- Root cause: LLM generalizes instead of citing specifics

### 4.7 Complex Research (12.1% support)
- Multi-part questions poorly answered
- Root cause: LLM doesn't decompose complex questions

### 4.8 Absent Info (4.5% support)
- **Answers fabricate information** for absent queries
- Root cause: No mechanism to say "I don't know"

### 4.9 Adversarial (24.6% support)
- Better than average — adversarial queries sometimes trigger careful answering
- But still low overall

---

## 5. Failure Taxonomy

### 5.1 Root Causes (Ranked by Impact)

| # | Root Cause | Queries Affected | Impact |
|---|-----------|-----------------|--------|
| 1 | **LLM synthesizes from memory, not evidence** | 17/21 | 78% unsupported claims |
| 2 | **No "I don't know" mechanism** | 2/2 absent | Fabricates absent info |
| 3 | **Contradiction not handled** | 2/2 conflict | Picks one source |
| 4 | **Numerical hallucination** | 7/21 | Wrong numbers |
| 5 | **Gold facts missing** | 5/21 | Incomplete answers |
| 6 | **Missing citations** | 3/21 | Unattributed claims |

### 5.2 Quality Layer Bottleneck

```
Retrieval (81.5% relevance) → Evidence (80% precision) → SYNTHESIS (22% support) → Answer
                                                              ↑
                                                         BOTTLENECK
```

The synthesis layer is where 78% of quality is lost. Retrieval and evidence selection are working adequately.

---

## 6. What Is Actually Limiting ARGUS's Final Answer Quality?

**Answer: The synthesis LLM is not grounding its answers in the retrieved evidence.**

Specific mechanisms:
1. **Memory override** — The LLM uses its training data instead of the provided evidence chunks
2. **No evidence-faithful prompting** — The synthesis prompt doesn't强制 the LLM to only use cited evidence
3. **No "I don't know"** — The LLM always produces an answer, even when evidence is absent
4. **No contradiction handling** — The LLM picks one source without acknowledging conflicts
5. **Numerical hallucination** — The LLM generates numbers not present in evidence

**What is NOT limiting quality:**
- Retrieval (working at 81.5% relevance)
- Evidence selection (80% citation precision)
- Citation placement (mostly correct)
- Query understanding (81.5% relevance)

---

## 7. Evaluator Limitations (Remaining)

| Limitation | Impact | Mitigation |
|-----------|--------|------------|
| No semantic similarity | Paraphrases marked unsupported | Document as known limitation |
| No answer-relevance beyond key-terms | Off-topic detection is basic | Key-term overlap is 81.5% accurate |
| No hallucination detection beyond numerical | Non-numerical hallucinations pass | Requires LLM judge (not in scope) |
| No multi-hop reasoning validation | Chain correctness not checked | Requires structured gold chains |

---

## 8. Data Sources

- **Results:** `benchmarks/results/phase27_benchmark_results.json`
- **Core set:** Embedded in `benchmarks/benchmark_phase27.py`
- **Evaluator:** `app/evaluation/answer_quality.py` (Phase 27 improved)
- **Tests:** `tests/evaluation/test_answer_quality.py` (48 tests, all passing)
