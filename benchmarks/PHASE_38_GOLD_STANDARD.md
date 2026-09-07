# Phase 38 — Gold-Standard Dataset Audit

**Status:** COMPLETE  
**Date:** 2026-09-07  

---

## Dataset Overview

Reused `benchmarks/eval_data/eval_plan_v1.json` (38 queries, 10 categories) with 12-document corpus in `benchmarks/eval_data/corpus_v1/`.

### Category Distribution

| Category | N | Description |
|----------|---|-------------|
| simple_lookup | 6 | Direct factual retrieval |
| normal_qa | 4 | Standard QA with synthesis |
| multi_hop | 3 | Cross-document reasoning |
| numerical | 3 | Number extraction/comparison |
| conflict | 3 | Contradictory source detection |
| absent_info | 4 | Questions about absent data |
| technical_explanation | 3 | Conceptual reasoning |
| multi_doc_synthesis | 3 | Multi-source integration |
| complex_research | 3 | Multi-step research |
| adversarial | 6 | Trick questions, edge cases |

### Expected Behaviors

| Behavior | N | Categories |
|----------|---|------------|
| answer_with_citations | 30 | All except absent_info |
| abstain | 4 | absent_info |
| (conflict handling) | 3 | conflict |

### Gold Fact Coverage by Category

| Category | Gold Facts | Coverage |
|----------|-----------|----------|
| simple_lookup | 6 unique entities | 83% found |
| normal_qa | 4 key facts | 100% found |
| multi_hop | 9 facts (5+ entities) | 33% found |
| numerical | 3 precise values | 67% found |
| conflict | 3 contradictions | 83% found |
| absent_info | "not available" | 8% found |
| technical_explanation | 5 concepts | 0% found |
| multi_doc_synthesis | 5 facts (Ohio, Memphis, etc.) | 100% found |
| complex_research | 9 facts (robot, Delta Sync, etc.) | 20% found |
| adversarial | 12 facts (mix) | 17% found |

## Quality of Gold Standard

### Strengths
- 38 queries covers all ARGUS use cases
- Gold facts are precise and verifiable
- 12-document corpus is realistic (Acme Corp scenario)
- Expected behaviors are clearly defined

### Weaknesses
- Some gold facts are too narrow (e.g., "New York City" vs. "New York")
- Technical explanation category has under-specified gold facts
- Conflict gold facts need explicit "acknowledge contradiction" wording
- No query difficulty levels assigned

## Existing Evaluator Coverage

| Metric | Source | Coverage |
|--------|--------|----------|
| Claim support | `answer_quality.py` | Full |
| Citation validation | `answer_quality.py` | Full |
| Gold fact coverage | `answer_quality.py` | Full |
| Numerical consistency | `answer_quality.py` | Full |
| Query relevance | `answer_quality.py` | Full |
| Hallucination detection | `phase38_quality_baseline.py` | New (3.7% rate) |
| Abstention correctness | `phase38_quality_baseline.py` | New (50% correct) |
| Conflict handling | `phase38_quality_baseline.py` | New (0% correct) |
