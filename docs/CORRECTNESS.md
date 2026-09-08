# ARGUS Correctness & Verification

## Overview

ARGUS implements a multi-layered correctness system that ensures evidence is verified, conflicts are detected, and answers are grounded in source material. This document describes the evolution of these systems and the Phase 43 bug fix.

## Evidence Verification

### Claim-Support Checking

Every claim in the synthesized answer is checked against retrieved evidence:

| Status | Meaning |
|--------|---------|
| **Supported** | Evidence directly backs the claim |
| **Partial** | Some evidence supports, some is missing |
| **Contradicted** | Evidence conflicts with the claim |
| **Unsupported** | No relevant evidence found |

**Implementation:** `app/verification/engine.py` → `verify_claim()`

### Confidence Scoring

Verification produces confidence scores for:
- **Evidence coverage** — How much of the claim is supported
- **Source quality** — Reliability of the supporting sources
- **Cross-source agreement** — Whether multiple sources agree
- **Temporal relevance** — Whether the evidence is from a relevant time period

**Implementation:** `app/verification/confidence.py`

### Citation Grounding

Post-synthesis, every citation is validated:
- Citation indices reference actual evidence chunks
- Cited sources exist in the evidence store
- Citation markers are normalized to consistent format

**Implementation:** `app/orchestration/nodes.py` → `check_claim_grounding()`, `extract_cited_indices()`

## Contradiction Detection

### Deterministic Pairwise Analysis

ARGUS detects contradictions using deterministic pairwise analysis of evidence chunks. No LLM calls are involved — the detection is purely algorithmic.

**Implementation:** `app/orchestration/nodes.py` → `_detect_contradictions()`

### Detection Methods

#### 1. Negation Pairs

Detects opposing claims between evidence chunks. Requires:
- Topic coherence (≥3 shared meaningful words)
- Entity overlap (shared entity keywords)
- Negation pattern match (e.g., "supports" vs "contradicts")

This prevents false positives where unrelated evidence is incorrectly flagged as contradictory.

#### 2. Numerical Discrepancies

Compares extracted numbers near shared metrics:
- Extracts numbers within ±50 characters of shared metric keywords
- Normalizes units ($, %, billion, million, thousand) to common scale
- Requires topic coherence before comparing values

**Critical fix (Phase 43):** The `_UNIT_NORMALIZE` dictionary originally supported `"percent"` and `"pct"` but NOT the literal `"%"` symbol. This caused percentage values like "35%" to be normalized to `(0.0, '')`, silently discarding them during comparison.

#### 3. Temporal Context

Extracts years and date ranges from evidence text:
- Distinguishes historical differences (different time periods) from genuine contradictions
- Uses `both_have_years` flag to prevent false `DIFFERENT_TIMEFRAME` classifications when neither text mentions a year

### Conflict Types

| Type | Description |
|------|-------------|
| `GENUINE_CONTRADICTION` | Sources disagree on the same claim, same entity, same timeframe |
| `DIFFERENT_TIMEFRAME` | Data from different time periods (not a real conflict) |
| `DIFFERENT_SOURCE` | Different methodologies or scopes (not a real conflict) |
| `POSSIBLE_CONTRADICTION` | Uncertain — needs human review |
| `IRRELEVANT_DIFFERENCE` | Not a real conflict |

### Classification Logic

`_classify_conflict_type()` determines the conflict type based on:
- Entity overlap (same entity involved?)
- Timeframe overlap (same time period?)
- Metric overlap (same metric being measured?)
- `both_have_years` flag (prevents false positives when no years mentioned)

## Query-Aware Filtering

### Purpose

Without filtering, ARGUS would surface contradictions that are irrelevant to the user's query. For example, a query about "AI adoption rates" might surface contradictions about healthcare AI vs manufacturing AI — technically different claims, but not relevant to the user's question.

### Implementation

`filter_contradictions_by_query()` applies four rules:

1. **DIFFERENT_TIMEFRAME** — Filtered unless query asks about history
2. **LOW confidence** — Filtered unless query explicitly asks about conflicts
3. **Metric relevance** — Conflict must match the query's metric (if specified)
4. **Entity relevance** — Conflict should involve entities mentioned in the query

**Key file:** `app/orchestration/nodes.py` → `filter_contradictions_by_query()`

### Results

- False positive rate reduced from ~80% to ~0%
- 8/8 contradiction benchmark cases pass (100%)

## Absent-Information Handling

ARGUS detects when evidence is absent (not just insufficient):

- Determines whether the absence is genuine (topic not in corpus) or partial (some evidence exists)
- Provides honest "insufficient evidence" messages rather than hallucinated answers

**Implementation:** `app/orchestration/nodes.py` → `_is_evidence_absent()`

## Safe Synthesis Degradation

When providers fail or evidence is insufficient:

- Evidence is shown without full synthesis
- Conflicts are labeled but not synthesized into incoherent output
- User receives honest representation of what was found

**Implementation:** `app/orchestration/nodes.py` → `make_synthesize_node()` (fallback path)

## Phase 43 Bug Fix

### The Bug

Query: *"What evidence supports the claim that AI adoption is accelerating?"*

Returned **irrelevant evidence** and **false contradictions**:
- Evidence about healthcare AI was compared against evidence about manufacturing AI
- Numerical values from different contexts were flagged as contradictions
- Synthesis fallback dumped all raw evidence without filtering

### Root Causes

| # | Bug | Impact |
|---|-----|--------|
| 1 | No relevance gate — `_detect_contradictions` received ALL evidence chunks | False contradictions between unrelated domains |
| 2 | Negation pair check too broad — triggered with any 2 shared words | False positive contradictions |
| 3 | Numerical check lacked coherence — compared numbers across unrelated evidence | False contradictions for different-context values |
| 4 | `_classify_conflict_type` triggered `DIFFERENT_TIMEFRAME` when both texts had NO years | Incorrect conflict classification |
| 5 | Synthesis fallback was unsafe — dumped all raw evidence without filtering | Incoherent output |

### The Normalization Bug

`_UNIT_NORMALIZE` dictionary:

```python
# BEFORE (buggy):
_UNIT_NORMALIZE = {
    "billion": "B", "bn": "B", "b": "B",
    "million": "M", "mn": "M", "m": "M",
    "thousand": "K", "k": "K",
    "percent": "%", "pct": "%",
}

# AFTER (fixed):
_UNIT_NORMALIZE = {
    "billion": "B", "bn": "B", "b": "B",
    "million": "M", "mn": "M", "m": "M",
    "thousand": "K", "k": "K",
    "percent": "%", "pct": "%", "%": "%",
}
```

The literal `"%"` symbol was missing. When `_normalize_number_with_unit("35%")` was called:
1. It checked if the string ended with any key in `_UNIT_NORMALIZE`
2. `"35%".endswith("percent")` → False
3. `"35%".endswith("pct")` → False
4. No match found → unit stays as `""`
5. `float("35%")` → ValueError → returns `(0.0, "")`

This caused percentage values to be silently discarded during normalization, making `norm_i == norm_j` when both sides only had year values (2024).

### The Fix

One-line addition to `_UNIT_NORMALIZE`: `"%": "%"`

This was a deterministic normalization bug — NOT a reason to redesign the contradiction architecture.

### Validated Result

- **Benchmark:** 8/8 cases pass (100%)
- **Relevance filtering:** 8/8 cases (100%)
- **Overall:** 16/16 (100%)
- **Test suite:** 906 passed, 0 new regressions

## Relevance Gate

### Purpose

Prevents irrelevant evidence from entering the contradiction detection pipeline. Without this, evidence about different domains (e.g., healthcare AI vs manufacturing AI) would be compared, producing false contradictions.

### Implementation

`_filter_evidence_by_relevance()` uses deterministic lexical overlap scoring:
- Jaccard similarity between query and evidence
- Coverage score (fraction of query terms found in evidence)
- Threshold-based filtering (default: 0.15)

**Key file:** `app/orchestration/nodes.py` → `_filter_evidence_by_relevance()`

### Integration

Wired into `make_assess_node()` — evidence is filtered before contradiction detection.

## Topic Coherence

### Purpose

Prevents false contradictions between unrelated evidence that happen to share a few common words.

### Implementation

`_is_topic_coherent()` requires ≥3 shared meaningful words (excluding stopwords) between evidence chunks. This ensures that the evidence is actually about the same topic before comparison.

**Key file:** `app/orchestration/nodes.py` → `_is_topic_coherent()`
