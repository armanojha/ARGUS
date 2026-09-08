# Phase 43: Real-World Correctness Fix

## Problem
Query: *"What evidence supports the claim that AI adoption is accelerating?"*

Returned **irrelevant evidence** and **false contradictions**:
- Evidence about healthcare AI was compared against evidence about manufacturing AI (different domains)
- Numerical values (e.g., "73%" vs "67%") from **different contexts** were flagged as contradictions
- Synthesis fallback dumped all raw evidence without filtering, producing unsafe/incoherent output

## Root Causes

| # | Bug | Impact |
|---|-----|--------|
| 1 | **No relevance gate** — `_detect_contradictions` received ALL evidence chunks, including semantically irrelevant ones | False contradictions between unrelated domains |
| 2 | **Negation pair check too broad** — `supports/contradicts` triggered when ANY two shared words existed, regardless of topic coherence | False POSITIVE contradictions |
| 3 | **Numerical discrepancy check lacked coherence** — compared numbers across unrelated evidence | False contradictions for different-context values |
| 4 | **`_classify_conflict_type` triggered `DIFFERENT_TIMEFRAME` when both texts had NO years** — `both_have_years` was False, so same-entity + no-year + same-metric → `GENUINE_CONTRADICTION` | Incorrect conflict classification |
| 5 | **Synthesis fallback was unsafe** — dumped all raw evidence without filtering by relevance or conflict type | Incoherent/incomplete output |

## Fixes

### `_unit_normalize` bug (critical)
- **Root cause:** `_UNIT_NORMALIZE` had `"percent": "%"` and `"pct": "%"` but NOT the literal `"%"` symbol
- `_normalize_number_with_unit("35%")` → `(0.0, '')` → percentage values were silently discarded
- **Fix:** Added `"%": "%"` to `_UNIT_NORMALIZE`
- **Impact:** Contradictions between "35%" and "52%" are now correctly detected

### New functions
| Function | Purpose |
|----------|---------|
| `_compute_query_relevance()` | Deterministic lexical overlap (Jaccard + coverage) — scores 0.0–1.0 |
| `_is_topic_coherent()` | Requires ≥3 shared meaningful words between evidence chunks |
| `_filter_evidence_by_relevance()` | Filters evidence by query relevance threshold (default 0.15) |

### Modified functions
| Function | Change |
|----------|--------|
| `_detect_contradictions` (negation pair check) | Now requires topic coherence AND entity overlap |
| `_detect_contradictions` (numerical check) | Now requires topic coherence before comparing numbers |
| `_classify_conflict_type` | `DIFFERENT_TIMEFRAME` now requires `both_have_years` |
| `make_assess_node` | Wires relevance gate: filters evidence before contradiction detection |
| `make_synthesize_node` | Filters evidence by relevance; filters conflicts by query relevance; provides "insufficient evidence" message |

### Brain UI
- Evidence inspector shows **"Query Relevance"** pill (Relevant/Filtered)
- Conflict inspector shows type with color coding (red = GENUINE, yellow = others)

## Benchmark Results (8 cases, 100% pass)

| Case | Expected | Got | Status |
|------|----------|-----|--------|
| unrelated_evidence | 0 | 0 | PASS |
| shared_vocabulary_analytics | 0 | 0 | PASS |
| shared_vocabulary_database | 0 | 0 | PASS |
| genuine_numerical_conflict | 1 | 1 | PASS |
| different_years | 0 | 0 | PASS |
| different_scopes | 0 | 0 | PASS |
| genuine_opposing_claims | 1 | 1 | PASS |
| mixed_relevant_irrelevant | 0 | 0 | PASS |

**Contradiction precision: 8/8 (100%)**
**Relevance filtering: 8/8 (100%)**
**Overall: 16/16 (100%)**

## Test Suite
- 903 passed (up from ~880 baseline)
- 15 new regression tests in `tests/orchestration/test_phase43_correctness.py`
- 5 pre-existing failures (3 CSV disabled, 2 FastAPI route iteration)
- **Zero new failures**

## Files Modified
| File | Lines changed |
|------|---------------|
| `app/orchestration/nodes.py` | ~100 (unit normalize fix, relevance gate, contradiction safety, synthesis fallback) |
| `app/ui/brain/index.html` | ~20 (relevance pill, conflict type colors) |
| `tests/orchestration/test_phase43_correctness.py` | 320 (15 tests) |
| `benchmarks/phase43_benchmark.py` | 196 (8-case diagnostic) |
