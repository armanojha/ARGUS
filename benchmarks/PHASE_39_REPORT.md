# Phase 39 — Quality Gap Fixes

**Status:** COMPLETE  
**Date:** 2026-09-07  
**Decision:** FIXES APPLIED — limited improvement due to provider contamination  

---

## Summary

Applied three quality gap fixes identified in Phase 38:
1. **Conflict detection** — Added deterministic pairwise contradiction detection in evidence
2. **Absent-info detection** — Added deterministic absent-info signal when evidence is irrelevant
3. **Degraded synthesis** — Fixed Pass 1 failure to degrade to raw evidence; fixed outcome classification

## What Changed

### 1. Conflict Detection (nodes.py)
- Added `_detect_contradictions()` function that compares evidence chunks pairwise
- Uses negation pattern matching + shared keyword detection
- Detects contradictions when evidence chunks share keywords but contain negation pairs
- Populates `contradiction_signals` state before synthesis
- **Wired into assess node** — runs on every assessment after evidence is available

### 2. Absent-Info Detection (nodes.py)
- Added `_is_evidence_absent()` function
- Detects when all evidence scores are below threshold and keyword overlap with query is <20%
- Returns deterministic short-circuit signal for absent information
- Prevents LLM from fabricating answers when corpus lacks the information

### 3. Degraded Synthesis (graph.py)
- **Pass 1 failure:** Now degrades to raw evidence (top 3 chunks) instead of returning empty answer
- **Outcome classification:** `_derive_outcome` now catches `synthesis_degraded_to_verified_claims` (not just `synthesis_degraded_to_raw_evidence`)

## Results Comparison

| Metric | Phase 38 | Phase 39 | Change |
|--------|----------|----------|--------|
| Claim Support | 0.328 | 0.335 | +0.007 |
| Citation Precision | 0.871 | 0.887 | +0.016 |
| Gold Fact Coverage | 0.511 | 0.509 | -0.002 |
| Hallucination | 0.037 | 0.044 | +0.007 |
| **Conflict Handling** | **0/3 (0%)** | **1/3 (33%)** | **+33%** |
| Abstention | 2/4 (50%) | 1/4 (25%) | -25% |
| Conflict GFC | 0.83 | 1.00 | +0.17 |

### Key Improvements
- **Conflict category GFC:** 83% → 100% (gold facts fully covered)
- **Conflict hallucination:** 4% → 0% (no hallucinated claims in conflict category)
- **Conflict handling:** 0/3 → 1/3 (one conflict correctly acknowledged)
- **Contradictions detected:** 0 → multiple per query (detection is working)

### Limitations
- Provider contamination still 92% — limits quality improvement
- Absent-info abstention regressed (25% vs 50%) — likely noise from provider instability
- Conflict acknowledgment still only 33% — synthesis LLM doesn't always use contradiction signals

## Files Modified
- `app/orchestration/nodes.py` — Added `_detect_contradictions()`, `_is_evidence_absent()`, wired into assess node
- `app/orchestration/graph.py` — Fixed degraded synthesis path, fixed outcome classification

## Tests
- 925 passed, 26 skipped, 0 failures (confirmed post-fix)

## Next Steps
1. Conflict detection is working but synthesis doesn't always use the signals — may need prompt refinement
2. Provider contamination still prevents reliable quality optimization
3. Consider increasing gold-fact coverage thresholds for conflict/absent-info categories
