# Phase 25 Report: Synthesis Quality Gate

## Summary

Phase 25 added a deterministic post-synthesis quality gate to ARGUS. The synthesis pipeline — the single point of failure for answer quality — previously had zero quality checks. This phase introduces four improvements:

1. **Contradiction-aware synthesis prompts** — when evidence contradicts itself, the synthesis prompt now includes a "CONTRADICTION ALERT" section instructing the LLM to acknowledge conflicts, present both sides, and avoid presenting contradictory claims as settled fact.

2. **Evidence quality annotations** — retrieval scores are now visible to the synthesis LLM, giving it a signal about which evidence is more topically relevant.

3. **Deterministic claim grounding check** — every sentence in the synthesized answer is checked for bracket citations. Unsupported sentences produce `unsupported_claim` warnings in the result metadata.

4. **Citation fabrication warning** — when the LLM produces no citations, the fallback mechanism (which auto-attaches top-3 evidence) now adds a `citation_fallback` warning instead of silently fabricating provenance.

---

## Files Modified

| File | Change |
|------|--------|
| `app/orchestration/prompts.py` | `build_synthesis_messages()` accepts `contradiction_signals` parameter; `_format_evidence_block()` accepts `include_scores` parameter |
| `app/orchestration/nodes.py` | `make_synthesize_node()` wires contradiction signals to prompt and runs `check_claim_grounding()` post-synthesis; new `check_claim_grounding()` function |
| `app/orchestration/graph.py` | `_build_result()` adds `citation_fallback` warning when no citations found |

## Files Created

| File | Purpose |
|------|---------|
| `tests/orchestration/test_synthesis_quality.py` | 20 unit tests for all Phase 25 features |
| `benchmarks/benchmark_synthesis_quality.py` | Deterministic benchmark (no LLM required) |
| `benchmarks/PHASE_25_DIAGNOSTIC.md` | Architecture audit and target selection |
| `benchmarks/PHASE_25_REPORT.md` | This file |

---

## Test Results

### New Tests (20)

| Test Class | Tests | Status |
|------------|-------|--------|
| `TestClaimGrounding` | 8 | All pass |
| `TestContradictionAwarePrompts` | 3 | All pass |
| `TestEvidenceQualityAnnotations` | 4 | All pass |
| `TestCitationFallbackWarning` | 2 | All pass |
| `TestSynthesizeNodePhase25` | 3 | All pass |

### Full Suite

| Metric | Before | After | Delta |
|--------|--------|-------|-------|
| Passed | 838 | 858 | +20 |
| Skipped | 26 | 26 | 0 |
| Failed | 0 | 0 | 0 |

---

## Benchmark Results

### Grounding Check Performance

| Test Case | Warnings | Expected | Latency (ms) |
|-----------|----------|----------|-------------|
| All sentences cited | 0 | 0 | 0.007 |
| Half cited | 2 | 2 | 0.007 |
| None cited | 2 | 2 | 0.007 |
| Fullwidth citations | 0 | 0 | 0.005 |
| Empty answer | 0 | 0 | 0.000 |
| Long mixed answer | 2 | 2 | 0.019 |

**Average grounding latency: 0.007ms** (negligible overhead)

### Contradiction Prompt Overhead

- Baseline prompt generation: 0.004ms
- With contradiction signals: 0.006ms
- **Overhead: 0.002ms** (negligible)

### Evidence Annotation Overhead

- Baseline (no scores): 0.003ms
- With scores: 0.004ms
- **Overhead: 0.002ms** (negligible)

---

## What Changed for the User

### Before Phase 25
- Synthesis prompt had no awareness of evidence contradictions
- Evidence quality (retrieval scores) was invisible to the synthesis LLM
- No check verified that answer sentences were cited
- Citation fallback silently fabricated provenance

### After Phase 25
- Contradictions are surfaced in the synthesis prompt with explicit instructions
- Evidence scores appear in the evidence block (e.g., `[1] (score: 0.87, source: ...)`)
- Every uncited sentence produces a warning in `result.warnings`
- Citation fallback produces a `citation_fallback` warning

### Example Warnings

```
unsupported_claim: This claim has no citation attached to it.
citation_fallback: LLM produced no bracket citations; top evidence auto-attached as citations
```

---

## What Did NOT Change

- The synthesis LLM call itself (same model, same temperature, same tier)
- The evidence selector (still picks minimal high-coverage subset)
- The verification engine (still post-hoc, still optional)
- The benchmark metrics (still token F1, claim support rate, etc.)
- The adaptive research policy
- The retrieval pipeline
- The planner
- The router

---

## Risk Assessment

| Risk | Status | Evidence |
|------|--------|----------|
| Regression in existing tests | Mitigated | 858 passed, 0 failed |
| LLM call cost increase | None | No new LLM calls |
| Synthesis prompt bloat | Minimal | Contradiction section only when signals exist |
| Overly aggressive grounding check | Mitigated | Uses same 50% lexical threshold as existing `claim_support_rate` |
| Breaking citation fallback | Mitigated | Fallback preserved, warning added |

---

## Recommendations

### Production Deployment
Phase 25 is **safe for production deployment**. All changes are:
- Additive (new warnings, new prompt sections)
- Non-breaking (existing behavior preserved)
- Zero-cost (no new LLM calls)
- Fully tested (20 new tests, 858 total passing)

### Future Work (Not Phase 25)
- LLM-as-judge evaluation for answer quality (requires live LLM benchmark)
- Semantic claim-evidence alignment (beyond lexical overlap)
- Verification engine integration (currently off by default)
