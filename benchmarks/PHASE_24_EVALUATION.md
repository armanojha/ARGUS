# Phase 24 Evaluation: Adaptive Research Orchestration

## 1. Executive Summary

Phase 24 implements a deterministic, zero-LLM research sufficiency assessment system intended to replace the LLM-based assess node for controlling how many retrieval iterations ARGUS runs before synthesizing an answer.

**Finding: Phase 24 provides no measurable benefit in its current form. The adaptive policy is not wired into the actual retrieval loop, and the benchmark that was supposed to compare fixed vs adaptive behavior was fundamentally flawed.**

Key evidence:
- The `benchmark_adaptive_research.py` runs identical single-pass retrieval for both conditions — the adaptive policy is applied as a post-hoc label, not as a loop controller
- When properly simulated with multi-iteration retrieval, 89.5% of queries need only 1 iteration regardless of policy
- The SynthesisGate has a 100% pass rate (too permissive)
- The sufficiency model misclassifies 47.4% of queries as INSUFFICIENT despite having strong evidence (R@10=0.85, coverage=1.0)
- The adaptive policy is NOT connected to `graph.py`, `nodes.py`, or any actual orchestration code

**Recommendation: OPTION A — Keep disabled.**

## 2. Phase 24 Architecture

### Actual Integration Point
Phase 24 components exist in `app/orchestration/adaptive_research.py` but are **not wired into the execution pipeline**. The only connection is:
- `app/config.py`: `adaptive_research_enabled = False` (feature flag)

The components are **not imported or called** by:
- `app/orchestration/graph.py` (state machine)
- `app/orchestration/nodes.py` (assess/retrieve/synthesize nodes)
- `app/orchestration/stopping.py` (stop condition checkers)

### What Phase 24 Actually Does
The components provide standalone diagnostic capability:
1. `ResearchSufficiency.assess()` — evaluates evidence quality deterministically
2. `MarginalGainCalculator.evaluate()` — detects diminishing returns
3. `PatternSpecificPolicies.get_policy()` — returns pattern-specific config
4. `SynthesisGate.check()` — evaluates pre-synthesis quality
5. `AdaptiveResearchPolicy.should_continue_retrieval()` — combines all above into a decision

None of these are called during actual query processing.

### Intended vs Actual Execution Flow

**Intended:**
```
Query → classify → plan → retrieve → [adaptive assessment] → retrieve/synthesize
```

**Actual (current):**
```
Query → classify → plan → retrieve → assess (LLM) → stop_check → synthesize
```

The adaptive policy is never consulted.

## 3. Baseline Behavior

From `baseline_orchestration.py` on 38 queries:

| Metric | Value |
|--------|-------|
| Pattern match rate | 7.9% (3/38) |
| Avg Recall@10 | 0.840 |
| Avg need coverage | 0.954 |
| Avg top-3 score | 0.929 |
| Avg source diversity | 6.7 |
| Queries with contradictions | 25/38 (66%) |
| Avg latency | 150ms |

### Per-Pattern Baseline

| Pattern | Count | R@10 | Coverage | Top-3 | Contradictions |
|---------|-------|------|----------|-------|----------------|
| simple_lookup | 6 | 1.000 | 1.000 | 0.940 | 14 |
| normal_qa | 4 | 0.938 | 1.000 | 0.922 | 17 |
| technical_explanation | 3 | 1.000 | 1.000 | 0.939 | 0 |
| multi_doc_synthesis | 3 | 0.933 | 1.000 | 0.938 | 7 |
| multi_hop | 3 | 0.822 | 0.583 | 0.944 | 7 |
| conflict | 3 | 0.500 | 1.000 | 0.884 | 19 |
| absent_info | 4 | 0.889 | 1.000 | 0.943 | 12 |
| numerical | 3 | 1.000 | 1.000 | 0.925 | 12 |
| complex_research | 3 | 0.633 | 0.833 | 0.935 | 11 |
| adversarial | 6 | 0.806 | 1.000 | 0.919 | 12 |

### Critical Observations
1. **Classification is severely broken**: Only 7.9% of queries match expected pattern. The `RetrievalPolicyRouter.classify_question()` is misclassifying almost everything.
2. **Retrieval quality is already high**: R@10=0.84, coverage=0.95, top-3=0.93. There is little room for improvement.
3. **Conflict queries have low R@10 (0.50)**: Gold chunks are not being retrieved for conflict queries.
4. **Multi-hop coverage is low (0.583)**: Evidence needs are not being fully satisfied.
5. **Contradictions are pervasive**: 66% of queries have text contradictions detected.

## 4. Fixed vs Adaptive Results

### Simulation Results (3-iteration max)

| Metric | Value |
|--------|-------|
| Avg iterations used | 1.13 |
| Avg evidence chunks | 8.3 |
| Avg Recall@5 | 0.752 |
| Avg Recall@10 | 0.848 |
| Avg need coverage | 1.000 |
| Avg top-3 score | 0.895 |
| Avg source diversity | 5.5 |
| Avg latency | 65ms |

### Iteration Distribution
- 1 iteration: 34 queries (89.5%)
- 2 iterations: 3 queries (7.9%)
- 3 iterations: 1 query (2.6%)

### Sufficiency Distribution
- INSUFFICIENT: 18 queries (47.4%)
- STRONG: 19 queries (50.0%)
- SUFFICIENT: 1 query (2.6%)

### Synthesis Gate
- Pass: 38 (100.0%)
- Block: 0 (0.0%)

### Gain Stop
- Stopped: 1 (2.6%)

## 5. Per-Pattern Analysis

### Simple Lookup (6 queries)
- **Iterations**: All 1
- **R@10**: 1.000 (perfect)
- **Sufficiency**: 4 INSUFFICIENT, 2 STRONG
- **Assessment**: Adaptive policy would NOT change behavior. Evidence is already perfect. The 4 INSUFFICIENT classifications are false negatives — the model has contradictions from retrieved chunks that are not actually relevant to the simple query.

### Normal QA (4 queries)
- **Iterations**: All 1
- **R@10**: 0.938
- **Sufficiency**: 3 INSUFFICIENT, 1 STRONG
- **Assessment**: No benefit from additional iterations. The INSUFFICIENT label is caused by contradiction detection in retrieved chunks.

### Technical Explanation (3 queries)
- **Iterations**: All 1
- **R@10**: 1.000 (perfect)
- **Sufficiency**: All STRONG
- **Assessment**: No room for improvement.

### Multi-Doc Synthesis (3 queries)
- **Iterations**: All 1
- **R@10**: 0.933
- **Sufficiency**: All STRONG
- **Assessment**: No room for improvement.

### Multi-Hop (3 queries)
- **Iterations**: 1, 2, 3 (avg 2.0)
- **R@10**: 0.822
- **Coverage**: 1.000 (improved from 0.583 in baseline)
- **Sufficiency**: 1 INSUFFICIENT, 2 STRONG
- **Assessment**: Additional iterations DO improve coverage (0.583 → 1.0). This is the only pattern where more research helps. However, the adaptive policy is not actually triggering these additional iterations — they happen because the simulation runs multiple passes.

### Conflict (3 queries)
- **Iterations**: All 1
- **R@10**: 0.500 (lowest of any pattern)
- **Sufficiency**: All INSUFFICIENT
- **Assessment**: Additional iterations would NOT help because the retrieval itself is failing to find gold chunks. The problem is retrieval quality, not iteration count.

### Absent Info (4 queries)
- **Iterations**: All 1
- **R@10**: 0.889
- **Sufficiency**: 3 INSUFFICIENT, 1 SUFFICIENT
- **Assessment**: No benefit from additional iterations. The information is genuinely absent.

### Numerical (3 queries)
- **Iterations**: All 1
- **R@10**: 1.000 (perfect)
- **Sufficiency**: 2 STRONG, 1 INSUFFICIENT
- **Assessment**: No room for improvement.

### Complex Research (3 queries)
- **Iterations**: 1, 2, 2 (avg 1.7)
- **R@10**: 0.633
- **Sufficiency**: All INSUFFICIENT
- **Assessment**: Additional iterations help slightly (R@10 improves from baseline). But the INSUFFICIENT classification suggests the model needs more evidence than the corpus provides.

### Adversarial (6 queries)
- **Iterations**: All 1
- **R@10**: 0.806
- **Sufficiency**: 1 INSUFFICIENT, 5 STRONG
- **Assessment**: No benefit from additional iterations.

## 6. Research-Round Analysis

### Marginal Gain by Iteration

| Iteration | Avg Gain | Min | Max | Count |
|-----------|----------|-----|-----|-------|
| 1 | 1.000 | 1.000 | 1.000 | 38 |
| 2 | 0.313 | 0.125 | 0.375 | 4 |
| 3+ | — | — | — | 1 |

**Analysis**: The first iteration always produces all evidence (gain=1.0). The second iteration produces 12.5-37.5% new evidence. With only 28 chunks in the corpus, diminishing returns are inevitable.

### Does Additional Research Help?
For the 4 queries that ran 2+ iterations:
- E2 (multi_hop): 11 chunks, R@10=0.67, coverage=1.0 — helped
- E3 (multi_hop): 11 chunks, R@10=0.80, coverage=1.0 — helped
- I1 (complex_research): 9 chunks, R@10=0.40, coverage=1.0 — marginal help
- I2 (complex_research): 11 chunks, R@10=0.62, coverage=1.0 — marginal help

**Conclusion**: Additional iterations help multi-hop and complex_research queries, but the gain is modest and the corpus is too small for this to be meaningful.

## 7. Synthesis Gate Analysis

The SynthesisGate with default parameters (min_evidence_count=1, min_coverage=0.0, min_sources=1) has a **100% pass rate**. It blocks nothing.

This means:
1. The gate is too permissive to be useful
2. Even queries with 0 recall pass the gate
3. The gate adds no value in its current configuration

To be useful, the gate would need much stricter thresholds (e.g., min_evidence_count=3, min_coverage=0.5, min_sources=2). But even then, the evidence quality is already high enough that the gate would rarely fire.

## 8. False Stop Analysis

The adaptive policy would "stop" (synthesize after 1 iteration) for 34/38 queries (89.5%). Of these:
- 19 are classified STRONG — correct to stop
- 14 are classified INSUFFICIENT — potentially wrong to stop
- 1 is classified SUFFICIENT — correct to stop

The 14 INSUFFICIENT queries that would stop prematurely:
- All have R@10 ≥ 0.50, coverage = 1.0, top-3 score ≥ 0.85
- The INSUFFICIENT label is caused by contradiction detection, not actual evidence gaps
- These queries already have sufficient evidence for synthesis

**Verdict**: No true false stops. The INSUFFICIENT classification is a false negative in the sufficiency model.

## 9. Unnecessary Research Analysis

The adaptive policy would force additional research for 4/38 queries (10.5%). Of these:
- E2 (multi_hop): 3 iterations, R@10=0.67 — beneficial
- E3 (multi_hop): 2 iterations, R@10=0.80 — beneficial
- I1 (complex_research): 2 iterations, R@10=0.40 — marginal benefit
- I2 (complex_research): 2 iterations, R@10=0.62 — marginal benefit

**Verdict**: The additional research is justified for multi-hop but marginal for complex_research. The latency cost is ~100ms per extra iteration.

## 10. Latency Analysis

| Condition | Avg Latency | P50 | P95 | Worst |
|-----------|-------------|-----|-----|-------|
| 1 iteration | 65ms | 58ms | 131ms | 149ms |
| 2 iterations | 112ms | 108ms | 116ms | 160ms |
| 3 iterations | 160ms | — | — | 160ms |

The latency cost of additional iterations is ~50-100ms per iteration. This is acceptable.

However, the baseline (existing pipeline) has avg latency of 150ms because it includes the full pipeline overhead (classification, planning, etc.). The simulation's 65ms only measures retrieval time.

## 11. Safety Analysis

### Feature Flag Behavior
- `adaptive_research_enabled = False` (default)
- When False, `AdaptiveResearchPolicy.should_continue_retrieval()` returns `action="continue_retrieval"` with reason "Adaptive research disabled"
- **The existing pipeline is completely unchanged when the flag is off**

### Loop Safety
- The adaptive policy respects `max_iterations` hard limit
- The adaptive policy respects `pending_subquestions` queue
- The MarginalGainCalculator requires `min_window=2` before stopping
- No infinite loop risk

### Regression Status
- 753 tests passed, 26 skipped, 0 failed
- Identical to pre-Phase 24 baseline
- No behavioral changes when feature flag is off

## 12. Failure Cases

### Critical: Benchmark Was Fundamentally Flawed
The original `benchmark_adaptive_research.py` ran identical single-pass retrieval for both baseline and adaptive, then applied the adaptive policy as a post-hoc label. The recall/coverage/latency numbers were identical for both conditions. This means the original Phase 24 "validation" measured nothing.

### Critical: Adaptive Policy Not Wired Into Pipeline
The `AdaptiveResearchPolicy` is never called during actual query processing. It exists as standalone diagnostic code. To actually change behavior, it would need to be integrated into `graph.py`'s assess node or `nodes.py`'s `make_assess_node()`.

### Sufficiency Model Misclassification
47.4% of queries are classified INSUFFICIENT despite having strong evidence. The model is overly influenced by contradiction detection — any contradiction downgrades the level, even when the contradictions are in irrelevant retrieved chunks.

### Classification Mismatch
Only 7.9% of queries match expected patterns. The `RetrievalPolicyRouter.classify_question()` is severely misclassifying queries, which means pattern-specific policies would be applied to the wrong queries.

## 13. Strengths

1. **Deterministic, zero-LLM assessment**: The sufficiency model is fast and predictable
2. **Well-structured components**: Clean separation of concerns (sufficiency, gain, patterns, gate)
3. **Feature-flagged**: No risk to existing behavior
4. **Good test coverage**: 29 tests, all passing
5. **Pattern-specific policies are well-reasoned**: Conflict/multi_hop requiring min 2 iterations makes sense

## 14. Weaknesses

1. **Not integrated into the pipeline**: The biggest issue — the code exists but is never called
2. **Benchmark was flawed**: The ablation measured nothing meaningful
3. **Sufficiency model is miscalibrated**: 47.4% false negative rate (INSUFFICIENT when evidence is strong)
4. **SynthesisGate is useless at default thresholds**: 100% pass rate
5. **Classification is broken**: Pattern-specific policies would be applied to wrong queries
6. **Corpus too small**: 28 chunks means diminishing returns after 1 iteration for most queries
7. **Contradiction detection is too sensitive**: Flags contradictions in irrelevant chunks

## 15. Final Recommendation

### OPTION A: Keep Adaptive Research Disabled

**Rationale:**
1. The adaptive policy is not wired into the pipeline — enabling it would do nothing
2. Even if wired in, 89.5% of queries need only 1 iteration — no benefit from additional research
3. The sufficiency model has a 47.4% false negative rate — it would make wrong decisions
4. The SynthesisGate is useless at default thresholds
5. The classification is broken — pattern-specific policies would target wrong queries
6. The existing pipeline already achieves R@10=0.84, coverage=0.95 — little room for improvement

### What Would Need to Change Before Activation

If Phase 24 were to be revisited:

1. **Wire the policy into the assess node** in `graph.py`/`nodes.py`
2. **Fix the sufficiency model** — contradiction detection should not downgrade evidence quality for irrelevant contradictions
3. **Tighten the SynthesisGate** — set meaningful thresholds
4. **Fix the classifier** — pattern-specific policies are useless if patterns are wrong
5. **Test on a larger corpus** — 28 chunks is too small to measure iteration benefits
6. **Build a proper benchmark** — one that actually runs the adaptive loop and measures before/after

## 16. Next Phase Recommendation

Phase 24 evaluation is complete. The adaptive research system is architecturally sound but practically inert. No Phase 25 work should depend on Phase 24 being active.

**The most impactful next step would be fixing the query classifier** (currently 7.9% accuracy), which affects retrieval quality across all phases, not just Phase 24.
