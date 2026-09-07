# Phase 35 Architecture Audit: End-to-End Pipeline

**Date:** 2026-09-07
**Status:** Diagnostic — no code changes

---

## Complete Execution Path

```
POST /api/v1/query
  │
  ├── start_run_telemetry(call_ceiling=16)
  │
  ├── classify_complexity(query)          [deterministic, 0ms]
  │     └── FAST? -> fast_path=True (skip analyze+plan+assess)
  │
  ├── build_graph()
  ├── _initial_state()
  │
  ├── Graph Execution:
  │     │
  │     ├── [ANALYZE]                      [LLM CALL #1: query_analysis]
  │     │     └── Skip if fast_path
  │     │
  │     ├── [PLAN]                         [LLM CALL #2: research_planning]
  │     │     └── Skip if fast_path
  │     │
  │     ├── [MEMORY_ENHANCE]               [optional, non-fatal]
  │     │
  │     ├── [RETRIEVE]  ← LOOP START
  │     │     ├── classify_question()       [deterministic]
  │     │     ├── execute_retrieval()
  │     │     │     ├── BM25 search         [~1ms]
  │     │     │     ├── Vector search       [~54ms]
  │     │     │     ├── Fusion              [~397ms]
  │     │     │     └── Rerank              [~420ms]
  │     │     └── _merge_evidence()
  │     │
  │     ├── [ASSESS]
  │     │     ├── Budget check              [deterministic, skip LLM]
  │     │     ├── Adaptive research check   [deterministic, skip LLM]
  │     │     ├── evidence_selector.select() [~0.85ms]
  │     │     └── LLM CALL #3: evidence_extraction
  │     │
  │     ├── [STOP_CHECK]                    [deterministic, 0ms]
  │     │     └── 5-condition evaluator
  │     │
  │     └── Loop back to RETRIEVE if not sufficient
  │
  │     ├── [SYNTHESIZE]
  │     │     ├── evidence_selector.select() [~0.85ms]
  │     │     └── LLM CALL #4: synthesis (tier=strong)
  │     │
  │     └── check_claim_grounding()         [deterministic]
  │
  ├── _build_result()                       [deterministic]
  │
  ├── _run_selective_verification()          [if enabled]
  │     └── LLM CALL #5: verification
  │
  └── end_run_telemetry()
```

## Per-Node External Call Matrix

| Node | LLM | Embedding | Retrieval | Reranker | EvidenceSelector | Short-Circuit |
|------|-----|-----------|-----------|----------|-----------------|---------------|
| classify_complexity | - | - | - | - | - | - |
| analyze | YES (query_analysis) | - | - | - | - | fast-path skip |
| plan | YES (research_planning) | - | - | - | - | fast-path skip |
| retrieve | - | YES | YES | YES | - | - |
| assess | YES (evidence_extraction) | - | - | - | YES | budget/adaptive skip |
| stop_check | - | - | - | - | - | no-op if None |
| synthesize | YES (synthesis) | - | - | - | YES | no-evidence skip |
| verification | YES (verification) | - | - | - | - | gate check |

## LLM Call Budget

| Path | Calls | Typical |
|------|-------|---------|
| Fast-path (simple query) | 1-2 | synthesis only + optional verification |
| Normal (1 iteration) | 3-4 | analyze + plan + assess + synthesis |
| Complex (2+ iterations) | 4-6 | analyze + plan + 2x(assess + retrieve) + synthesis |
| With verification | +1 | verification (post-synthesis) |
| Hard ceiling | 16 | Never exceeds |

## Key Cost Drivers (from previous phases)

| Component | Latency | Quality Impact |
|-----------|---------|----------------|
| Reranking | ~420ms | MRR +0.045, nDCG +0.054 |
| Fusion | ~397ms | N/A (deterministic) |
| Embedding | ~54ms | N/A (required for vector search) |
| BM25 | ~1ms | N/A (required for lexical search) |
| Evidence selector | ~0.85ms | N/A (minimal overhead) |
| LLM calls | ~1-5s each | Primary quality driver |

## Existing Optimizations

1. **Fast-path gate**: Skips analyze+plan+assess for simple queries (saves 2-3 LLM calls)
2. **Concurrent BM25+Vector**: Runs via `asyncio.to_thread`
3. **Adaptive stopping**: 5-condition evaluator prevents unnecessary iterations
4. **Adaptive research policy**: Deterministic short-circuits before LLM assess call
5. **Evidence-aware tier downgrade**: Uses cheaper models when evidence is strong
6. **Budget clamping**: Hard limits on iterations and tokens
7. **Selective verification**: Skips for simple/low-risk queries
8. **Gap detector**: Targets missing evidence instead of blind re-retrieval

## Frozen Components

- Phase 30 verified synthesis strategy
- Phase 34 reranker (cross-encoder/ms-marco-MiniLM-L-6-v2)
- Phase 33 caching decisions
- Evidence selector configuration
- Adaptive research policy behavior
