# Phase 33 Diagnostic

**Date:** 2026-09-07
**Status:** Diagnostic complete — minimal optimization justified

---

## Executive Summary

ARGUS is **already efficient** in most areas. The only measurable caching issue is `search_async()` bypassing the result cache, but its real-world impact is minimal because:
1. Agentic loops rarely repeat the same subquery across iterations
2. Evidence selector overhead is negligible (0.85ms avg)
3. No duplicate embeddings in single-query mode
4. The actual bottleneck is reranking (86.7% of retrieval time), which is model inference — not a caching issue

---

## D1: Embedding Cache

**Status:** EFFECTIVE — no change needed

- Cache: 64-entry LRU, keyed on exact query text
- Hit rate: 100% for repeated queries
- Duplicate embeddings: 0/11 in single-query mode
- Latency saving: ~50ms per cached embedding

## D2: Retrieval Result Cache

**Status:** PARTIALLY EFFECTIVE — minor fix available

- Cache: 128-entry LRU, keyed on `query:top_k:weights:norm`
- **Sync `search()`:** Cache works correctly (5378x speedup on hit)
- **Async `search_async()`:** BYPASSES cache entirely (production path)
- Real-world impact: LOW — agentic loops rarely repeat the same subquery

## D3: Query-Planning Cache

**Status:** NOT NEEDED

- Planning is deterministic and fast (<1ms)
- Plans are computed once per query and stored in state
- No repeated computation

## D4: Claim-Generation Cache

**Status:** NOT NEEDED — HIGH RISK

- Claim generation is called once per request
- Caching would require invalidation on evidence changes
- Risk of stale claims outweighs benefit

## D5: Synthesis Cache

**Status:** NOT NEEDED — HIGH RISK

- Synthesis is called once per request
- Caching would risk stale answers
- Risk of incorrect answers outweighs benefit

## D6: Duplicate LLM Calls

**Status:** NO DUPLICATES DETECTED

- Each LLM call has a unique purpose (analysis, planning, assessment, synthesis)
- No exact-match duplicate calls observed
- Provider fallback creates different-model calls, not duplicates

## D7: Concurrency

**Status:** EFFECTIVE — no change needed

- BM25 + Vector searches run concurrently via `asyncio.to_thread`
- Policy router dispatches methods concurrently via `asyncio.gather`
- No sequential bottlenecks in retrieval path

## D8: Evidence Reuse

**Status:** EFFECTIVE — no change needed

- Evidence pool is accumulated across iterations via `_merge_evidence()`
- Dedup by chunk_id prevents redundant evidence
- Evidence selector runs on accumulated pool

## D9: Cache Invalidation

**Status:** CORRECT — no change needed

- `mark_dirty()` clears all caches on index rebuild
- FAISS/BM25 indexes rebuilt only when dirty
- No stale data risk

## D10: Memory Interaction

**Status:** NO INTERACTION — no change needed

- Memory system is independent of caching
- Memory queries are infrequent and fast

## D11: Actual Cache Hit/Miss Rates

| Cache | Hits | Misses | Hit Rate |
|-------|------|--------|----------|
| Embedding cache | 11 | 0 | 100% |
| Retrieval result cache (sync) | 1 | 10 | 9% |
| Retrieval result cache (async) | 0 | 0 | N/A (bypassed) |
| Evidence selector | N/A | 11 | N/A (no cache) |

## D12: End-to-End Latency Breakdown

| Stage | Avg (ms) | % of Total |
|-------|----------|------------|
| Reranking | 2957 | 86.7% |
| Fusion | 397 | 11.6% |
| Embedding | 54 | 1.6% |
| BM25 | 1 | <0.1% |
| Vector | 0.5 | <0.1% |
| EvidenceSelector | 0.85 | <0.1% |
| **Total retrieval** | **3410** | **100%** |

**Dominant stage: Reranking (86.7%)** — This is CrossEncoder model inference, not a caching issue.

---

## Production Decision

**OPTION A — NO CHANGE JUSTIFIED**

Rationale:
1. The only measurable issue (`search_async()` cache bypass) has minimal real-world impact
2. Agentic loops rarely repeat the same subquery across iterations
3. Evidence selector overhead is negligible (0.85ms avg)
4. The actual bottleneck (reranking) is model inference, not caching
5. Adding cache complexity to `search_async()` would increase code surface area for minimal benefit

If future evidence shows agentic loops frequently repeat subqueries, the fix is simple:
add cache check/store to `search_async()` (3-line change). But the current data does not justify it.
