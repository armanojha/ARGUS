# Phase 33 Architecture Audit

**Date:** 2026-09-07
**Status:** Diagnostic — no code changes

---

## 1. Production Execution Path

```
POST /api/v1/query
  │
  ▼
start_run_telemetry()                          [api/orchestration.py:53]
  │
  ▼
run_query(query)                               [orchestration/graph.py:595]
  │
  ├── get_settings(), get_router(),             [singletons, @lru_cache]
  │   get_hybrid_retriever(), get_reranker()
  │
  ├── retriever.ensure_indexes()               [retrieval/hybrid.py:48]
  │     ├── assign_bm25_doc_ids()              [SQL, if dirty]
  │     ├── assign_embedding_indices()         [SQL, if dirty]
  │     ├── bm25.build_index()                 [if dirty]
  │     ├── embedder.embed_chunks()            [EMBEDDING, if dirty]
  │     └── vector.build_index()               [FAISS, if dirty]
  │
  ├── classify_complexity(query)               [llm_gateway/routing/complexity.py:88]
  │     ├── FAST → fast_path=True, skip to retrieve
  │     └── BALANCED/STRONG → fast_path=False
  │
  ├── [fast_path=False] analyze                [orchestration/nodes.py:113, LLM CALL #1]
  │     └── router.complete(QueryAnalysis)
  │
  ├── [fast_path=False] plan                   [orchestration/nodes.py:138, LLM CALL #2]
  │     └── router.complete(ResearchPlan)
  │
  ├── retrieve (LOOP)                          [orchestration/nodes.py:206]
  │     ├── policy_router.classify_question()  [deterministic]
  │     ├── policy_router.execute_retrieval()  [retrieval/router.py:246]
  │     │     ├── BM25 → bm25.search()         [no embedding]
  │     │     ├── VECTOR → embed_query()       [EMBEDDING CALL]
  │     │     │     + faiss.search()
  │     │     └── HYBRID → both concurrent
  │     ├── reranker.rerank()                  [CrossEncoder, local]
  │     └── _merge_evidence()                  [dedup by chunk_id]
  │
  ├── assess (if not fast_path)                [orchestration/nodes.py:281, LLM CALL #3]
  │     ├── evidence_selector.select()         [PASS 1: O(n²) semantic dedup]
  │     ├── router.complete(EvidenceAssessment) [LLM]
  │     └── queue next subquery or sufficient
  │
  ├── stop_check                               [orchestration/stopping.py, deterministic]
  │
  ├── [loop back to retrieve if not sufficient]
  │
  ├── synthesize                               [orchestration/nodes.py:496, LLM CALL #4]
  │     ├── evidence_selector.select()         [PASS 2: O(n²) semantic dedup AGAIN]
  │     └── router.complete(synthesis)         [LLM]
  │
  └── END
```

## 2. Every Expensive Operation

| Operation | File:Line | Type | Cost |
|-----------|-----------|------|------|
| Query analysis | nodes.py:113 | LLM call (structured) | ~1-3s |
| Research planning | nodes.py:138 | LLM call (structured) | ~1-3s |
| BM25 search | bm25.py:101 | CPU (tokenize + score) | ~1-10ms |
| Embedding generation | embeddings.py:68 | CPU (SentenceTransformer) | ~50-200ms |
| FAISS search | vector.py:98 | CPU (inner product) | ~1-5ms |
| Evidence selector | evidence_selector.py:102 | CPU (O(n²) cosine sim) | ~5-50ms |
| Cross-encoder rerank | reranker/reranker.py:45 | CPU (CrossEncoder) | ~10-50ms |
| Evidence assessment | nodes.py:336 | LLM call (structured) | ~1-3s |
| Claim generation | two_pass_synthesis.py:194 | LLM call (structured) | ~2-5s |
| Final rendering | two_pass_synthesis.py:297 | LLM call (free-form) | ~2-5s |

## 3. Every LLM Call

| Call | When | Structured? | Tier |
|------|------|-------------|------|
| Query analysis | Non-fast-path only | Yes | Auto |
| Research planning | Non-fast-path only | Yes | Auto |
| Evidence assessment | Per assess iteration | Yes | Auto (may downgrade) |
| Synthesis (single-pass) | Always | No | "strong" |
| Pass 1 claim gen | Two-pass enabled | Yes | "strong" |
| Pass 2 final render | Two-pass enabled | No | "strong" |
| Verification | Verification enabled | Yes | "strong" |

## 4. Every Embedding Call

| Call | File:Line | Scope | Cached? |
|------|-----------|-------|---------|
| Query embedding for vector search | hybrid.py:194 | Per retrieval iteration | Yes (LRU 64) |
| Chunk embedding during index build | hybrid.py:71 | Only when dirty | No (batch) |
| Evidence selector semantic dedup | evidence_selector.py:193 | Per select() call | No (uses pre-computed) |

## 5. Every Vector Search

| Call | File:Line | Scope | Cached? |
|------|-----------|-------|---------|
| FAISS search for query | vector.py:98 | Per retrieval iteration | No |
| Evidence selector embedding lookup | evidence_selector.py:193 | Per select() call | No (reads from FAISS) |

## 6. Every BM25 Search

| Call | File:Line | Scope | Cached? |
|------|-----------|-------|---------|
| BM25 search for query | bm25.py:101 | Per retrieval iteration | No |

## 7. Every Repeated Query / Evidence Transformation

| Duplicate | Occurrences | Impact |
|-----------|-------------|--------|
| `evidence_selector.select()` on same evidence | 2-3x per query | O(n²) semantic dedup each time |
| `search_async()` bypasses result cache | Every retrieval iteration | Redundant BM25 + vector computation |
| `build_assessment_messages()` / `build_synthesis_messages()` | Once each | Trivial (string formatting) |

## 8. Existing Caching Mechanisms

| Cache | Location | Key | Size | Effective? |
|-------|----------|-----|------|------------|
| Settings singleton | config.py:604 | None | 1 | Yes |
| LLM router singleton | llm_gateway/__init__.py:17 | None | 1 | Yes |
| HybridRetriever result cache | hybrid.py:40 | query:top_k:weights:norm | 128 | **NO — bypassed by search_async()** |
| Embedding query cache | embeddings.py:39 | exact query text | 64 | Yes |
| BM25 index (disk) | data/indexes/bm25.pkl | N/A | All chunks | Yes |
| FAISS index (disk) | data/indexes/faiss.index | N/A | All chunks | Yes |
| OCR result cache | ingestion/ocr.py:71 | SHA-256 of settings | 2000 | Yes |
| CrossEncoder model | reranker/reranker.py | None | 1 | Yes |
| SentenceTransformer model | embeddings.py | None | 1 | Yes |

## 9. Existing In-Memory State

| State | Scope | Lifetime |
|-------|-------|----------|
| OrchestrationState | Per request | Request lifecycle |
| Evidence pool | Per request | Accumulated across iterations |
| Subquery queue | Per request | Request lifecycle |
| Gain history | Per request | Request lifecycle |

## 10. Persistent Memory

| Component | Storage | Location |
|-----------|---------|----------|
| Evidence store | SQLite (WAL) | data/evidence.db |
| Evidence graph | NetworkX pickle | data/graph/evidence_graph.pkl |
| Telemetry runs | JSONL | data/telemetry/runs.jsonl |
| Memory system | SQLite | data/memory/memory.db |

## 11. Async/Concurrent Execution

| Operation | Concurrency | Method |
|-----------|-------------|--------|
| BM25 + Vector search | Concurrent | `asyncio.to_thread` × 2 |
| Policy router method dispatch | Concurrent | `asyncio.gather` with semaphore |
| LLM calls | Sequential (within graph) | `await router.complete()` |
| Evidence selector | Sequential | Synchronous CPU |
| Reranker | Sequential (within thread) | `asyncio.to_thread` |

## 12. Key Findings

### Critical: `search_async()` Result Cache Bypass
- **Location:** `hybrid.py:153-215`
- **Problem:** The production retrieval path calls `search_async()` which never checks or populates `_result_cache`
- **Impact:** Every retrieval iteration recomputes BM25 + vector search from scratch, even for identical queries
- **Evidence:** Sync `search()` has cache logic (lines 108-149); async `search_async()` has none
- **Fix:** Add cache check/store to `search_async()` — mirror the sync path

### Moderate: Evidence Selector Redundancy
- **Location:** `nodes.py:334`, `nodes.py:518`, `graph.py:267`
- **Problem:** `evidence_selector.select()` called 2-3 times per query on potentially identical evidence
- **Impact:** O(n²) semantic dedup runs multiple times; moderate CPU cost
- **Fix:** Memoize with evidence-list fingerprint as key

### Minor: No LLM Response Caching
- Every LLM call hits the provider. Expected behavior for non-deterministic outputs.
- Not recommended for optimization (stale evidence risk, low exact-match hit rate).

## 13. Existing Non-Caching Optimizations

| Mechanism | Location | Impact |
|-----------|----------|--------|
| Fast-path gate | complexity.py:88 | Skips analyze+plan+assess for simple queries |
| Evidence-aware tier downgrade | nodes.py:362 | Uses cheaper models when evidence is strong |
| Adaptive stopping | stopping.py:290 | Stops early when evidence sufficient |
| Adaptive gap detection | seeking.py:55 | Targets missing evidence |
| Question pattern routing | router.py:159 | Selects optimal retrieval methods |
| Parent-context expansion | router.py:302 | Expands long-report evidence |
| Source diversification | router.py:307 | Ensures multi-source coverage |
| Budget clamping | nodes.py:169 | Hard limits on iterations/tokens |
