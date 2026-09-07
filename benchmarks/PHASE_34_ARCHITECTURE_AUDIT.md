# Phase 34 Architecture Audit: Reranking Path

**Date:** 2026-09-07
**Status:** Diagnostic — no code changes

---

## 1. Current Reranker

| Property | Value |
|----------|-------|
| Model | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Library | `sentence_transformers.CrossEncoder` |
| Parameters | ~22M |
| Model size | ~80MB (disk) |
| Architecture | 6-layer MiniLM with cross-attention |
| Max sequence length | 512 tokens (default) |
| Device | **CPU** (no device config in code) |
| Batch size | 16 (`model.predict(pairs, batch_size=16)`) |
| Lazy loading | Yes — loaded on first `rerank()` call |
| Thread safety | Yes — `threading.Lock` on model init |
| Singleton | Yes — `Reranker._model` class variable |

## 2. Production Execution Path

```
retrieve_node()                          [nodes.py:212]
  │
  ├── policy_router.execute_retrieval()   [router.py:246]
  │     ├── dispatch BM25/VECTOR/HYBRID   [concurrent via asyncio.gather]
  │     ├── _fuse() per-method            [deterministic]
  │     ├── metadata filters
  │     ├── parent context expansion
  │     ├── source diversification
  │     └── reranker.rerank(query, fused, top_k=8)  [CROSS-ENCODER INFERENCE]
  │
  └── OR fallback path:
        ├── retriever.search_async()      [BM25 + Vector concurrent]
        └── reranker.rerank(query, results, top_k=8)
```

## 3. Input/Output

| Property | Value |
|----------|-------|
| Input candidates | Up to `orchestration_retrieval_top_k` (default 8) after policy router fusion |
| Output candidates | `top_k` (default 8, same as input) |
| Candidate text length | Up to ~512 tokens per chunk (model limit) |
| Pairs scored | `len(candidates)` query-passage pairs |

**Note:** The policy router limits candidates to `orchestration_retrieval_top_k` (8) BEFORE reranking, so the reranker processes at most 8 candidates in production.

The benchmark used `top_k=20` → `rerank_top_k=10`, which is different from production (`top_k=8` → `top_k=8`).

## 4. Where Reranking Happens

| Location | File:Line | When |
|----------|-----------|------|
| Policy router primary path | router.py:314-315 | Every retrieval with policy router enabled |
| Policy router fallback path | router.py:334-335 | When confidence fallback triggers |
| Retrieve node fallback | nodes.py:234, 240 | When policy router not enabled |
| Agent coordinator | coordinator.py:358 | Multi-agent debate reranking |
| API retrieval endpoint | retrieval.py:116-117 | Direct API calls with `use_reranker=True` |

## 5. Latency Breakdown (from Phase 33)

| Metric | Value |
|--------|-------|
| Avg rerank latency | 2957ms |
| P50 rerank latency | 1846ms |
| P95 rerank latency | 8230ms |
| First query (model load) | 14,325ms |
| Subsequent queries | 1,670-2,135ms |
| % of total retrieval | 86.7% |

## 6. Model Loading

- Lazy-loaded on first `rerank()` call via `_get_model()`
- `CrossEncoder(model_name)` downloads from HuggingFace Hub on first use
- Model cached in class variable `Reranker._model` — shared across all instances
- Subsequent queries reuse the loaded model

## 7. Inference Characteristics

- **Backend:** PyTorch (via sentence-transformers)
- **Device:** CPU (no `device` parameter passed to `CrossEncoder`)
- **Batch processing:** `model.predict(pairs, batch_size=16)` — processes all pairs in batches of 16
- **Tokenization:** Internal to CrossEncoder (SentencePiece/WordPiece)
- **No GPU transfer:** Model stays on CPU throughout

## 8. Key Observations

1. **Model is CPU-only** — The RTX 3050 6GB GPU is NOT being used for reranking
2. **Only 8 candidates** — Production reranks at most 8 candidates (not 20 as in benchmarks)
3. **Singleton model** — Model loaded once, shared across all queries
4. **Synchronous inference** — `model.predict()` is a blocking CPU call
5. **Thread pool offload** — Reranking runs in `asyncio.to_thread` from the async policy router, so it doesn't block the event loop
6. **First-query penalty** — 14.3s model load on cold start, then ~2s per query
7. **No device config** — No setting to control CPU/GPU device for reranking

## 9. Potential Optimization Paths

| Path | Description | Risk |
|------|-------------|------|
| GPU inference | Move model to RTX 3050 6GB | Low — model is 80MB, fits easily |
| Smaller model | Use a lighter cross-encoder | Medium — may reduce quality |
| Reduced candidates | Rerank fewer candidates | Low — but may reduce quality |
| Skip reranking | For high-confidence retrievals | Medium — needs calibration |
| ONNX optimization | Convert model to ONNX | Low — requires model conversion |
| Quantization | INT8/FP16 quantization | Low — may reduce quality |
