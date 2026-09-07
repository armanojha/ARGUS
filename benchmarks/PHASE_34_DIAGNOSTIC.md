# Phase 34 Diagnostic

**Date:** 2026-09-07
**Status:** Diagnostic complete

---

## Executive Summary

The current reranker (`cross-encoder/ms-marco-MiniLM-L-6-v2`) adds **~414ms** per query (not ~2957ms as Phase 33 reported — that included model loading overhead). It provides **measurable quality improvements** (MRR +0.0455, nDCG +0.0544) but a faster alternative exists: `cross-encoder/ettin-reranker-17m-v1` is **1.86x faster on CPU** with better quality.

---

## D1: Current Reranker

| Property | Value |
|----------|-------|
| Model | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Parameters | 22M |
| Device | CPU (CUDA not available) |
| Latency | 414ms avg for 20 candidates |
| Batch size | 16 (production), optimal is 4 |

## D2: Reranking Impact

| Metric | NO RERANK | RERANK | Delta |
|--------|-----------|--------|-------|
| R@10 | 0.4545 | 0.4545 | +0.0000 |
| MRR | 0.4091 | 0.4545 | +0.0455 |
| nDCG@10 | 0.3925 | 0.4469 | +0.0544 |

**Top-1 changed in 64% of queries (7/11).**
**Avg 8.7 rank changes per query.**

## D3: Per-Pattern Impact

| Pattern | nDCG Delta | Impact |
|---------|------------|--------|
| multi_hop | +0.4359 | **Significant** |
| numerical | +0.1266 | **Moderate** |
| conflict | +0.0355 | Minor |
| simple_lookup | +0.0000 | None (already perfect) |
| technical_explanation | +0.0000 | None (already perfect) |
| absent_info | +0.0000 | None (no gold facts) |
| adversarial | +0.0000 | None (no gold facts) |

## D4: Candidate Count Impact

| Candidates | Latency | nDCG@10 |
|------------|---------|---------|
| 5 | 106ms | 0.4486 |
| 10 | 211ms | 0.4521 |
| 20 | 414ms | 0.4469 |
| 30 | 531ms | 0.4469 |
| 50 | 527ms | 0.4469 |

**Diminishing returns beyond 10 candidates.** 10 candidates gives best nDCG/latency ratio.

## D5: Hardware

- **CUDA: NOT available** — GPU inference not possible on this machine
- **CPU latency: 407ms** for 20 candidates
- **Optimal batch size: 4** (373ms vs 407ms with batch 16)

## D6: Alternative Models

| Model | Params | CPU pairs/sec | Speedup | Quality |
|-------|--------|---------------|---------|---------|
| **ettin-reranker-17m-v1** | **17M** | **267** | **1.86x** | **Better** |
| ms-marco-MiniLM-L4-v2 | 19M | 206 | 1.43x | Slightly lower |
| ms-marco-MiniLM-L6-v2 (current) | 22M | 144 | 1.00x | Baseline |
| ms-marco-MiniLM-L12-v2 | 33M | 76 | 0.53x | Slightly higher |
