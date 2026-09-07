#!/usr/bin/env python3
"""Phase 34: Hardware/Inference Audit.

Tests CPU vs GPU inference for the reranker model.
Measures latency, VRAM usage, and quality equivalence.
"""
from __future__ import annotations

import gc
import io
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from benchmarks.benchmark_fusion import build_benchmark_store


def main():
    print("=" * 90)
    print("Phase 34: Hardware/Inference Audit")
    print("=" * 90)

    # Check CUDA
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        gpu_name = torch.cuda.get_device_name(0) if cuda_available else "N/A"
        gpu_memory = torch.cuda.get_device_properties(0).total_mem / (1024**3) if cuda_available else 0
        print(f"\nCUDA available: {cuda_available}")
        print(f"GPU: {gpu_name}")
        print(f"GPU memory: {gpu_memory:.1f} GB")
    except Exception as e:
        print(f"\nCUDA check failed: {e}")
        cuda_available = False
        gpu_name = "N/A"
        gpu_memory = 0

    # Build store
    print("\n[1] Building store...")
    store, chunk_id_map = build_benchmark_store()
    from app.retrieval.hybrid import HybridRetriever
    retriever = HybridRetriever(store=store)
    retriever.ensure_indexes()

    # Prepare test data
    query = "Where is Acme Corporation headquartered?"
    candidates = retriever.search(query, top_k=20)
    pairs = [(query, c.text) for c in candidates]
    print(f"  Candidates: {len(candidates)}")

    # ── CPU Benchmark ────────────────────────────────────────────────────
    print("\n[2] CPU benchmark...")
    from sentence_transformers import CrossEncoder

    model_cpu = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu")

    # Warm up
    _ = model_cpu.predict(pairs[:5], batch_size=16, show_progress_bar=False)

    # Benchmark
    cpu_latencies = []
    for _ in range(5):
        t0 = time.perf_counter()
        scores_cpu = model_cpu.predict(pairs, batch_size=16, show_progress_bar=False)
        ms = (time.perf_counter() - t0) * 1000
        cpu_latencies.append(ms)

    cpu_avg = sum(cpu_latencies) / len(cpu_latencies)
    print(f"  CPU avg: {cpu_avg:.1f}ms (n=5)")
    print(f"  CPU scores: {list(scores_cpu[:3])}")

    # ── GPU Benchmark ────────────────────────────────────────────────────
    if cuda_available:
        print("\n[3] GPU benchmark...")
        try:
            import torch

            # Move model to GPU
            t0 = time.perf_counter()
            model_gpu = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device="cuda")
            gpu_load_ms = (time.perf_counter() - t0) * 1000
            print(f"  GPU model load: {gpu_load_ms:.1f}ms")

            # Warm up
            _ = model_gpu.predict(pairs[:5], batch_size=16, show_progress_bar=False)
            torch.cuda.synchronize()

            # Benchmark
            gpu_latencies = []
            for _ in range(5):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                scores_gpu = model_gpu.predict(pairs, batch_size=16, show_progress_bar=False)
                torch.cuda.synchronize()
                ms = (time.perf_counter() - t0) * 1000
                gpu_latencies.append(ms)

            gpu_avg = sum(gpu_latencies) / len(gpu_latencies)
            print(f"  GPU avg: {gpu_avg:.1f}ms (n=5)")
            print(f"  GPU scores: {list(scores_gpu[:3])}")

            # VRAM usage
            vram_used = torch.cuda.max_memory_allocated() / (1024**2)
            print(f"  VRAM used: {vram_used:.1f} MB")

            # Speedup
            speedup = cpu_avg / gpu_avg
            print(f"\n  Speedup: {speedup:.1f}x")
            print(f"  Latency reduction: {cpu_avg - gpu_avg:.1f}ms")

            # Quality check
            import numpy as np
            score_diff = np.max(np.abs(np.array(scores_cpu) - np.array(scores_gpu)))
            print(f"  Max score difference: {score_diff:.6f}")

            # Clean up
            del model_gpu
            gc.collect()
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"  GPU benchmark failed: {e}")
            gpu_avg = None
    else:
        print("\n[3] GPU benchmark: SKIPPED (CUDA not available)")
        gpu_avg = None

    # ── Batch size experiment ────────────────────────────────────────────
    print("\n[4] Batch size experiment (CPU)...")
    batch_sizes = [1, 4, 8, 16, 32]
    batch_results = {}

    for bs in batch_sizes:
        latencies = []
        for _ in range(3):
            t0 = time.perf_counter()
            _ = model_cpu.predict(pairs, batch_size=bs, show_progress_bar=False)
            ms = (time.perf_counter() - t0) * 1000
            latencies.append(ms)
        avg = sum(latencies) / len(latencies)
        per_pair = avg / len(pairs)
        batch_results[bs] = {"avg_ms": round(avg, 2), "per_pair_ms": round(per_pair, 2)}
        print(f"  batch_size={bs:>3}: {avg:>7.1f}ms total, {per_pair:.2f}ms/pair")

    # Save results
    results = {
        "cuda_available": cuda_available,
        "gpu_name": gpu_name,
        "gpu_memory_gb": round(gpu_memory, 1),
        "cpu_avg_ms": round(cpu_avg, 2),
        "gpu_avg_ms": round(gpu_avg, 2) if gpu_avg else None,
        "speedup": round(cpu_avg / gpu_avg, 1) if gpu_avg else None,
        "batch_sizes": batch_results,
        "candidate_count": len(candidates),
    }

    output_dir = Path("benchmarks/results")
    output_dir.mkdir(exist_ok=True)
    out_path = output_dir / "phase34_hardware_audit.json"
    with out_path.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"\nReport saved to {out_path}")

    # Summary
    print("\n" + "=" * 90)
    print("SUMMARY")
    print("=" * 90)
    print(f"  CPU latency: {cpu_avg:.1f}ms for {len(candidates)} candidates")
    if gpu_avg:
        print(f"  GPU latency: {gpu_avg:.1f}ms for {len(candidates)} candidates")
        print(f"  Speedup: {cpu_avg/gpu_avg:.1f}x")
        print(f"  Latency reduction: {cpu_avg - gpu_avg:.1f}ms")
    print(f"  Optimal batch size: {min(batch_results, key=lambda k: batch_results[k]['avg_ms'])}")

    print("\n" + "=" * 90)
    print("DONE")
    print("=" * 90)


if __name__ == "__main__":
    main()
