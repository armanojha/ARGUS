"""Debug contradiction detection."""
import sys
sys.path.insert(0, ".")
from benchmarks.benchmark_fusion import build_benchmark_store
from app.retrieval.hybrid import HybridRetriever
from app.verification.deterministic import TextContradictionDetector

store, chunk_id_map = build_benchmark_store()
retriever = HybridRetriever(store=store)
retriever.ensure_indexes()
detector = TextContradictionDetector()

for q in ["What was Acme annual revenue", "How many employees does Acme have", "Ohio plant utilization"]:
    results = retriever.search(q, top_k=20)
    contras = detector.detect_contradictions(results, min_severity=0.3)
    print(f"Q: {q} -> {len(contras)} contradictions")
    for c in contras:
        print(f"  {c.description}")
        print(f"    chunk_a={c.chunk_a_id[:8]} doc_a={c.source_a[:8]}")
        print(f"    chunk_b={c.chunk_b_id[:8]} doc_b={c.source_b[:8]}")
    print()
