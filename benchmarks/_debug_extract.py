"""Debug extraction patterns."""
import sys
sys.path.insert(0, ".")
from benchmarks.benchmark_fusion import build_benchmark_store
from app.retrieval.hybrid import HybridRetriever
from app.verification.deterministic import TextContradictionDetector

store, chunk_id_map = build_benchmark_store()
retriever = HybridRetriever(store=store)
retriever.ensure_indexes()
detector = TextContradictionDetector()

results = retriever.search("revenue employees utilization", top_k=28)
for r in results:
    claims = detector._extract_claims(r)
    if claims:
        print(f"doc={str(r.document_id)[:8]} score={r.score:.3f}")
        for c in claims:
            print(f"  cat={c['category']} val={c['value']} unit={c['unit']} text={c['text'][:60]}")
