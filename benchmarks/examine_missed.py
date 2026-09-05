"""Look up actual missed chunk text from the store."""
import json
import tempfile
from pathlib import Path
from benchmarks.benchmark_fusion import build_benchmark_store

# Build store (same as benchmark)
store, chunk_id_map = build_benchmark_store()

# Load forensics
with open('data/benchmark_reports/phase18_forensics.json') as f:
    data = json.load(f)

# Analyze each failing query
for target_id in ["I1", "E2", "E3", "J4", "J6"]:
    for q in data:
        if q["query_id"] == target_id:
            print(f"{'=' * 80}")
            print(f"{target_id}: {q['canonical_class']} - {q['recall_at_10']:.3f}")
            print(f"{'=' * 80}")
            print(f"Query: {q['query_text'][:100]}")
            print(f"Gold docs: {q['supporting_docs']}, Gold facts: {q['gold_facts']}")
            print(f"Missed chunks ({len(q['misses'])}):")
            for miss_id in q['misses']:
                # Look up the actual chunk from store
                import uuid
                try:
                    chunk_uuid = uuid.UUID(miss_id)
                    refs = store.get_evidence_refs([chunk_uuid], [0.0])
                    if refs:
                        ref = refs[0]
                        print(f"\n  CHUNK: {miss_id[:12]}...")
                        print(f"    doc_id: {ref.document_id}")
                        print(f"    text: {ref.text[:400]}")
                    else:
                        print(f"\n  CHUNK: {miss_id[:12]}... (not found)")
                except Exception as e:
                    print(f"\n  CHUNK: {miss_id[:12]}... (error: {e})")
            print()
            break
