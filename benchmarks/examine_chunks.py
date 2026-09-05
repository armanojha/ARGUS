"""Examine the actual missed chunks and corpus content."""
import json
from pathlib import Path

# Load forensics
with open('data/benchmark_reports/phase18_forensics.json') as f:
    data = json.load(f)

# Load corpus to see actual chunk texts
corpus_dir = Path("benchmarks/eval_data/corpus_v1")
chunks_by_doc = {}
for path in sorted(corpus_dir.glob("*.md")):
    doc_id = path.stem
    short_id = doc_id.split("-")[0] + "-" + doc_id.split("-")[1] if "-" in doc_id else doc_id
    text = path.read_text(encoding="utf-8")
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    current_chunk = ""
    doc_chunks = []
    for para in paragraphs:
        if len(current_chunk) + len(para) > 512 and current_chunk:
            doc_chunks.append(current_chunk)
            current_chunk = para
        else:
            current_chunk = (current_chunk + "\n\n" + para).strip()
    if current_chunk:
        doc_chunks.append(current_chunk)
    chunks_by_doc[short_id] = doc_chunks

# Build chunk_id map (same as benchmark)
from benchmarks.benchmark_fusion import build_benchmark_store
store, chunk_id_map = build_benchmark_store()

# Reverse map: chunk_id -> doc_id, chunk_index
chunk_info = {}
for doc_id, chunk_ids in chunk_id_map.items():
    for i, cid in enumerate(chunk_ids):
        chunk_info[cid] = {"doc_id": doc_id, "index": i}

# Analyze I1 failures
print("=" * 80)
print("I1: complex_research FAILURE - Detailed Analysis")
print("=" * 80)
for q in data:
    if q["query_id"] == "I1":
        print(f"Query: {q['query_text']}")
        print(f"Gold docs: {q['supporting_docs']}")
        print(f"Missed chunks: {q['misses']}")
        for miss_id in q['misses']:
            info = chunk_info.get(miss_id, {})
            doc_id = info.get("doc_id", "?")
            idx = info.get("index", "?")
            print(f"\n  Missed chunk {miss_id[:12]}... from {doc_id} index={idx}:")
            if doc_id in chunks_by_doc and isinstance(idx, int) and idx < len(chunks_by_doc[doc_id]):
                text = chunks_by_doc[doc_id][idx]
                print(f"    TEXT: {text[:300]}...")
            else:
                print(f"    (chunk not found in corpus)")
        break

print()
print("=" * 80)
print("E2: multi_hop FAILURE - Detailed Analysis")
print("=" * 80)
for q in data:
    if q["query_id"] == "E2":
        print(f"Query: {q['query_text']}")
        print(f"Gold docs: {q['supporting_docs']}")
        print(f"Missed chunks: {q['misses']}")
        for miss_id in q['misses']:
            info = chunk_info.get(miss_id, {})
            doc_id = info.get("doc_id", "?")
            idx = info.get("index", "?")
            print(f"\n  Missed chunk {miss_id[:12]}... from {doc_id} index={idx}:")
            if doc_id in chunks_by_doc and isinstance(idx, int) and idx < len(chunks_by_doc[doc_id]):
                text = chunks_by_doc[doc_id][idx]
                print(f"    TEXT: {text[:300]}...")
        break

print()
print("=" * 80)
print("E3: multi_hop PARTIAL - Detailed Analysis")
print("=" * 80)
for q in data:
    if q["query_id"] == "E3":
        print(f"Query: {q['query_text']}")
        print(f"Gold docs: {q['supporting_docs']}")
        print(f"Missed chunks: {q['misses']}")
        for miss_id in q['misses']:
            info = chunk_info.get(miss_id, {})
            doc_id = info.get("doc_id", "?")
            idx = info.get("index", "?")
            print(f"\n  Missed chunk {miss_id[:12]}... from {doc_id} index={idx}:")
            if doc_id in chunks_by_doc and isinstance(idx, int) and idx < len(chunks_by_doc[doc_id]):
                text = chunks_by_doc[doc_id][idx]
                print(f"    TEXT: {text[:300]}...")
        break

print()
print("=" * 80)
print("J4: adversarial PARTIAL - Detailed Analysis")
print("=" * 80)
for q in data:
    if q["query_id"] == "J4":
        print(f"Query: {q['query_text']}")
        print(f"Gold docs: {q['supporting_docs']}")
        print(f"Gold facts: {q['gold_facts']}")
        print(f"Missed chunks: {q['misses']}")
        for miss_id in q['misses']:
            info = chunk_info.get(miss_id, {})
            doc_id = info.get("doc_id", "?")
            idx = info.get("index", "?")
            print(f"\n  Missed chunk {miss_id[:12]}... from {doc_id} index={idx}:")
            if doc_id in chunks_by_doc and isinstance(idx, int) and idx < len(chunks_by_doc[doc_id]):
                text = chunks_by_doc[doc_id][idx]
                print(f"    TEXT: {text[:300]}...")
        break

print()
print("=" * 80)
print("J6: adversarial PARTIAL - Detailed Analysis")
print("=" * 80)
for q in data:
    if q["query_id"] == "J6":
        print(f"Query: {q['query_text']}")
        print(f"Gold docs: {q['supporting_docs']}")
        print(f"Gold facts: {q['gold_facts']}")
        print(f"Missed chunks: {q['misses']}")
        for miss_id in q['misses']:
            info = chunk_info.get(miss_id, {})
            doc_id = info.get("doc_id", "?")
            idx = info.get("index", "?")
            print(f"\n  Missed chunk {miss_id[:12]}... from {doc_id} index={idx}:")
            if doc_id in chunks_by_doc and isinstance(idx, int) and idx < len(chunks_by_doc[doc_id]):
                text = chunks_by_doc[doc_id][idx]
                print(f"    TEXT: {text[:300]}...")
        break
