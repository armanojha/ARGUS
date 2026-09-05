"""Comprehensive diagnostic trace for I1 (complex_research) query."""
from __future__ import annotations

import asyncio
import json
import os
import sys
import uuid
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.retrieval.bm25 import BM25Retriever
from app.retrieval.vector import FAISSVectorStore
from app.retrieval.embeddings import EmbeddingGenerator
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.planner import EvidenceNeedPlanner
from app.retrieval.multi_query import MultiQueryRetriever
from app.retrieval.router import RetrievalPolicyRouter


def build_store():
    from benchmarks.benchmark_fusion import build_benchmark_store
    return build_benchmark_store()


def get_chunk_text(store, chunk_id_str):
    try:
        cid = uuid.UUID(chunk_id_str)
        refs = store.get_evidence_refs([cid], [0.0])
        if refs:
            return refs[0].text
    except Exception:
        pass
    return None


def get_chunk_source(store, chunk_id_str):
    try:
        cid = uuid.UUID(chunk_id_str)
        refs = store.get_evidence_refs([cid], [0.0])
        if refs:
            return refs[0].source_path, refs[0].document_id
    except Exception:
        pass
    return None, None


def main():
    I1_QUERY = "Assess the argument that Acme is diversifying into robotics and evaluate what it depends on."
    I1_GOLD_FACTS = ["robot", "Delta Sync", "Atlas", "warehouse", "Memphis"]
    I1_SUPPORTING_DOCS = ["doc-i", "doc-b"]

    print("=" * 100)
    print("I1 COMPREHENSIVE RETRIEVAL DIAGNOSTIC")
    print("=" * 100)
    print(f"Query: {I1_QUERY}")
    print(f"Gold facts: {I1_GOLD_FACTS}")
    print(f"Supporting docs: {I1_SUPPORTING_DOCS}")

    # Stage 0: Build store
    print("\n" + "=" * 80)
    print("STAGE 0: BUILD STORE & IDENTIFY GOLD CHUNKS")
    print("=" * 80)

    store, chunk_id_map = build_store()
    total_chunks = sum(len(v) for v in chunk_id_map.values())
    print(f"Total chunks in corpus: {total_chunks}")
    print(f"Documents: {list(chunk_id_map.keys())}")

    gold_ids = set()
    for doc_id in I1_SUPPORTING_DOCS:
        if doc_id in chunk_id_map:
            gold_ids.update(chunk_id_map[doc_id])
    print(f"\nGold chunk IDs ({len(gold_ids)} chunks):")
    for cid in sorted(gold_ids):
        text = get_chunk_text(store, cid)
        source_path, doc_id = get_chunk_source(store, cid)
        preview = text[:120].replace("\n", " ") if text else "NOT FOUND"
        print(f"  {cid}  source={source_path}  doc={doc_id}")
        print(f"    TEXT: {preview}...")

    # Map gold facts to chunks
    print("\nGold fact -> chunk mapping:")
    for fact in I1_GOLD_FACTS:
        found_in = []
        for cid in gold_ids:
            text = get_chunk_text(store, cid)
            if text and fact.lower() in text.lower():
                found_in.append(cid)
        print(f"  '{fact}' -> {len(found_in)} chunks: {found_in}")

    # Stage 1: Classification
    print("\n" + "=" * 80)
    print("STAGE 1: QUESTION CLASSIFICATION")
    print("=" * 80)

    router = RetrievalPolicyRouter()
    pattern = router.classify_question(I1_QUERY)
    print(f"Classified pattern: {pattern.value}")
    print(f"Is plannable: {pattern.value in EvidenceNeedPlanner.PLANNABLE_PATTERNS}")

    mix = router.get_retrieval_mix(pattern)
    print(f"Retrieval mix methods: {[m.value for m in mix.methods]}")
    print(f"BM25 weight: {mix.bm25_weight}, Vector weight: {mix.vector_weight}")

    # Stage 2: Planner
    print("\n" + "=" * 80)
    print("STAGE 2: EVIDENCE NEED PLANNER DECOMPOSITION")
    print("=" * 80)

    planner = EvidenceNeedPlanner()
    plan = planner.plan(I1_QUERY, pattern.value)
    print(f"Plan is_planned: {plan.is_planned}")
    print(f"Evidence needs: {plan.need_count}")

    for i, need in enumerate(plan.evidence_needs):
        print(f"\n  Need [{i}]:")
        print(f"    topic: {need.topic}")
        print(f"    entities: {need.entities}")
        print(f"    claim_type: {need.claim_type.value}")
        print(f"    search_query: {need.search_query}")
        print(f"    priority: {need.priority.value}")

    if plan.search_variants:
        print("\nSearch variants:")
        for i, sv in enumerate(plan.search_variants):
            print(f"  [{i}] {sv}")

    # Stage 3: Build indexes
    print("\n" + "=" * 80)
    print("STAGE 3: BUILD RETRIEVAL INDEXES")
    print("=" * 80)

    bm25 = BM25Retriever(store)
    vector = FAISSVectorStore(store)
    embedder = EmbeddingGenerator()
    retriever = HybridRetriever(store=store, bm25=bm25, vector=vector, embedder=embedder)
    retriever.ensure_indexes()
    print("BM25 + FAISS indexes built")

    # Stage 4: Per-subquery analysis
    print("\n" + "=" * 80)
    print("STAGE 4: PER-SUBQUERY RETRIEVAL ANALYSIS")
    print("=" * 80)

    all_queries = [I1_QUERY] + [n.search_query for n in plan.evidence_needs if n.search_query != I1_QUERY]

    for qi, query in enumerate(all_queries):
        print(f"\n--- QUERY [{qi}]: {query}")

        # BM25
        bm25_results = bm25.search(query, top_k=30)
        bm25_scores = {str(cid): score for cid, score in bm25_results}

        print(f"  BM25 results ({len(bm25_results)} hits):")
        bm25_rank = 0
        for cid, score in bm25_results:
            bm25_rank += 1
            cid_str = str(cid)
            is_gold = "GOLD" if cid_str in gold_ids else "     "
            text = get_chunk_text(store, cid_str)
            preview = text[:80].replace("\n", " ") if text else "?"
            print(f"    #{bm25_rank:2d} {is_gold} score={score:.4f} {cid_str[:12]}  {preview}")

        # Vector
        query_emb = embedder.embed_texts([query])[0]
        vector_results = vector.search(query_emb, top_k=30)
        vector_scores = {str(cid): score for cid, score in vector_results}

        print(f"  Vector results ({len(vector_results)} hits):")
        vec_rank = 0
        for cid, score in vector_results:
            vec_rank += 1
            cid_str = str(cid)
            is_gold = "GOLD" if cid_str in gold_ids else "     "
            text = get_chunk_text(store, cid_str)
            preview = text[:80].replace("\n", " ") if text else "?"
            print(f"    #{vec_rank:2d} {is_gold} score={score:.4f} {cid_str[:12]}  {preview}")

        # Gold chunk coverage for this query
        print(f"  Gold chunk coverage:")
        for gid in sorted(gold_ids):
            bm25_rank = list(bm25_scores.keys()).index(gid) + 1 if gid in bm25_scores else None
            vec_rank = list(vector_scores.keys()).index(gid) + 1 if gid in vector_scores else None
            bm25_s = bm25_scores.get(gid, 0)
            vec_s = vector_scores.get(gid, 0)
            text = get_chunk_text(store, gid)
            preview = text[:60].replace("\n", " ") if text else "?"
            if bm25_rank or vec_rank:
                print(f"    {gid[:12]} BM25: rank={bm25_rank} score={bm25_s:.4f} | Vec: rank={vec_rank} score={vec_s:.4f}")
                print(f"               text: {preview}")

    # Stage 5: Hybrid retrieval
    print("\n" + "=" * 80)
    print("STAGE 5: HYBRID RETRIEVAL (FULL PIPELINE)")
    print("=" * 80)

    hybrid_results = retriever.search(I1_QUERY, top_k=20)
    print(f"\nHybrid results for original query ({len(hybrid_results)} hits):")
    for i, ref in enumerate(hybrid_results):
        cid_str = str(ref.chunk_id)
        is_gold = "GOLD" if cid_str in gold_ids else "     "
        bm25_s = ref.metadata.get("bm25_score", 0)
        vec_s = ref.metadata.get("vector_score", 0)
        print(f"  #{i+1:2d} {is_gold} fused={ref.score:.4f} bm25={bm25_s:.4f} vec={vec_s:.4f} {cid_str[:12]}  source={ref.source_path}")
        print(f"      text: {ref.text[:80].replace(chr(10), ' ')}")

    # Stage 6: Multi-query retrieval
    print("\n" + "=" * 80)
    print("STAGE 6: MULTI-QUERY RETRIEVAL (PLANNED PIPELINE)")
    print("=" * 80)

    multi_query = MultiQueryRetriever(router=router, retriever=retriever, top_k=10)
    result = asyncio.run(multi_query.retrieve(plan))

    print(f"\nMulti-query result:")
    print(f"  Total candidates: {result.total_candidates}")
    print(f"  Selected: {len(result.selected)}")
    print(f"  Source diversity: {result.source_diversity}")
    print(f"  Recovery activated: {result.recovery_activated}")
    print(f"  Recovery type: {result.recovery_type.value}")
    print(f"  Coverage before recovery: {result.coverage_before_recovery:.3f}")
    print(f"  Coverage after recovery: {result.coverage_after_recovery:.3f}")

    print(f"\n  Need coverage:")
    for need in plan.evidence_needs:
        cov = result.need_coverage.get(need.id, 0)
        print(f"    {need.id}: {cov:.2f} - {need.topic[:40]}")

    print(f"\n  Selected candidates ({len(result.selected)}):")
    for i, ref in enumerate(result.selected):
        cid_str = str(ref.chunk_id)
        is_gold = "GOLD" if cid_str in gold_ids else "     "
        need_id = ref.metadata.get("evidence_need_id", "")
        multi_need = ref.metadata.get("multi_need_hit", False)
        print(f"  #{i+1:2d} {is_gold} fused={ref.score:.4f} {cid_str[:12]}  need={need_id[:8]} multi={multi_need}")
        print(f"      source={ref.source_path}  text: {ref.text[:80].replace(chr(10), ' ')}")

    # Stage 7: Gold chunk hit/miss analysis
    print("\n" + "=" * 80)
    print("STAGE 7: GOLD CHUNK HIT/MISS ANALYSIS")
    print("=" * 80)

    retrieved_ids = [str(r.chunk_id) for r in result.selected]
    hits = set(retrieved_ids[:10]) & gold_ids
    misses = gold_ids - set(retrieved_ids[:10])

    print(f"\nHits ({len(hits)}/5):")
    for cid in sorted(hits):
        text = get_chunk_text(store, cid)
        source_path, doc_id = get_chunk_source(store, cid)
        print(f"  {cid}  source={source_path}  doc={doc_id}")
        print(f"    TEXT: {text[:200].replace(chr(10), ' ') if text else 'NOT FOUND'}")

    print(f"\nMisses ({len(misses)}/5):")
    for cid in sorted(misses):
        text = get_chunk_text(store, cid)
        source_path, doc_id = get_chunk_source(store, cid)

        bm25_results_full = bm25.search(I1_QUERY, top_k=50)
        bm25_full_scores = {str(c): s for c, s in bm25_results_full}
        bm25_rank = list(bm25_full_scores.keys()).index(cid) + 1 if cid in bm25_full_scores else None

        query_emb = embedder.embed_texts([I1_QUERY])[0]
        vector_results_full = vector.search(query_emb, top_k=50)
        vector_full_scores = {str(c): s for c, s in vector_results_full}
        vec_rank = list(vector_full_scores.keys()).index(cid) + 1 if cid in vector_full_scores else None

        print(f"  {cid}  source={source_path}  doc={doc_id}")
        print(f"    TEXT: {text[:200].replace(chr(10), ' ') if text else 'NOT FOUND'}")
        print(f"    BM25 rank: {bm25_rank} (score: {bm25_full_scores.get(cid, 0):.4f})")
        print(f"    Vector rank: {vec_rank} (score: {vector_full_scores.get(cid, 0):.4f})")

        # Check which subqueries find this chunk
        found_by = []
        for qi, q in enumerate(all_queries):
            q_bm25 = bm25.search(q, top_k=50)
            q_bm25_scores = {str(c): s for c, s in q_bm25}
            if cid in q_bm25_scores:
                found_by.append(f"BM25[{qi}]")
            q_emb = embedder.embed_texts([q])[0]
            q_vec = vector.search(q_emb, top_k=50)
            q_vec_scores = {str(c): s for c, s in q_vec}
            if cid in q_vec_scores:
                found_by.append(f"Vec[{qi}]")
        print(f"    Found by subqueries: {found_by if found_by else 'NONE'}")

        # Check parent/neighbor chunks
        if doc_id:
            doc_chunks = store.get_chunks_by_document(doc_id)
            if doc_chunks:
                ordinal = None
                for chunk in doc_chunks:
                    if str(chunk.id) == cid:
                        ordinal = chunk.ordinal
                        break
                if ordinal is not None:
                    neighbors = [c for c in doc_chunks if abs(c.ordinal - ordinal) <= 1 and c.id != uuid.UUID(cid)]
                    if neighbors:
                        print(f"    Neighbor chunks ({len(neighbors)}):")
                        for nc in neighbors:
                            nc_text = nc.text[:80].replace("\n", " ")
                            print(f"      ordinal={nc.ordinal} {nc.id}: {nc_text}")

    # Stage 8: Cross-document bridge analysis
    print("\n" + "=" * 80)
    print("STAGE 8: CROSS-DOCUMENT BRIDGE ANALYSIS")
    print("=" * 80)

    doc_i_chunks = chunk_id_map.get("doc-i", [])
    print(f"\ndoc-i chunks ({len(doc_i_chunks)}):")
    for cid in doc_i_chunks:
        text = get_chunk_text(store, cid)
        has_atlas = "atlas" in text.lower() if text else False
        has_delta = "delta sync" in text.lower() if text else False
        has_robot = "robot" in text.lower() if text else False
        print(f"  {cid}: atlas={has_atlas} delta_sync={has_delta} robot={has_robot}")
        print(f"    {text[:150].replace(chr(10), ' ') if text else 'NOT FOUND'}")

    doc_b_chunks = chunk_id_map.get("doc-b", [])
    print(f"\ndoc-b chunks ({len(doc_b_chunks)}):")
    for cid in doc_b_chunks:
        text = get_chunk_text(store, cid)
        has_robotics = "robot" in text.lower() if text else False
        has_acme = "acme" in text.lower() if text else False
        has_warehouse = "warehouse" in text.lower() if text else False
        has_memphis = "memphis" in text.lower() if text else False
        print(f"  {cid}: robot={has_robotics} acme={has_acme} warehouse={has_warehouse} memphis={has_memphis}")
        print(f"    {text[:150].replace(chr(10), ' ') if text else 'NOT FOUND'}")

    # Stage 9: Root cause
    print("\n" + "=" * 80)
    print("STAGE 9: ROOT CAUSE ANALYSIS")
    print("=" * 80)

    recall = len(hits) / len(gold_ids) if gold_ids else 0
    print(f"\nFinal recall@10: {recall:.3f} ({len(hits)}/{len(gold_ids)})")

    print(f"\nGold fact coverage by chunk:")
    for gid in sorted(gold_ids):
        text = get_chunk_text(store, gid)
        if text:
            found_facts = [f for f in I1_GOLD_FACTS if f.lower() in text.lower()]
            print(f"  {gid}: facts={found_facts}")

    print(f"\nMISSED CHUNKS (no gold facts found in them):")
    for gid in sorted(gold_ids):
        text = get_chunk_text(store, gid)
        if text:
            found_facts = [f for f in I1_GOLD_FACTS if f.lower() in text.lower()]
            if not found_facts:
                print(f"  {gid} - {text[:80].replace(chr(10), ' ')}")

    # Save
    report = {
        "query": I1_QUERY,
        "gold_facts": I1_GOLD_FACTS,
        "supporting_docs": I1_SUPPORTING_DOCS,
        "pattern": pattern.value,
        "plan_is_planned": plan.is_planned,
        "evidence_needs": [
            {"id": n.id, "topic": n.topic, "search_query": n.search_query}
            for n in plan.evidence_needs
        ],
        "gold_chunks": {gid: {
            "text_preview": (get_chunk_text(store, gid) or "")[:200],
            "contains_facts": [f for f in I1_GOLD_FACTS if f.lower() in (get_chunk_text(store, gid) or "").lower()],
        } for gid in sorted(gold_ids)},
        "hits": list(hits),
        "misses": list(misses),
        "recall": recall,
        "selected_chunks": [
            {"chunk_id": str(r.chunk_id), "score": r.score, "source": r.source_path}
            for r in result.selected
        ],
    }
    report_path = Path("data/benchmark_reports/i1_diagnostic.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nReport saved to {report_path}")


if __name__ == "__main__":
    main()
