"""Phase 20 Benchmark: Entity Linking Impact Measurement (v2).

Uses the existing benchmark_retrieval_eval infrastructure for proper
retrieval quality comparison with and without entity linking.
"""
from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from benchmarks.benchmark_fusion import build_benchmark_store
from app.retrieval.entity_linking import EntityLinker, build_entity_index, CorpusEntityIndex
from app.retrieval.router import RetrievalPolicyRouter, get_retrieval_policy_router
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.planner import EvidenceNeedPlanner


def _load_corpus_chunks() -> list[dict]:
    """Load benchmark corpus as chunk dicts."""
    corpus_dir = _REPO / "benchmarks" / "eval_data" / "corpus_v1"
    chunks = []
    for doc_path in sorted(corpus_dir.glob("*.md")):
        doc_id = doc_path.stem
        text = doc_path.read_text(encoding="utf-8")
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        current_chunk = ""
        chunk_idx = 0
        for para in paragraphs:
            if len(current_chunk) + len(para) > 512 and current_chunk:
                chunks.append({
                    "text": current_chunk,
                    "chunk_id": f"{doc_id}_c{chunk_idx}",
                    "document_id": doc_id,
                })
                chunk_idx += 1
                current_chunk = para
            else:
                current_chunk = current_chunk + "\n\n" + para if current_chunk else para
        if current_chunk:
            chunks.append({
                "text": current_chunk,
                "chunk_id": f"{doc_id}_c{chunk_idx}",
                "document_id": doc_id,
            })
    return chunks


def run_targeted_test(
    entity_linking_enabled: bool = True,
) -> dict:
    """Run targeted test on I1, E2, E3 queries with entity linking."""
    print(f"\n{'='*60}")
    print(f"Targeted Test: entity_linking={'ON' if entity_linking_enabled else 'OFF'}")
    print(f"{'='*60}")

    # Build store
    store, chunk_id_map = build_benchmark_store()

    # Build entity index if enabled
    entity_linker = None
    if entity_linking_enabled:
        corpus_chunks = _load_corpus_chunks()
        entity_linker = EntityLinker(max_expansions=3, min_expansion_score=0.1)
        entity_linker.build_from_chunks(corpus_chunks)
        print(f"  Entities: {len(entity_linker.index.entities)}")
        print(f"  Relationships: {len(entity_linker.index.relationships)}")

    # Create router
    router = get_retrieval_policy_router()
    retriever = HybridRetriever(store=store)
    retriever.ensure_indexes()

    # Initialize planner and multi-query retriever
    router._planner = EvidenceNeedPlanner()
    from app.retrieval.multi_query import MultiQueryRetriever
    router._multi_query_retriever = MultiQueryRetriever(
        router=router, retriever=retriever, top_k=10,
    )

    # Target queries
    target_queries = [
        {
            "id": "I1",
            "query": "Assess the argument that Acme is diversifying away from industrial chemicals, and evaluate what it depends on.",
            "eval_class": "complex_research",
            "gold_facts": ["robot", "Delta Sync", "Atlas", "warehouse", "Memphis"],
            "gold_docs": ["doc-i", "doc-b"],
        },
        {
            "id": "E2",
            "query": "Which plant is downstream of Ohio and depends indirectly on PetroKem?",
            "eval_class": "multi_hop",
            "gold_facts": ["Monterrey", "Mexico"],
            "gold_docs": ["doc-c"],
        },
        {
            "id": "E3",
            "query": "The inspection robot integrates with which distribution center, and that integration depends on which Atlas feature?",
            "eval_class": "multi_hop",
            "gold_facts": ["Memphis", "Delta Sync"],
            "gold_docs": ["doc-i", "doc-b"],
        },
    ]

    results = {}

    for target in target_queries:
        query = target["query"]
        eval_class = target["eval_class"]
        gold_facts = set(target["gold_facts"])
        gold_docs = set(target["gold_docs"])

        print(f"\n  Testing {target['id']}: {query[:60]}...")

        # Classify
        pattern = router.classify_question(query)
        pattern_value = pattern.value if hasattr(pattern, 'value') else pattern

        # Expand with entity linking if enabled
        expanded_subqueries = []
        if entity_linking_enabled and entity_linker and entity_linker.index.entities:
            # Create plan
            plan = router._planner.plan(query, pattern_value)
            if plan.is_planned:
                # Expand sub-queries
                for need in plan.evidence_needs:
                    original = need.search_query
                    need.search_query = entity_linker.expand_subquery(need.search_query, pattern_value)
                    expanded_subqueries.append({
                        "original": original,
                        "expanded": need.search_query,
                    })
                print(f"    Expanded sub-queries:")
                for esq in expanded_subqueries:
                    print(f"      '{esq['original']}' -> '{esq['expanded']}'")

        # Run retrieval
        start = time.perf_counter()
        try:
            retrieved = asyncio.run(
                router.execute_planned_retrieval(query, pattern, retriever)
            )
        except Exception as e:
            print(f"    Error: {e}")
            retrieved = []
        elapsed_ms = (time.perf_counter() - start) * 1000

        # Analyze results
        retrieved_text = " ".join(r.text.lower() for r in retrieved)
        retrieved_docs = {str(r.document_id)[:6] for r in retrieved}
        retrieved_chunk_ids = [str(r.chunk_id)[:8] for r in retrieved]

        found_facts = {gf for gf in gold_facts if gf.lower() in retrieved_text}
        found_docs = gold_docs & retrieved_docs

        recall = len(found_facts) / max(1, len(gold_facts))

        print(f"    Pattern: {pattern_value}")
        print(f"    Retrieved: {len(retrieved)} chunks")
        print(f"    Found facts: {found_facts}")
        print(f"    Missing facts: {gold_facts - found_facts}")
        print(f"    Recall: {recall:.3f}")
        print(f"    Latency: {elapsed_ms:.1f}ms")

        results[target["id"]] = {
            "query": query,
            "pattern": pattern_value,
            "recall": recall,
            "found_facts": list(found_facts),
            "missing_facts": list(gold_facts - found_facts),
            "n_retrieved": len(retrieved),
            "latency_ms": round(elapsed_ms, 1),
            "expanded_subqueries": expanded_subqueries,
        }

    return results


if __name__ == "__main__":
    # Run without entity linking
    results_off = run_targeted_test(entity_linking_enabled=False)

    # Run with entity linking
    results_on = run_targeted_test(entity_linking_enabled=True)

    # Compare
    print(f"\n{'='*60}")
    print(f"COMPARISON: Entity Linking Impact on Target Cases")
    print(f"{'='*60}")
    print(f"{'Case':<5} {'No EL':>8} {'With EL':>8} {'Delta':>8} {'Facts Found'}")
    print(f"{'-'*60}")
    for case_id in ["I1", "E2", "E3"]:
        r_off = results_off[case_id]
        r_on = results_on[case_id]
        delta = r_on["recall"] - r_off["recall"]
        facts_on = ",".join(r_on["found_facts"]) if r_on["found_facts"] else "none"
        print(f"{case_id:<5} {r_off['recall']:>8.3f} {r_on['recall']:>8.3f} {delta:>+8.3f} {facts_on}")

    # Save results
    output = {
        "baseline": results_off,
        "entity_linking": results_on,
    }
    output_path = _REPO / "data" / "benchmark_reports" / "phase20_targeted.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"\nResults saved to {output_path}")
