"""Phase 16: Inspect individual failures to understand what's missing."""
import asyncio, json
from pathlib import Path
from benchmarks.benchmark_fusion import build_benchmark_store
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.router import RetrievalPolicyRouter
from app.retrieval.planner import EvidenceNeedPlanner

# Load queries
eval_path = Path("benchmarks/eval_data/eval_plan_v1.json")
with eval_path.open() as f:
    plan = json.load(f)
queries = {q["id"]: q for q in plan.get("queries", [])}

# Build store
store, chunk_id_map = build_benchmark_store()
retriever = HybridRetriever(store)
retriever.ensure_indexes()
router = RetrievalPolicyRouter()
planner = EvidenceNeedPlanner()

# Target failures
failures = ["F1", "F2", "F3", "I1", "I2", "I3", "E2", "E3", "J4", "J6"]

for qid in failures:
    q = queries[qid]
    query = q["query"]
    pattern = q["class"]
    gold_ids = set()
    for doc_id in q.get("supporting_docs", []):
        if doc_id in chunk_id_map:
            gold_ids.update(chunk_id_map[doc_id])

    # Get plan
    pat_enum = router.classify_question(query)
    plan_result = planner.plan(query, pat_enum)

    # Run planned retrieval
    refs = asyncio.run(router.execute_planned_retrieval(
        query, pat_enum, retriever, top_k=10
    ))
    retrieved_ids = {str(r.chunk_id) for r in refs}
    hits = retrieved_ids & gold_ids
    missing = gold_ids - retrieved_ids

    print(f"\n{'='*70}")
    print(f"FAILURE: {qid} | pattern={pattern} | pat_enum={pat_enum}")
    print(f"Query: {query[:80]}")
    print(f"Gold: {len(gold_ids)} | Retrieved: {len(refs)} | Hits: {len(hits)} | Missing: {len(missing)}")

    # Show evidence needs
    if plan_result.is_planned:
        print(f"\nEvidence Needs ({len(plan_result.needs)}):")
        for i, need in enumerate(plan_result.needs):
            print(f"  {i+1}. [{need.claim_type.value}] {need.search_query[:60]}")
            print(f"     Priority: {need.priority.value} | Requires opposing: {need.requires_opposing_evidence}")

    # Show missing chunks
    if missing:
        print(f"\nMissing chunks:")
        for mid in missing:
            # Find chunk in store
            chunks = store.get_chunks_by_ids([__import__('uuid').UUID(mid)])
            if chunks:
                c = chunks[0]
                text_preview = c.text[:100].replace('\n', ' ')
                print(f"  ID: {mid}")
                print(f"  Text: {text_preview}...")
                print(f"  Section: {c.section_path}")
                print(f"  Ordinal: {c.ordinal}")
                print(f"  Token count: {c.token_count}")
                print()

    # Show retrieved chunks with scores
    print(f"\nRetrieved chunks:")
    for r in refs[:5]:
        in_gold = "HIT" if str(r.chunk_id) in gold_ids else "miss"
        print(f"  #{r.rank} [{in_gold}] score={r.score:.3f} | {r.text[:60].replace(chr(10), ' ')}...")
