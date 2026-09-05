"""Quick analysis of failing queries."""
import json

with open('data/benchmark_reports/phase18_forensics.json') as f:
    data = json.load(f)

for target_id in ["I1", "E2", "E3", "J4", "J6"]:
    for q in data:
        if q["query_id"] == target_id:
            print(f"=== {q['query_id']}: {q['canonical_class']} ===")
            print(f"  Query: {q['query_text'][:100]}")
            print(f"  Gold docs: {q['supporting_docs']}")
            print(f"  Gold facts: {q['gold_facts']}")
            print(f"  R@5={q['recall_at_5']:.3f} R@10={q['recall_at_10']:.3f}")
            print(f"  Planner: {q['planner_activated']}, needs={len(q['evidence_needs'])}, subqueries={len(q['subqueries'])}")
            for i, need in enumerate(q['evidence_needs']):
                print(f"    Need {i}: claim={need['claim_type']}, topic={need['topic'][:50]}")
                print(f"           query={need['search_query'][:80]}")
            print(f"  Coverage: {q['verifier_coverage']}")
            print(f"  Hits: {len(q['hits'])}, Misses: {len(q['misses'])}")
            print(f"  Missed: {q['misses'][:5]}")
            for c in q['candidates'][:5]:
                print(f"    [{c['rank']}] score={c['score']:.3f} need={c['evidence_need_id'][:12]} text={c['text_preview'][:70]}")
            print()
            break
