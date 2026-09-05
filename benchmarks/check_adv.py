"""Check per-query results from latest benchmark report."""
import json

with open("data/benchmark_reports/retrieval_evaluation.json") as f:
    r = json.load(f)

print("complex_research:")
for q in r["per_query"]:
    if q["pattern"] == "complex_research":
        print(f'  {q["query_id"]:6s} R@5={q["recall_5"]:.2f} R@10={q["recall_10"]:.2f} nDCG={q["ndcg_10"]:.2f} lat={q["latency_ms"]:.0f}ms')

print("\nmulti_hop:")
for q in r["per_query"]:
    if q["pattern"] == "multi_hop":
        print(f'  {q["query_id"]:6s} R@5={q["recall_5"]:.2f} R@10={q["recall_10"]:.2f} nDCG={q["ndcg_10"]:.2f} lat={q["latency_ms"]:.0f}ms')

print("\nadversarial:")
for q in r["per_query"]:
    if q["pattern"] == "adversarial":
        print(f'  {q["query_id"]:6s} R@5={q["recall_5"]:.2f} R@10={q["recall_10"]:.2f} nDCG={q["ndcg_10"]:.2f} lat={q["latency_ms"]:.0f}ms')

print("\nAll patterns:")
from collections import defaultdict
by_pat = defaultdict(list)
for q in r["per_query"]:
    by_pat[q["pattern"]].append(q)
for pat in sorted(by_pat):
    qs = by_pat[pat]
    avg_r10 = sum(q["recall_10"] for q in qs) / len(qs)
    avg_lat = sum(q["latency_ms"] for q in qs) / len(qs)
    print(f"  {pat:25s} R@10={avg_r10:.3f} avg_lat={avg_lat:.0f}ms n={len(qs)}")
