"""Phase 19 Ablation Results Analysis."""
import json
from pathlib import Path

ABLATIONS = ["full", "bm25_off", "vector_off", "no_planner", "no_multi_query", "no_recovery"]
LABELS = {
    "full": "FULL SYSTEM",
    "bm25_off": "- BM25",
    "vector_off": "- DENSE",
    "no_planner": "- PLANNER",
    "no_multi_query": "- MULTI-Q",
    "no_recovery": "- RECOVERY",
}

results = {}
for abl in ABLATIONS:
    path = Path(f"data/benchmark_reports/ablation_{abl}.json")
    if path.exists():
        with path.open() as f:
            data = json.load(f)
            results[abl] = data.get(abl, data)

# Summary table
print("=" * 90)
print("PHASE 19 ABLATION RESULTS")
print("=" * 90)
print(f"{'Configuration':<20} {'R@5':>6} {'R@10':>6} {'nDCG':>6} {'Avg ms':>8} {'P95 ms':>8}")
print("-" * 90)
for abl in ABLATIONS:
    if abl not in results:
        continue
    s = results[abl].get("summary", results[abl])
    label = LABELS.get(abl, abl)
    r5 = s.get("recall_at_5", s.get("recall_5", 0))
    r10 = s.get("recall_at_10", s.get("recall_10", 0))
    ndcg = s.get("ndcg_at_10", s.get("ndcg_10", 0))
    avg = s.get("avg_latency_ms", 0)
    p95 = s.get("p95_latency_ms", 0)
    print(f"{label:<20} {r5:>6.3f} {r10:>6.3f} {ndcg:>6.3f} {avg:>7.1f} {p95:>7.1f}")

# Delta table
print("\n" + "=" * 90)
print("DELTA FROM FULL SYSTEM")
print("=" * 90)
baseline = results.get("full", {}).get("summary", results.get("full", {}))
b_r5 = baseline.get("recall_at_5", baseline.get("recall_5", 0))
b_r10 = baseline.get("recall_at_10", baseline.get("recall_10", 0))
b_ndcg = baseline.get("ndcg_at_10", baseline.get("ndcg_10", 0))
b_avg = baseline.get("avg_latency_ms", 0)
b_p95 = baseline.get("p95_latency_ms", 0)

print(f"{'Configuration':<20} {'dR@5':>6} {'dR@10':>6} {'dnDCG':>6} {'dAvg':>8} {'dP95':>8} {'Verdict':>12}")
print("-" * 90)
for abl in ABLATIONS:
    if abl == "full" or abl not in results:
        continue
    s = results[abl].get("summary", results[abl])
    label = LABELS.get(abl, abl)
    dr5 = s.get("recall_at_5", s.get("recall_5", 0)) - b_r5
    dr10 = s.get("recall_at_10", s.get("recall_10", 0)) - b_r10
    dndcg = s.get("ndcg_at_10", s.get("ndcg_10", 0)) - b_ndcg
    davg = s.get("avg_latency_ms", 0) - b_avg
    dp95 = s.get("p95_latency_ms", 0) - b_p95

    # Classify value
    if dr10 < -0.03:
        verdict = "HIGH VALUE"
    elif dr10 < -0.01:
        verdict = "MED VALUE"
    elif davg < -30:
        verdict = "LOW COST"
    else:
        verdict = "LOW VALUE"
    print(f"{label:<20} {dr5:>+6.3f} {dr10:>+6.3f} {dndcg:>+6.3f} {davg:>+7.1f} {dp95:>+7.1f} {verdict:>12}")

# Per-pattern breakdown for key ablations
print("\n" + "=" * 90)
print("PER-PATTERN Recall@10 BY ABLATION")
print("=" * 90)
patterns = ["simple_lookup", "numerical", "technical_explanation", "normal_qa",
            "multi_doc_synthesis", "multi_hop", "complex_research", "conflict",
            "adversarial", "absent_info"]
header = f"{'Pattern':<25}"
for abl in ABLATIONS:
    header += f" {LABELS.get(abl, abl)[:8]:>8}"
print(header)
print("-" * 90)

for pat in patterns:
    row = f"{pat:<25}"
    for abl in ABLATIONS:
        if abl not in results:
            row += "      -"
            continue
        s = results[abl].get("summary", results[abl])
        pp = s.get("per_pattern", {})
        val = pp.get(pat, {}).get("recall_10", float("nan"))
        if isinstance(val, float) and not (val != val):  # not NaN
            row += f" {val:>8.3f}"
        else:
            row += "      -"
    print(row)

# Component value classification
print("\n" + "=" * 90)
print("COMPONENT VALUE CLASSIFICATION")
print("=" * 90)
print("""
HIGH VALUE / MODERATE COST:
  BM25:       R@10 drops 0.002 when removed. Actually near-zero impact on R@10!
              But R@5 improves 0.761→0.795 (vector-only is better for top-5).
              BM25 adds ~7ms avg. KEEP (helps some patterns, very low cost).

  DENSE:      R@10 drops 0.041 when removed. Critical for adversarial (0.944→0.806),
              absent_info (1.0→0.889), simple_lookup (1.0→0.944).
              Dense adds ~35ms avg via embedding generation. ESSENTIAL.

  PLANNER:    R@10 drops 0.058 when removed. Critical for conflict, complex_research,
              multi_hop (all go through planner). P95 drops 856→131ms (huge).
              Planner itself costs ~0.5ms. The cost is in multi-query execution.
              ESSENTIAL for complex patterns. Simple patterns bypass it.

  MULTI-Q:    R@10 drops 0.044 when removed. The multi-query retriever runs multiple
              sub-queries and does coverage-aware selection. Adds ~100ms avg.
              ESSENTIAL for complex patterns.

LOW VALUE / LOW COST:
  RECOVERY:   R@10 drops 0.014 when removed. Latency drops 73ms.
              Recovery activates only when coverage < 1.0.
              MARGINAL VALUE. Consider making selective.

  RANK_FUSION: Identical to baseline. Already using rank normalization.
              NO COST. NO IMPACT. Keep as-is.
""")
