#!/usr/bin/env python3
"""Analyze Phase 29 ablation results."""
import json
import io
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

with open("benchmarks/results/phase29_ablation.json") as f:
    data = json.load(f)

# Compare per-query
print("Per-query comparison (claim support):")
print(f"{'Query':<15} {'Class':<22} {'Baseline':>8} {'TwoPass':>8} {'Delta':>8}")
print("-" * 65)

for i in range(21):
    a = data["A_baseline"][i]
    d = data["D_two_pass"][i]
    a_val = a.get("claim_support_rate", 0)
    d_val = d.get("claim_support_rate", 0)
    delta = d_val - a_val
    marker = "++" if delta > 0.2 else ("+" if delta > 0 else ("--" if delta < -20 else ""))
    print(f"{a['query_id']:<15} {a['class']:<22} {a_val:>7.0%} {d_val:>7.0%} {delta:>+7.0%} {marker}")

# Absent info
print("\nAbsent info answers:")
for i in [17, 18, 19, 20]:
    a = data["A_baseline"][i]
    d = data["D_two_pass"][i]
    print(f"\n{a['query_id']} [{a['class']}]:")
    print(f"  Baseline: {a['answer'][:200]}")
    print(f"  TwoPass:  {d['answer'][:200]}")

# Conflict answers
print("\nConflict answers:")
for i in [8, 9]:
    a = data["A_baseline"][i]
    d = data["D_two_pass"][i]
    print(f"\n{a['query_id']} [{a['class']}]:")
    print(f"  Baseline: {a['answer'][:200]}")
    print(f"  TwoPass:  {d['answer'][:200]}")

# Provider distribution
print("\nProvider distribution:")
for s in ["A_baseline", "D_two_pass"]:
    providers = {}
    for r in data[s]:
        p = r.get("provider", "unknown")
        providers[p] = providers.get(p, 0) + 1
    print(f"  {s}: {providers}")
