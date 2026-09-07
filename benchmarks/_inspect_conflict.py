import json
plan = json.load(open('benchmarks/eval_data/eval_plan_v1.json'))
conflict = [q for q in plan['queries'] if q.get('conflict')]
print(f'{len(conflict)} conflict queries')
for q in conflict:
    print(f"{q['id']}: {q['query'][:80]}")
    gf = q.get('gold_facts', [])
    print(f"  Gold facts: {gf}")
    print(f"  Class: {q.get('class', '')}")
    print()
