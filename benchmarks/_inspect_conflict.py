import json
plan = json.load(open('benchmarks/eval_data/eval_plan_v1.json'))
conflict = [q for q in plan['queries'] if q.get('category') == 'conflict']
for q in conflict:
    print(f"{q['id']}: {q['question']}")
    gf = q.get('gold_facts', [])
    print(f"  Gold facts: {gf}")
    print(f"  Expected: {q.get('expected_behavior', '')}")
    print()
