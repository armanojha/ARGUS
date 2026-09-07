import json
plan = json.load(open('benchmarks/eval_data/eval_plan_v1.json'))
docs = plan.get('documents', [])
print(f'{len(docs)} documents')
for d in docs:
    if isinstance(d, dict):
        title = d.get('title', '')[:60]
        print(f"  {d.get('doc_id', '')}: {title}")
    else:
        print(f"  {str(d)[:80]}")
