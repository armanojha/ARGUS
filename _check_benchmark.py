import json

with open('benchmarks/data/questions_v1.json', 'r') as f:
    raw = json.load(f)

q = raw['items'][0]
gold_answer = q['gold_answer']
gold_evidence = q['gold_evidence']
print(f"Raw gold_answer: {gold_answer}")
print(f"Raw gold_evidence: {gold_evidence}")

from app.evaluation.answer_quality import evaluate_answer, decompose_claims, _build_cited_answer
from dataclasses import dataclass

@dataclass
class FakeCitation:
    text: str
    ref_id: int = 1

answer = _build_cited_answer(gold_evidence)
print(f"Built answer: {repr(answer)}")
print(f"Decomposed claims: {decompose_claims(answer)}")

citations = [FakeCitation(text=e, ref_id=i+1) for i, e in enumerate(gold_evidence)]
result = evaluate_answer(answer, citations, gold_facts=gold_evidence)
print(f"Claims count: {result.total_claims}")
print(f"Citation presence: {result.citation_presence_rate}")
print(f"Claim support: {result.claim_support_rate}")
for c in result.claims:
    print(f"  Claim: {repr(c.claim_text)}")
    print(f"    Status: {c.support_status}")
    print(f"    Citation IDs: {c.citation_ids}")
    print(f"    Resolved texts: {c.resolved_evidence_texts}")
