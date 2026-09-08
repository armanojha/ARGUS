import re
from app.orchestration.nodes import _METRIC_KEYWORDS, _detect_contradictions, _is_topic_coherent, _classify_conflict_type, _extract_entity_keywords, _detect_temporal_context
from app.evidence.models import EvidenceRef, SourceType
from uuid import uuid4

def mk(t):
    return EvidenceRef(chunk_id=uuid4(), document_id=uuid4(), source_id=uuid4(), source_path='x', source_type=SourceType.TEXT, text=t, score=0.5, rank=1)

# Case 4: numerical conflict
e1 = mk("Enterprise revenue growth was 35% in 2024 according to industry survey of 500 companies.")
e2 = mk("Enterprise revenue growth reached 52% in 2024 based on annual technology report.")

text_i = e1.text.lower()
text_j = e2.text.lower()

words_i = set(re.findall(r"\b\w{4,}\b", text_i))
words_j = set(re.findall(r"\b\w{4,}\b", text_j))
shared_metrics = words_i & words_j & _METRIC_KEYWORDS
print("Shared metrics:", shared_metrics)

for metric in shared_metrics:
    for label, text in [("i", text_i), ("j", text_j)]:
        for m in re.finditer(r"\b" + re.escape(metric) + r"\b", text):
            start = max(0, m.start() - 50)
            end = min(len(text), m.end() + 50)
            window = text[start:end]
            nums = re.findall(r"\$?[\d,]+\.?\d*\s*(?:billion|million|thousand|%)?", window)
            print(f"  [{label}] metric={metric}, window={window!r}, nums={nums}")

# Check topic coherence
print("Topic coherent:", _is_topic_coherent(e1.text, e2.text, min_shared_significant=3))

# Check entities
entities_i = _extract_entity_keywords(e1.text)
entities_j = _extract_entity_keywords(e2.text)
print("Entities i:", entities_i)
print("Entities j:", entities_j)
print("Entity overlap:", entities_i & entities_j)

# Check temporal
ctx_i = _detect_temporal_context(e1.text)
ctx_j = _detect_temporal_context(e2.text)
print("Years i:", ctx_i["years"])
print("Years j:", ctx_j["years"])

# Run detection
c = _detect_contradictions([e1, e2], query="revenue growth rate")
print("Contradictions:", len(c))
for x in c:
    print(" ", x.get("conflict_type"), x.get("description", "")[:120])

# Case 7: opposing claims
print("\n=== Case 7 ===")
e3 = mk("The Acme Cloud Platform supports feature X for data processing and analytics workflows.")
e4 = mk("The Acme Cloud Platform does not support feature X in its current release version.")

c2 = _detect_contradictions([e3, e4], query="Acme Cloud Platform support feature X")
print("Raw contradictions:", len(c2))
for x in c2:
    print(" ", x.get("conflict_type"), x.get("description", "")[:120])
    print("  years_i:", x.get("timeframe_i"))
    print("  years_j:", x.get("timeframe_j"))
    print("  entity_overlap:", x.get("entity_overlap"))
