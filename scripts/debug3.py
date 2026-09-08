"""Debug why Case 4 numerical conflict isn't detected."""
import re
from app.orchestration.nodes import (
    _METRIC_KEYWORDS, _is_topic_coherent, _extract_entity_keywords,
    _detect_temporal_context, _normalize_number_with_unit, _normalize_value,
    _detect_contradictions,
)
from app.evidence.models import EvidenceRef, SourceType
from uuid import uuid4

def mk(t):
    return EvidenceRef(chunk_id=uuid4(), document_id=uuid4(), source_id=uuid4(),
                       source_path='benchmark', source_type=SourceType.TEXT,
                       text=t, score=0.5, rank=1)

# Case 4 - SHORTER (benchmark)
e1s = mk("Enterprise revenue growth was 35% in 2024 according to industry survey.")
e2s = mk("Enterprise revenue growth reached 52% in 2024 based on annual report.")
c_short = _detect_contradictions([e1s, e2s], query="enterprise revenue growth rate in 2024")
print(f"SHORT: {len(c_short)} contradictions")
for x in c_short:
    print(f"  {x.get('conflict_type')}: {x.get('description', '')[:100]}")

# Case 4 - LONGER (debug script)
e1l = mk("Enterprise revenue growth was 35% in 2024 according to industry survey of 500 companies.")
e2l = mk("Enterprise revenue growth reached 52% in 2024 based on annual technology report.")
c_long = _detect_contradictions([e1l, e2l], query="enterprise revenue growth rate")
print(f"LONG: {len(c_long)} contradictions")
for x in c_long:
    print(f"  {x.get('conflict_type')}: {x.get('description', '')[:100]}")

# Now debug the short case manually
text_i = e1s.text.lower()
text_j = e2s.text.lower()
words_i = set(re.findall(r"\b\w{4,}\b", text_i))
words_j = set(re.findall(r"\b\w{4,}\b", text_j))
shared_metrics = words_i & words_j & _METRIC_KEYWORDS
print(f"\nDEBUG SHORT:")
print(f"  words_i: {sorted(words_i)}")
print(f"  words_j: {sorted(words_j)}")
print(f"  shared_metrics: {shared_metrics}")
print(f"  topic_coherent: {_is_topic_coherent(e1s.text, e2s.text)}")
print(f"  entities_i: {_extract_entity_keywords(e1s.text)}")
print(f"  entities_j: {_extract_entity_keywords(e2s.text)}")

if shared_metrics and _is_topic_coherent(e1s.text, e2s.text):
    metric_nums_i = set()
    metric_nums_j = set()
    raw_nums_i = []
    raw_nums_j = []
    for metric in shared_metrics:
        for text, num_set, raw_list in [(text_i, metric_nums_i, raw_nums_i), (text_j, metric_nums_j, raw_nums_j)]:
            for m in re.finditer(r"\b" + re.escape(metric) + r"\b", text):
                start = max(0, m.start() - 50)
                end = min(len(text), m.end() + 50)
                window = text[start:end]
                nums = re.findall(r"\$?[\d,]+\.?\d*\s*(?:billion|million|thousand|%)?", window)
                print(f"  metric={metric} window={window!r} nums={nums}")
                for n in nums:
                    cleaned = re.sub(r"[,$]", "", n.strip())
                    if cleaned:
                        num_set.add(cleaned)
                        raw_nums_i.append(_normalize_number_with_unit(n)) if num_set is metric_nums_i else raw_nums_j.append(_normalize_number_with_unit(n))

    print(f"  metric_nums_i: {metric_nums_i}")
    print(f"  metric_nums_j: {metric_nums_j}")
    print(f"  different: {metric_nums_i != metric_nums_j}")
