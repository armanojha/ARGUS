"""Debug why Case 4 short text fails normalization check."""
import re
from app.orchestration.nodes import (
    _METRIC_KEYWORDS, _is_topic_coherent, _normalize_number_with_unit, _normalize_value,
)

text_i = "enterprise revenue growth was 35% in 2024 according to industry survey."
text_j = "enterprise revenue growth reached 52% in 2024 based on annual report."

words_i = set(re.findall(r"\b\w{4,}\b", text_i))
words_j = set(re.findall(r"\b\w{4,}\b", text_j))
shared_metrics = words_i & words_j & _METRIC_KEYWORDS

metric_nums_i = set()
metric_nums_j = set()
raw_nums_i = []
raw_nums_j = []

for metric in shared_metrics:
    for label, text, num_set, raw_list in [
        ("i", text_i, metric_nums_i, raw_nums_i),
        ("j", text_j, metric_nums_j, raw_nums_j),
    ]:
        for m in re.finditer(r"\b" + re.escape(metric) + r"\b", text):
            start = max(0, m.start() - 50)
            end = min(len(text), m.end() + 50)
            window = text[start:end]
            for n in re.findall(r"\$?[\d,]+\.?\d*\s*(?:billion|million|thousand|%)?", window):
                cleaned = re.sub(r"[,$]", "", n.strip())
                if cleaned:
                    num_set.add(cleaned)
                    raw_list.append(_normalize_number_with_unit(n))

print(f"metric_nums_i: {metric_nums_i}")
print(f"metric_nums_j: {metric_nums_j}")
print(f"different: {metric_nums_i != metric_nums_j}")

norm_i = {_normalize_value(v, u) for v, u in raw_nums_i if v > 0}
norm_j = {_normalize_value(v, u) for v, u in raw_nums_j if v > 0}
print(f"raw_nums_i: {raw_nums_i}")
print(f"raw_nums_j: {raw_nums_j}")
print(f"norm_i: {norm_i}")
print(f"norm_j: {norm_j}")
print(f"norm_i == norm_j: {norm_i == norm_j}")
print(f"norm_i and norm_j: {bool(norm_i and norm_j)}")

# The key question: does the continue fire?
if norm_i and norm_j and norm_i == norm_j:
    print("WOULD CONTINUE (skip contradiction)")
else:
    print("WOULD NOT CONTINUE (add contradiction)")
