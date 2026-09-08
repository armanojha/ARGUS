import re
from app.orchestration.nodes import _METRIC_KEYWORDS, _NEGATION_PAIRS, _is_topic_coherent

text_i = "enterprise revenue growth was 35% in 2024 according to industry survey of 500 companies."
text_j = "enterprise revenue growth reached 52% in 2024 based on annual technology report."

words_i = set(re.findall(r"\b\w{4,}\b", text_i))
words_j = set(re.findall(r"\b\w{4,}\b", text_j))
shared_metrics = words_i & words_j & _METRIC_KEYWORDS
print("shared_metrics:", shared_metrics)
print("topic_coherent:", _is_topic_coherent(text_i, text_j, min_shared_significant=3))

# Check numerical
if shared_metrics and _is_topic_coherent(text_i, text_j, min_shared_significant=3):
    metric_nums_i = set()
    metric_nums_j = set()
    for metric in shared_metrics:
        for label, text, num_set in [("i", text_i, metric_nums_i), ("j", text_j, metric_nums_j)]:
            for m in re.finditer(r"\b" + re.escape(metric) + r"\b", text):
                start = max(0, m.start() - 50)
                end = min(len(text), m.end() + 50)
                window = text[start:end]
                for n in re.findall(r"\$?[\d,]+\.?\d*\s*(?:billion|million|thousand|%)?", window):
                    cleaned = re.sub(r"[,$]", "", n.strip())
                    if cleaned:
                        num_set.add(cleaned)
    print("nums_i:", metric_nums_i)
    print("nums_j:", metric_nums_j)
    print("equal:", metric_nums_i == metric_nums_j)
    print("different:", metric_nums_i != metric_nums_j)
