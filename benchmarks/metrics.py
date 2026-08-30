"""Deterministic benchmark metrics (Phase 12.3, V2 §13).

Every metric is computed from the surface output of a run (answer text,
cited chunk ids, retrieved chunk ids, verification signal) against gold
evidence. No live LLM is required for scoring — the harness stays
reproducible offline. An optional ``judge`` callable may override the
lexical answer-faithfulness proxy when a live judge is preferred.

V3-specific metrics (vault personalization gain, reindex cost, write-back
usefulness) are computed where the run exposes the required data and are
reported as not-applicable otherwise.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

NOT_APPLICABLE = float("nan")

_CITATION_MARKER = re.compile(r"\[\s*(\d+)\s*\]")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_TOKEN = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> set[str]:
    return set(_TOKEN.findall(text.lower()))


def _tokenize_words(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def token_f1(predicted: str, gold: str) -> float:
    """Token-level F1 between two texts (lexical faithfulness proxy, V2 §13 'answer faithfulness')."""
    p = _tokenize_words(predicted)
    g = _tokenize_words(gold)
    p_counts: dict[str, int] = {}
    g_counts: dict[str, int] = {}
    for t in p:
        p_counts[t] = p_counts.get(t, 0) + 1
    for t in g:
        g_counts[t] = g_counts.get(t, 0) + 1
    overlap = 0
    for t, c in p_counts.items():
        overlap += min(c, g_counts.get(t, 0))
    if overlap == 0:
        return 0.0
    precision = overlap / len(p) if p else 0.0
    recall = overlap / len(g) if g else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def recall_at_k(retrieved: list[str], gold: set[str], k: int) -> float:
    """Recall@K (V2 §13): fraction of gold evidence chunks in the top-K retrieved."""
    if not gold:
        return NOT_APPLICABLE
    return sum(1 for cid in retrieved[:k] if cid in gold) / len(gold)


def evidence_precision(cited: list[str], gold: set[str]) -> float:
    """Evidence precision (V2 §13): fraction of cited evidence chunks that are gold."""
    if not cited:
        return 0.0
    return sum(1 for cid in cited if cid in gold) / len(cited)


def citation_correctness(answer: str, cited: list[str], gold: set[str]) -> float:
    """Citation correctness (V2 §13): fraction of bracket citations in the answer
    that resolve to a gold evidence chunk (provenance accuracy)."""
    refs = [int(m) for m in _CITATION_MARKER.findall(answer)]
    if not refs:
        return 0.0
    resolved = [cited[r - 1] for r in refs if 1 <= r <= len(cited)]
    if not resolved:
        return 0.0
    return sum(1 for cid in resolved if cid in gold) / len(resolved)


def _lexical_grounding(sentence: str, chunk_texts: list[str]) -> bool:
    tokens = _tokens(sentence)
    if not tokens:
        return False
    for chunk in chunk_texts:
        chunk_tokens = _tokens(chunk)
        if not chunk_tokens:
            continue
        overlap = len(tokens & chunk_tokens)
        if overlap / len(tokens) >= 0.5:
            return True
    return False


def claim_support_rate(
    answer: str,
    cited: list[str],
    gold: set[str],
    chunk_text_by_id: dict[str, str],
) -> float:
    """Claim support rate (V2 §13): fraction of answer sentences that are grounded.

    A sentence is grounded if it carries a valid bracket citation whose chunk
    is gold, or it shares >=50% lexical tokens with a gold cited chunk. This is
    a deterministic proxy for 'claims supported by cited evidence'.
    """
    sentences = [s.strip() for s in _SENTENCE_SPLIT.split(answer) if s.strip()]
    if not sentences:
        return 0.0
    supported = 0
    for sentence in sentences:
        refs = [int(m) for m in _CITATION_MARKER.findall(sentence)]
        has_gold_ref = any(1 <= r <= len(cited) and cited[r - 1] in gold for r in refs)
        if has_gold_ref:
            supported += 1
            continue
        gold_chunk_texts = [
            chunk_text_by_id[cid] for cid in cited if cid in gold and cid in chunk_text_by_id
        ]
        if _lexical_grounding(sentence, gold_chunk_texts):
            supported += 1
    return supported / len(sentences)


def contradiction_recall(detected: bool, expected: bool) -> float:
    """Contradiction recall (V2 §13): 1.0 when an expected contradiction is detected.

    Only applicable to items whose gold expects a contradiction; not-applicable
    (reported separately) otherwise so we never punish non-confrontational items.
    """
    if not expected:
        return NOT_APPLICABLE
    return 1.0 if detected else 0.0


def temporal_accuracy(answer: str, gold_years: list[str]) -> float:
    """Temporal accuracy (V2 §13): whether the answer contains the gold year(s)."""
    if not gold_years:
        return NOT_APPLICABLE
    return 1.0 if any(year in answer for year in gold_years) else 0.0


def answer_faithfulness(
    answer: str,
    gold_answer: str,
    judge: Callable[[str, str, str], float] | None = None,
) -> float:
    """Answer faithfulness (V2 §13): lexical F1 vs gold (or an injected judge)."""
    if judge is not None:
        return float(judge(answer, gold_answer, ""))
    return token_f1(answer, gold_answer)


def adversarial_robustness(
    cited: list[str], distractor_ids: set[str], adversarial: bool
) -> float:
    """Adversarial robustness: 1.0 when no distractor (wrong) document is cited."""
    if not adversarial:
        return NOT_APPLICABLE
    return 0.0 if any(cid in distractor_ids for cid in cited) else 1.0


def vault_personalization_gain(metadata: dict[str, Any]) -> float:
    """V3 §15: answer-quality lift attributable to vault personalization.

    Computed only when the run exposes the ``vault_personalization_gain``
    counter (Obsidian/V3 feature area, Phase 09+); not-applicable otherwise
    so offline/non-V3 runs are never spuriously penalized.
    """
    value = (metadata or {}).get("vault_personalization_gain")
    return NOT_APPLICABLE if value is None else float(value)


def reindex_cost(metadata: dict[str, Any]) -> float:
    """V3 §15: cost (ms) of reindexing the vault for a run's personalization."""
    value = (metadata or {}).get("reindex_duration_ms")
    return NOT_APPLICABLE if value is None else float(value)


def write_back_usefulness(metadata: dict[str, Any]) -> float:
    """V3 §15: usefulness score of the Obsidian write-back hypothesis note."""
    value = (metadata or {}).get("write_back_usefulness")
    return NOT_APPLICABLE if value is None else float(value)


def compute_item_scores(
    item: Any,
    output: Any,
    gold_ids: set[str],
    distractor_ids: set[str],
    chunk_text_by_id: dict[str, str],
    judge: Callable[[str, str, str], float] | None = None,
) -> dict[str, float]:
    """Compute the full per-item metric vector (V2 §13 + adversarial)."""
    answer = output.answer or ""
    cited = output.cited_chunk_ids or []
    retrieved = output.retrieved_chunk_ids or []
    gold_years = getattr(item, "gold_years", []) or []
    adversarial = bool(getattr(item, "adversarial_type", None))

    scores: dict[str, float] = {
        "recall_at_5": recall_at_k(retrieved, gold_ids, 5),
        "recall_at_10": recall_at_k(retrieved, gold_ids, 10),
        "evidence_precision": evidence_precision(cited, gold_ids),
        "citation_correctness": citation_correctness(answer, cited, gold_ids),
        "claim_support_rate": claim_support_rate(answer, cited, gold_ids, chunk_text_by_id),
        "contradiction_recall": contradiction_recall(
            bool(getattr(output, "contradiction_detected", False)),
            bool(getattr(item, "expect_contradiction", False)),
        ),
        "temporal_accuracy": temporal_accuracy(answer, gold_years),
        "answer_faithfulness": answer_faithfulness(answer, item.gold_answer, judge),
        "adversarial_robustness": adversarial_robustness(
            cited, distractor_ids, adversarial
        ),
    }
    return scores


def aggregate_scores(
    per_item: list[tuple[Any, dict[str, float]]],
    outputs: list[Any] | None = None,
) -> dict[str, dict[str, float]]:
    """Aggregate per-item metric vectors into mean scores with applicability counts.

    Runtime counters come from the benchmark run outputs (when provided), so a
    pure-metric call with `outputs=None` still yields the metric means.
    """
    metric_names = [
        "recall_at_5",
        "recall_at_10",
        "evidence_precision",
        "citation_correctness",
        "claim_support_rate",
        "contradiction_recall",
        "temporal_accuracy",
        "answer_faithfulness",
        "adversarial_robustness",
    ]
    aggregated: dict[str, dict[str, float]] = {}
    for name in metric_names:
        values = [s[name] for _, s in per_item if s.get(name) == s.get(name)]  # drop nan
        applicable = len(values)
        aggregated[name] = {
            "value": (sum(values) / applicable) if applicable else float("nan"),
            "applicable": applicable,
        }
    # Aggregation of runtime counters.
    if outputs:
        total_loop = sum(int(getattr(o, "loop_count", 0)) for o in outputs)
        total_tokens = sum(int(getattr(o, "tokens_used", 0)) for o in outputs)
        total_latency = sum(int(getattr(o, "latency_ms", 0)) for o in outputs)
        total_failed = sum(int(getattr(o, "failed_calls", 0)) for o in outputs)
        n = len(outputs)
        aggregated["avg_loop_count"] = {"value": total_loop / n, "applicable": n}
        aggregated["avg_tokens_per_query"] = {"value": total_tokens / n, "applicable": n}
        aggregated["avg_latency_ms"] = {"value": total_latency / n, "applicable": n}
        aggregated["total_failed_calls"] = {"value": float(total_failed), "applicable": n}
    # V3 §15 counters: populated only when a run exposes the V3 metadata.
    _V3_KEYS = [
        ("vault_personalization_gain", "vault_personalization_gain"),
        ("reindex_cost", "reindex_duration_ms"),
        ("write_back_usefulness", "write_back_usefulness"),
    ]
    if outputs:
        for metric_name, key in _V3_KEYS:
            values = [
                float(meta[key])
                for o in outputs
                if key in (meta := (getattr(o, "metadata", None) or {}))
            ]
            aggregated[metric_name] = {
                "value": (sum(values) / len(values)) if values else float("nan"),
                "applicable": len(values),
            }
    return aggregated


def by_type_breakdown(
    per_item: list[tuple[Any, dict[str, float]]],
) -> dict[str, dict[str, float]]:
    """Per-question-type mean of the headline metrics (V2 §13.1 breakdown)."""
    headline = ["recall_at_10", "evidence_precision", "answer_faithfulness"]
    bucket: dict[str, list[dict[str, float]]] = {}
    for item, scores in per_item:
        bucket.setdefault(item.type, []).append(scores)
    breakdown: dict[str, dict[str, float]] = {}
    for item_type, group in bucket.items():
        breakdown[item_type] = {
            name: (sum(s[name] for s in group) / len(group) if group else float("nan"))
            for name in headline
        }
    return breakdown