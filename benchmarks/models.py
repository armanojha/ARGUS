"""Benchmark data models (Phase 12.3).

In-memory schemas for benchmark items and per-item run outputs. All of the
evaluation logic operates on these plain dataclasses so the harness can be
run against the real ARGUS pipelines (live keys) or a scripted stub
(offline smoke) without coupling evaluation to any one runtime path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Canonical benchmark item types (V2 §13.1): 20 each in the v1 set.
ITEM_TYPES = ["easy_factual", "multi_hop", "temporal", "contradiction", "synthesis"]

# Canonical adversarial categories (V2 §13.1).
ADVERSARIAL_TYPES = [
    "near_duplicate_incorrect",
    "revised_facts",
    "differing_metric_definitions",
    "ambiguous_entity_names",
    "incomplete_evidence",
]


@dataclass
class BenchmarkItem:
    """A single benchmark question with its gold answer and evidence.

    `gold_evidence` passages are the ground-truth sources for the gold
    answer (the benchmark corpus is built from these). `distractor_evidence`
    are adversarial documents that look plausible but are wrong; the harness
    tracks whether a run is misled by them.
    """

    id: str
    type: str
    question: str
    gold_answer: str
    gold_evidence: list[str] = field(default_factory=list)
    gold_years: list[str] = field(default_factory=list)
    expect_contradiction: bool = False
    adversarial_type: str | None = None
    distractor_evidence: list[str] = field(default_factory=list)


@dataclass
class BenchmarkRunOutput:
    """Deterministic surface output of one pipeline run over one item.

    Everything the metrics need, in pure data form — no live objects, so
    reports remain reproducible after the run finishes.
    """

    item_id: str
    answer: str
    cited_chunk_ids: list[str] = field(default_factory=list)
    retrieved_chunk_ids: list[str] = field(default_factory=list)
    loop_count: int = 0
    tokens_used: int = 0
    latency_ms: int = 0
    failed_calls: int = 0
    verification_status: str | None = None
    contradiction_detected: bool = False
    warned: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CorpusContext:
    """Reference handle to the rented benchmark corpus for one run."""

    gold_chunk_ids: dict[str, list[str]] = field(default_factory=dict)
    distractor_chunk_ids: dict[str, list[str]] = field(default_factory=dict)
    chunk_text_by_id: dict[str, str] = field(default_factory=dict)
    item_meta: dict[str, dict[str, Any]] = field(default_factory=dict)
    build_duration_ms: int = 0