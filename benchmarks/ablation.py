"""Ablation harness (Phase 12.4): compare ARGUS pipeline variants.

Each variant is a ``Pipeline`` (same surface as the 12.3 runner) built purely
by composing **existing** ARGUS services with declarative toggles, so every
measured difference is a structural one — never an arbitrary re-implementation.
Variants:

* ``baseline_rag``         — BM25-only single-shot retrieval + standalone
                             synthesis node (no loop, no verification).
* ``hybrid_rag``           — hybrid (BM25 + FAISS) single-shot retrieval +
                             standalone synthesis node.
* ``agentic_rag``          — full Phase 02 orchestration loop with adaptive
                             policies disabled (clean loop), no verification.
* ``graphrag_variant``     — hybrid-seeded graph retrieval + synthesis node.
* ``argus_without_verifier`` — full Phase 02 loop (policies on) minus the
                             Phase 04 verification step.
* ``argus_without_obsidian`` — full loop + verification with Obsidian disabled
                             (configuration-control variant).
* ``full_argus``           — full loop + verification (reference artifact).

`VariantSpec` is purely declarative; no model selection happens here (routing
stays explicit server-side, config-managed).
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from app.config import Settings, get_settings
from app.evidence.store import EvidenceStore
from app.orchestration.models import ResearchPlan, StopReason
from app.orchestration.nodes import extract_cited_indices, make_synthesize_node
from app.orchestration.state import OrchestrationState
from app.reranking.reranker import NoOpReranker
from app.retrieval.hybrid import HybridRetriever
from app.verification.engine import verify_claim
from app.verification.models import VerificationRequest, VerificationStatus

from .models import BenchmarkItem, BenchmarkRunOutput, CorpusContext
from .runner import Pipeline, default_sources, load_items, make_full_argus_pipeline

VariantId = Literal[
    "baseline_rag",
    "hybrid_rag",
    "agentic_rag",
    "graphrag_variant",
    "argus_without_verifier",
    "argus_without_obsidian",
    "full_argus",
]

VARIANT_ORDER: list[VariantId] = [
    "baseline_rag",
    "hybrid_rag",
    "agentic_rag",
    "graphrag_variant",
    "argus_without_verifier",
    "argus_without_obsidian",
    "full_argus",
]

# Toggles that strip the Phase 02 loop down to its pure core (no adaptive
# policy, no active evidence seeking, no stopping logic, no memory, no
# multi-agent scaffolding) — used by the `agentic_rag` variant.
_LOOP_TOGGLES: dict[str, Any] = {
    "retrieval_policy_enabled": False,
    "active_evidence_seeking_enabled": False,
    "stopping_logic_enabled": False,
    "memory_enabled": False,
    "multiagent_enabled": False,
}


@dataclass(frozen=True)
class VariantSpec:
    """Declarative description of one ablation variant (Phase 12.4)."""

    id: VariantId
    label: str
    description: str


VARIANTS: dict[VariantId, VariantSpec] = {
    v.id: v
    for v in [
        VariantSpec(
            "baseline_rag",
            "Baseline RAG",
            "BM25-only single-shot retrieval, single synthesis call, no verification.",
        ),
        VariantSpec(
            "hybrid_rag",
            "Hybrid RAG",
            "BM25 + FAISS single-shot retrieval, single synthesis call, no verification.",
        ),
        VariantSpec(
            "agentic_rag",
            "Agentic RAG",
            "Phase 02 agentic loop (policies off), no verification.",
        ),
        VariantSpec(
            "graphrag_variant",
            "GraphRAG variant",
            "Hybrid-seeded graph retrieval + standalone synthesis node.",
        ),
        VariantSpec(
            "argus_without_verifier",
            "ARGUS w/o verifier",
            "Full Phase 02 loop (policies on) minus the verification step.",
        ),
        VariantSpec(
            "argus_without_obsidian",
            "ARGUS w/o Obsidian",
            "Full loop + verification with Obsidian explicitly disabled.",
        ),
        VariantSpec(
            "full_argus",
            "Full ARGUS",
            "Full loop + verification (reference for deltas).",
        ),
    ]
}


def _standalone_synthesis_pipeline(
    variant: VariantId,
    *,
    router: Any,
    retriever: HybridRetriever,
    graph_store: Any,
    settings: Settings,
) -> Pipeline:
    """Single-shot retrieval + standalone synthesis (baseline/hybrid/graph variants)."""

    synthesize = make_synthesize_node(router, settings)

    async def pipeline(item: BenchmarkItem, corpus: CorpusContext) -> BenchmarkRunOutput:
        t0 = time.monotonic()
        if variant == "baseline_rag":
            retrieved = retriever.search_bm25_only(item.question, top_k=10)
        elif variant == "graphrag_variant":
            from app.graph.retrieval import GraphRetriever

            graph_retriever = GraphRetriever(
                graph_store=graph_store,
                hybrid_retriever=retriever,
                settings=settings,
            )
            retrieved = graph_retriever.search(item.question, top_k=10)
        else:
            retrieved = retriever.search(item.question, top_k=10)

        retrieved_ids = [str(r.chunk_id) for r in retrieved]
        plan = ResearchPlan(
            objective=item.question,
            entities=[item.question[:40]],
            time_window=None,
            subquestions=[item.question],
        )
        state: OrchestrationState = {
            "request_id": f"bench:{item.id}",
            "query": item.question,
            "max_iterations": 1,
            "token_budget": settings.orchestration_token_budget,
            "query_analysis": None,
            "plan": plan,
            "pending_subquestions": [],
            "issued_subqueries": [item.question],
            "evidence": retrieved,
            "consecutive_empty_retrievals": 0,
            "iteration": 1,
            "tokens_used": len(item.question) // 4,
            "sufficient": True,
            "stop_reason": StopReason.SUFFICIENT_EVIDENCE.value,
            "answer": None,
            "warnings": [],
        }
        synthesized = await synthesize(state)

        answer = synthesized.get("answer", "")
        warnings = list(synthesized.get("warnings", []) or [])
        if not retrieved:
            answer = answer or "No supporting evidence was retrieved for this question."
            warnings.append("empty_retrieval")

        cited_ids = [
            str(retrieved[i - 1].chunk_id)
            for i in extract_cited_indices(answer, len(retrieved))
        ]

        return BenchmarkRunOutput(
            item_id=item.id,
            answer=answer,
            cited_chunk_ids=cited_ids,
            retrieved_chunk_ids=retrieved_ids,
            loop_count=1,
            tokens_used=len(answer) // 4,
            latency_ms=int((time.monotonic() - t0) * 1000),
            failed_calls=0,
            warned=warnings,
            metadata={"variant": variant, "retrieved_count": len(retrieved)},
        )

    return pipeline


def make_variants(
    *,
    router: Any,
    corpus: CorpusContext,
    settings: Settings | None = None,
) -> dict[VariantId, Pipeline]:
    """Build all ablation variant pipelines over one benchmark corpus."""
    settings = settings or get_settings()
    sources = default_sources(corpus)
    retriever: HybridRetriever = sources["retriever"]
    graph_store = sources["graph_store"]
    store: EvidenceStore = sources["store"]

    def loop(variant_settings: Settings, *, verify: bool) -> Pipeline:
        return make_full_argus_pipeline(
            router=router,
            evidence_store=store,
            graph_store=graph_store,
            retriever=retriever,
            reranker=NoOpReranker(),
            settings=variant_settings,
            verify_answer=verify,
        )

    def single(variant: VariantId) -> Pipeline:
        return _standalone_synthesis_pipeline(
            variant, router=router, retriever=retriever, graph_store=graph_store, settings=settings
        )

    return {
        "baseline_rag": single("baseline_rag"),
        "hybrid_rag": single("hybrid_rag"),
        "agentic_rag": loop(settings.model_copy(update=_LOOP_TOGGLES), verify=False),
        "graphrag_variant": single("graphrag_variant"),
        "argus_without_verifier": loop(settings, verify=False),
        "argus_without_obsidian": loop(
            settings.model_copy(update={"obsidian_enabled": False}), verify=True
        ),
        "full_argus": loop(settings, verify=True),
    }


def run_ablation(
    *,
    router: Any,
    question_path: Path | None = None,
    limit: int | None = None,
    working_dir: Path | None = None,
    out_dir: Path | None = None,
    settings: Settings | None = None,
    name: str = "ARGUS ablation v1",
) -> dict[str, Any]:
    """Run every variant over the shared question set and compare against `full_argus`.

    Returns a JSON-serializable report with per-variant metric aggregates and a
    delta table versus the reference variant. Reports are written to `out_dir`
    when provided (ablation_report.json / ablation_report.md).
    """
    from .runner import build_corpus, score_items

    settings = settings or get_settings()
    items = load_items(question_path or Path(__file__).resolve().parent / "data" / "questions_v1.json")
    if limit is not None:
        items = items[:limit]

    if working_dir is None:
        import tempfile

        work = Path(tempfile.mkdtemp(prefix="argus_ablation_"))
    else:
        work = working_dir
    corpus = build_corpus(items, work)

    import asyncio

    per_variant: dict[str, dict[str, Any]] = {}
    for variant_id in VARIANT_ORDER:
        pipeline = make_variants(router=router, corpus=corpus, settings=settings)[variant_id]
        outputs = asyncio.run(_run_all(pipeline, items, corpus))
        scored = score_items(items, outputs, corpus)
        per_variant[variant_id] = {
            "label": VARIANTS[variant_id].label,
            "description": VARIANTS[variant_id].description,
            "metrics": scored["metrics"],
            "by_type": scored["by_type"],
        }

    reference = per_variant["full_argus"]["metrics"]
    deltas: dict[str, dict[str, float]] = {}
    for variant_id, data in per_variant.items():
        deltas[variant_id] = {
            name: _delta(reference.get(name, {}).get("value"), info.get("value"))
            for name, info in data["metrics"].items()
            if name in reference and name not in {"avg_loop_count", "avg_tokens_per_query", "avg_latency_ms", "total_failed_calls"}
        }

    report: dict[str, Any] = {
        "name": name,
        "generated_at": datetime.now(UTC).isoformat(),
        "item_count": len(items),
        "variants": per_variant,
        "deltas_vs_full_argus": deltas,
        "reference_variant": "full_argus",
        "corpus": {
            "build_duration_ms": corpus.build_duration_ms,
            "gold_chunks": sum(len(v) for v in corpus.gold_chunk_ids.values()),
        },
    }
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        import json as _json

        (out_dir / "ablation_report.json").write_text(
            _json.dumps(report, indent=2, default=str), encoding="utf-8"
        )
        (out_dir / "ablation_report.md").write_text(ablation_markdown(report), encoding="utf-8")
    return report


async def _run_all(pipeline: Pipeline, items: list[BenchmarkItem], corpus: CorpusContext) -> list[BenchmarkRunOutput]:
    outputs: list[BenchmarkRunOutput] = []
    for item in items:
        outputs.append(await pipeline(item, corpus))
    return outputs


def _delta(reference: Any, value: Any) -> float | None:
    if reference is None or value is None:
        return None
    try:
        r, v = float(reference), float(value)
    except (TypeError, ValueError):
        return None
    if r != r or v != v:  # NaN-safe
        return None
    return round(v - r, 4)


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        f = float(value)
    except (TypeError, ValueError):
        return str(value)
    return "n/a" if f != f else f"{f:.4f}"


def ablation_markdown(report: dict[str, Any]) -> str:
    """Render the ablation report (metrics x variants + delta vs full)."""
    lines: list[str] = [
        f"# {report['name']}",
        "",
        f"- generated: {report['generated_at']}",
        f"- items: {report['item_count']}",
        "",
        "## Metrics by variant (value / applicable)",
        "",
        "| Variant | "
        + " | ".join(
            m for m in [
                "recall_at_10", "evidence_precision", "citation_correctness",
                "claim_support_rate", "contradiction_recall", "temporal_accuracy",
                "answer_faithfulness", "adversarial_robustness",
                "avg_loop_count", "avg_latency_ms",
            ]
        )
        + " |",
        "|" + "---|" * 11,
    ]
    for variant_id in VARIANT_ORDER:
        data = (report.get("variants") or {}).get(variant_id)
        if not data:
            continue
        metrics = data["metrics"]
        row = [VARIANTS[variant_id].label]
        for m in [
            "recall_at_10", "evidence_precision", "citation_correctness",
            "claim_support_rate", "contradiction_recall", "temporal_accuracy",
            "answer_faithfulness", "adversarial_robustness", "avg_loop_count", "avg_latency_ms",
        ]:
            info = metrics.get(m, {})
            row.append(_fmt(info.get("value")))
        lines.append("| " + " | ".join(row) + " |")

    lines += ["", "## Delta vs full_argus (value difference)", "", "| Variant | " + " | ".join(
        m for m in [
            "recall_at_10", "evidence_precision", "citation_correctness",
            "claim_support_rate", "contradiction_recall", "temporal_accuracy",
            "answer_faithfulness", "adversarial_robustness",
        ]
    ) + " |", "|" + "---|" * 9]
    for variant_id in VARIANT_ORDER:
        delta = (report.get("deltas_vs_full_argus") or {}).get(variant_id) or {}
        row = [VARIANTS[variant_id].label]
        for m in [
            "recall_at_10", "evidence_precision", "citation_correctness",
            "claim_support_rate", "contradiction_recall", "temporal_accuracy",
            "answer_faithfulness", "adversarial_robustness",
        ]:
            row.append(_fmt(delta.get(m)))
        lines.append("| " + " | ".join(row) + " |")

    lines += ["", "## V3-specific", "", "- vault personalization gain: reported per run in the 12.3 benchmark report.",
              "- reindex cost: reported per run in the 12.3 benchmark report.",
              "- write-back usefulness: reported per run in the 12.3 benchmark report."]
    return "\n".join(lines) + "\n"