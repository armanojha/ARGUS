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

import json
import math
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from app.config import Settings, get_settings
from app.evidence.store import EvidenceStore
from app.logging_config import get_logger
from app.orchestration.models import ResearchPlan, StopReason
from app.orchestration.nodes import extract_cited_indices, make_synthesize_node
from app.orchestration.state import OrchestrationState
from app.reranking.reranker import NoOpReranker
from app.retrieval.hybrid import HybridRetriever

from .models import BenchmarkItem, BenchmarkRunOutput, CorpusContext
from .runner import Pipeline, default_sources, load_items, make_full_argus_pipeline

logger = get_logger("argus.benchmark.ablation")

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
    from .runner import build_corpus

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

    # Single persistent event loop owned by the harness owns the whole
    # ablation. Running `asyncio.run(...)` per variant created a *fresh* loop
    # every time while the router's lazy `httpx.AsyncClient` stayed bound to
    # the first loop, so a lingering/in-flight connection produced
    # `RuntimeError: Event loop is closed` at variant boundaries. One loop
    # also lets us close the router (and therefore every provider client) on
    # the same loop that created them.
    per_variant, variant_failures, item_failures = asyncio.run(
        _run_all_variants(
            router=router,
            items=items,
            corpus=corpus,
            settings=settings,
            checkpoint_path=(out_dir / "ablation_checkpoint.json") if out_dir is not None else None,
        )
    )

    reference = (per_variant.get("full_argus") or {}).get("metrics")
    deltas: dict[str, dict[str, float | None]] = {}
    if reference is not None:
        for vid, data in per_variant.items():
            deltas[vid] = {
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
        "variant_failures": variant_failures,
        "item_failures": item_failures,
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


async def _run_all_variants(
    *,
    router: Any,
    items: list[BenchmarkItem],
    corpus: CorpusContext,
    settings: Settings,
    checkpoint_path: Path | None = None,
) -> tuple[dict[str, dict[str, Any]], dict[str, str], dict[str, list[dict[str, str]]]]:
    """Run every variant inside one persistent event loop and score each.

    Lifespan strategy: a single loop owns the entire ablation, so the router's
    lazy ``httpx.AsyncClient`` (created on first request) is bound to one loop
    and no client ever survives its owning loop. The router is closed in a
    ``finally`` *before* `asyncio.run` returns, which releases every provider
    connection cleanly and cancels any lingering in-flight work.

    Isolation:
    * per-item — `_run_all` records a failing item as a failure output and
      continues, so one pathological item never aborts a variant;
    * per-variant — a failing variant is recorded in `variant_failures` and
      the remaining variants still run and are scored.

    When `checkpoint_path` is set, progress is checkpointed: after every item
    of the in-flight variant (per-item outputs, so an interrupted variant is
    continued rather than restarted) and after every completed variant
    (aggregated scores, so a later re-launch skips completed variants). On
    startup any existing checkpoint is loaded and the run resumes from it, so a
    hard process interruption preserves all variants already scored.

    Returns ``(per_variant, variant_failures, item_failures)``: per-variant
    label/description/metrics/by_type; a map of variant_id -> error string for
    variants that could not be run at all; and a per-variant list of recorded
    item failures ``[{"item_id": ..., "error": ...}, ...]``.
    """
    from .runner import score_items

    per_variant: dict[str, dict[str, Any]] = {}
    variant_failures: dict[str, str] = {}
    item_failures: dict[str, list[dict[str, str]]] = {}
    partial: dict[str, dict[str, Any]] = {}

    restored = _load_checkpoint(checkpoint_path)
    if restored:
        per_variant = dict(restored.get("per_variant") or {})
        variant_failures = dict(restored.get("variant_failures") or {})
        item_failures = {k: list(v) for k, v in (restored.get("item_failures") or {}).items()}
        partial = {k: dict(v) for k, v in (restored.get("partial") or {}).items()}
        if per_variant:
            logger.info("ablation_resumed", completed_variants=sorted(per_variant))

    try:
        for variant_id in VARIANT_ORDER:
            if variant_id in per_variant:
                continue
            try:
                pipeline = make_variants(router=router, corpus=corpus, settings=settings)[
                    variant_id
                ]
                resume_state = partial.get(variant_id)
                resume_outputs = (
                    [_output_from_dict(d) for d in resume_state.get("outputs", [])]
                    if resume_state
                    else []
                )
                resume_failures = (
                    list(resume_state.get("failures", [])) if resume_state else []
                )

                def after_item(
                    all_outputs: list[BenchmarkRunOutput],
                    all_failures: list[dict[str, str]],
                    _variant_id: str = variant_id,
                ) -> None:
                    partial[_variant_id] = {
                        "outputs": [_output_to_dict(o) for o in all_outputs],
                        "failures": all_failures,
                    }
                    try:
                        _write_checkpoint(
                            checkpoint_path,
                            active_variant=_variant_id,
                            per_variant=per_variant,
                            variant_failures=variant_failures,
                            item_failures=item_failures,
                            partial=partial,
                        )
                    except OSError:  # noqa: BLE001 - checkpoint must never abort the run
                        logger.warning("ablation_checkpoint_write_failed")

                outputs, failures = await _run_all(
                    pipeline,
                    items,
                    corpus,
                    resume_outputs=resume_outputs,
                    resume_failures=resume_failures,
                    after_item=after_item if checkpoint_path is not None else None,
                )
                scored = score_items(items, outputs, corpus)
                per_variant[variant_id] = {
                    "label": VARIANTS[variant_id].label,
                    "description": VARIANTS[variant_id].description,
                    "metrics": scored["metrics"],
                    "by_type": scored["by_type"],
                }
                if failures:
                    item_failures[variant_id] = failures
                partial.pop(variant_id, None)
                if checkpoint_path is not None:
                    try:
                        _write_checkpoint(
                            checkpoint_path,
                            active_variant=None,
                            per_variant=per_variant,
                            variant_failures=variant_failures,
                            item_failures=item_failures,
                            partial=partial,
                        )
                    except OSError:  # noqa: BLE001 - checkpoint must never abort the run
                        logger.warning("ablation_checkpoint_write_failed")
            except Exception as exc:  # noqa: BLE001 - variant-level fault isolation
                variant_failures[variant_id] = f"{type(exc).__name__}: {exc}"
                logger.error(
                    "ablation_variant_failed",
                    variant=variant_id,
                    error=f"{type(exc).__name__}: {exc}",
                )
                continue
    finally:
        close = getattr(router, "aclose", None)
        if close is not None:
            try:
                await close()
            except Exception:  # noqa: BLE001 - closing must never mask run results
                logger.warning("ablation_router_close_failed")
    return per_variant, variant_failures, item_failures


def _load_checkpoint(checkpoint_path: Path | None) -> dict[str, Any]:
    """Load a prior ablation_checkpoint.json ({} when absent/corrupt).

    The checkpoint is the single source of truth for resuming: completed
    variants (aggregated scores) are skipped, and an in-flight variant's
    per-item outputs are continued rather than restarted.
    """
    if checkpoint_path is None or not checkpoint_path.exists():
        return {}
    try:
        data = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _write_checkpoint(
    checkpoint_path: Path,
    *,
    active_variant: VariantId | None,
    per_variant: dict[str, dict[str, Any]],
    variant_failures: dict[str, str],
    item_failures: dict[str, list[dict[str, str]]],
    partial: dict[str, dict[str, Any]],
) -> None:
    """Persist the running state; callers must wrap OSError so this never aborts."""
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.write_text(
        json.dumps(
            {
                "active_variant": active_variant,
                "completed_variants": sorted(per_variant),
                "per_variant": per_variant,
                "variant_failures": variant_failures,
                "item_failures": item_failures,
                "partial": partial,
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )


def _output_to_dict(output: BenchmarkRunOutput) -> dict[str, Any]:
    return asdict(output)


def _output_from_dict(data: dict[str, Any]) -> BenchmarkRunOutput:
    return BenchmarkRunOutput(**data)


async def _run_all(
    pipeline: Pipeline,
    items: list[BenchmarkItem],
    corpus: CorpusContext,
    *,
    resume_outputs: list[BenchmarkRunOutput] | None = None,
    resume_failures: list[dict[str, str]] | None = None,
    after_item: Callable[[list[BenchmarkRunOutput], list[dict[str, str]]], None] | None = None,
) -> tuple[list[BenchmarkRunOutput], list[dict[str, str]]]:
    """Run every item for one variant; a single failing item must not abort the rest.

    Returns ``(outputs, failures)`` where ``failures`` is the list of recorded
    item failures in run order (empty when every item completed).

    Resume: when ``resume_outputs``/``resume_failures`` are provided (from the
    per-item checkpoint of a previously interrupted variant), items whose id is
    already recorded are skipped and the partial outputs are continued.
    ``after_item`` (when set) is invoked after every newly-run item with the
    running (outputs, failures); checkpoint writers use it to persist progress
    so an interruption loses at most the in-flight item.
    """
    outputs = list(resume_outputs or [])
    failures = list(resume_failures or [])
    done_ids = {o.item_id for o in outputs}
    for item in items:
        if item.id in done_ids:
            continue
        try:
            outputs.append(await pipeline(item, corpus))
        except Exception as exc:  # noqa: BLE001 - per-item fault isolation
            error = f"{type(exc).__name__}: {exc}"
            logger.error(
                "ablation_item_failed",
                item=item.id,
                error=error,
            )
            failures.append({"item_id": item.id, "error": error})
            outputs.append(
                BenchmarkRunOutput(
                    item_id=item.id,
                    answer="",
                    warned=[f"harness_item_failed:{error}"],
                    metadata={"item_error": error},
                )
            )
        if after_item is not None:
            try:
                after_item(outputs, failures)
            except (OSError, TypeError, ValueError):  # noqa: BLE001 - checkpoint must never abort the run
                logger.warning("ablation_checkpoint_write_failed")
    return outputs, failures


def _delta(reference: Any, value: Any) -> float | None:
    if reference is None or value is None:
        return None
    try:
        r, v = float(reference), float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(r) or math.isnan(v):  # NaN-safe
        return None
    return round(v - r, 4)


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        f = float(value)
    except (TypeError, ValueError):
        return str(value)
    return "n/a" if math.isnan(f) else f"{f:.4f}"


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

    variant_failures = report.get("variant_failures") or {}
    if variant_failures:
        lines += ["", "## Variant failures", ""]
        for vid, err in variant_failures.items():
            lines.append(f"- `{vid}`: {err}")

    item_failures = report.get("item_failures") or {}
    if item_failures:
        lines += ["", "## Item failures", ""]
        for vid, failures in item_failures.items():
            for entry in failures:
                lines.append(f"- `{vid}` :: `{entry['item_id']}`: {entry['error']}")

    lines += ["", "## V3-specific", "", "- vault personalization gain: reported per run in the 12.3 benchmark report.",
              "- reindex cost: reported per run in the 12.3 benchmark report.",
              "- write-back usefulness: reported per run in the 12.3 benchmark report."]
    return "\n".join(lines) + "\n"