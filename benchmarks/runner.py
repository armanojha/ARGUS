"""Benchmark runner (Phase 12.3): corpus, pipelines, scoring, reports.

Composition of **existing** ARGUS services only:
  * corpus: Phase 01 `EvidenceStore` + `HybridRetriever` built from the
    benchmark question set (gold passages + adversarial distractors);
  * pipeline: Phase 02 `run_query` orchestration + Phase 04 `verify_claim`
    (both injectable so live runs and offline stubs share one harness);
  * scoring: `benchmarks.metrics` (deterministic).

Nothing here modifies core modules and no model selection happens here —
routing stays explicit server-side (config-managed).
"""

from __future__ import annotations

import hashlib
import json
import math
import tempfile
import time
import uuid
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.config import Settings, get_settings
from app.evidence.models import Chunk, Document, Source, SourceType
from app.evidence.store import EvidenceStore
from app.llm_gateway.telemetry import end_run_telemetry, start_run_telemetry
from app.retrieval.bm25 import BM25Retriever
from app.retrieval.embeddings import EmbeddingGenerator
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.vector import FAISSVectorStore
from app.verification.engine import verify_claim
from app.verification.models import VerificationRequest, VerificationStatus

from .metrics import aggregate_scores, by_type_breakdown, compute_item_scores
from .models import BenchmarkItem, BenchmarkRunOutput, CorpusContext

DEFAULT_QUESTION_PATH = Path(__file__).resolve().parent / "data" / "questions_v1.json"

# Pipeline callable surface used by the runner.
Pipeline = Callable[[BenchmarkItem, CorpusContext], Awaitable[BenchmarkRunOutput]]


# ---------------------------------------------------------------- data loading

def load_items(path: Path = DEFAULT_QUESTION_PATH) -> list[BenchmarkItem]:
    """Load the benchmark question set from JSON (100 items + adversarial cases)."""
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    items: list[BenchmarkItem] = []
    for raw in payload.get("items", []):
        items.append(
            BenchmarkItem(
                id=str(raw["id"]),
                type=str(raw["type"]),
                question=str(raw["question"]),
                gold_answer=str(raw["gold_answer"]),
                gold_evidence=[str(p) for p in raw.get("gold_evidence", [])],
                gold_years=[str(y) for y in raw.get("gold_years", [])],
                expect_contradiction=bool(raw.get("expect_contradiction", False)),
            )
        )
    for raw in payload.get("adversarial_cases", []):
        items.append(
            BenchmarkItem(
                id=str(raw["id"]),
                type="adversarial",
                question=str(raw["question"]),
                gold_answer=str(raw["gold_answer"]),
                gold_evidence=[str(p) for p in raw.get("gold_evidence", [])],
                expect_contradiction=bool(raw.get("expect_contradiction", False)),
                adversarial_type=str(raw.get("adversarial_type", "near_duplicate_incorrect")),
                distractor_evidence=[str(p) for p in raw.get("distractor_evidence", [])],
            )
        )
    return items


# ---------------------------------------------------------------- corpus build

def _checksum(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def build_corpus(
    items: list[BenchmarkItem],
    working_dir: Path,
    *,
    settings: Settings | None = None,
) -> CorpusContext:
    """Ingest the question set's evidence passages into a temp evidence store.

    Each gold/distractor passage becomes one chunk of a dedicated document.
    Returns the chunk-id maps the scoring layer needs, ready for retrieval.
    """
    settings = settings or get_settings()
    start = time.monotonic()
    working_dir.mkdir(parents=True, exist_ok=True)
    store = EvidenceStore(
        db_path=working_dir / "evidence.db",
        bm25_index_path=working_dir / "indexes" / "bm25.pkl",
        faiss_index_path=working_dir / "indexes" / "faiss.index",
    )

    gold_ids: dict[str, list[str]] = {}
    distractor_ids: dict[str, list[str]] = {}
    chunk_text_by_id: dict[str, str] = {}
    item_meta: dict[str, dict[str, Any]] = {}

    # Content-addressed corpus: one document/chunk per unique passage text,
    # shared by every item that references the same passage (this mirrors a
    # real knowledge base and keeps gold-chunk ids stable across items).
    passage_to_chunk: dict[str, str] = {}

    def _ingest(passage: str, path: str, item_id: str) -> str:
        checksum = _checksum(passage)
        chunk_id = passage_to_chunk.get(passage)
        if chunk_id is None:
            source = store.upsert_source(
                Source(type=SourceType.TEXT, path=path, checksum=checksum)
            )
            doc = store.insert_document(
                Document(source_id=source.id, version=1, checksum=checksum, chunking_strategy="benchmark")
            )
            chunk = store.insert_chunks(
                [Chunk(document_id=doc.id, ordinal=0, text=passage, token_count=len(passage.split()))]
            )[0]
            chunk_id = str(chunk.id)
            passage_to_chunk[passage] = chunk_id
            chunk_text_by_id[chunk_id] = passage
            item_meta.setdefault(item_id, {}).setdefault("sources", []).append(str(source.path))
        return chunk_id

    for item in items:
        gold_ids[item.id] = []
        distractor_ids[item.id] = []
        for i, passage in enumerate(item.gold_evidence or [], start=1):
            gold_ids[item.id].append(
                _ingest(passage, f"bench/{item.id}/gold-{i}", item.id)
            )
        for j, passage in enumerate(item.distractor_evidence or [], start=1):
            distractor_ids[item.id].append(
                _ingest(passage, f"bench/{item.id}/distractor-{j}", item.id)
            )

    # Build the retrieval index over the full corpus (gold + distractors).
    bm25 = BM25Retriever(store, index_path=store.bm25_index_path)
    vector = FAISSVectorStore(store, index_path=store.faiss_index_path)
    retriever = HybridRetriever(store, bm25=bm25, vector=vector, embedder=EmbeddingGenerator())
    retriever.ensure_indexes()

    ctx = CorpusContext(
        gold_chunk_ids=gold_ids,
        distractor_chunk_ids=distractor_ids,
        chunk_text_by_id=chunk_text_by_id,
        item_meta=item_meta,
        build_duration_ms=int((time.monotonic() - start) * 1000),
    )
    ctx._store = store  # type: ignore[attr-defined]
    ctx._retriever = retriever  # type: ignore[attr-defined]
    return ctx


def default_sources(corpus: CorpusContext) -> dict[str, Any]:
    """Resolve the corpus's evidence store, retriever, and graph store for pipelines."""
    from app.graph.store import EvidenceGraphStore

    store: EvidenceStore = corpus._store  # type: ignore[attr-defined]
    retriever: HybridRetriever = corpus._retriever  # type: ignore[attr-defined]
    graph_store = EvidenceGraphStore(
        graph_path=store.db_path.with_suffix(".graph.pkl"),
        evidence_store=store,
    )
    return {"store": store, "retriever": retriever, "graph_store": graph_store}


# ---------------------------------------------------------------- pipelines

async def _run_query_safe(*, query: str, **kwargs: Any) -> Any:
    from app.orchestration.graph import run_query

    return await run_query(query, **kwargs)


def make_full_argus_pipeline(
    *,
    router: Any,
    evidence_store: EvidenceStore,
    graph_store: Any,
    retriever: HybridRetriever,
    reranker: Any | None = None,
    settings: Settings | None = None,
    verify_answer: bool = True,
) -> Pipeline:
    """Full ARGUS pipeline: Phase 02 loop + Phase 04 verification (V2 §13.2 'Full ARGUS')."""

    from app.reranking.reranker import NoOpReranker

    settings = settings or get_settings()
    reranker = reranker or NoOpReranker()

    async def pipeline(item: BenchmarkItem, corpus: CorpusContext) -> BenchmarkRunOutput:
        t0 = time.monotonic()
        retrieved = retriever.search(item.question, top_k=10)
        retrieved_ids = [str(r.chunk_id) for r in retrieved]

        start_run_telemetry(
            call_ceiling=settings.multimodel_call_ceiling,
            run_id=f"bench:{item.id}",
        )
        try:
            result = await _run_query_safe(
                query=item.question,
                request_id=f"bench:{item.id}",
                router=router,
                retriever=retriever,
                reranker=reranker,
                settings=settings,
            )
        finally:
            summary = end_run_telemetry() or {}

        cited_ids = [str(c.chunk_id) for c in result.citations]
        status = None
        contradiction = False
        if verify_answer and result.answer:
            try:
                verifier = await verify_claim(
                    VerificationRequest(
                        claim_id=uuid.uuid4(),
                        claim_text=result.answer,
                        supporting_chunk_ids=[uuid.UUID(c) for c in cited_ids],
                        temporal_context=result.plan.time_window or None,
                        entity_names=list(result.plan.entities),
                    ),
                    router=router,
                    evidence_store=evidence_store,
                    graph_store=graph_store,
                    settings=settings,
                    request_id=result.request_id,
                )
                status = verifier.status.value
                contradiction = verifier.status == VerificationStatus.CONTRADICTED or bool(
                    verifier.contradictions
                )
            except Exception:  # noqa: BLE001 - verification failure degrades to ERROR status
                status = VerificationStatus.ERROR.value

        return BenchmarkRunOutput(
            item_id=item.id,
            answer=result.answer,
            cited_chunk_ids=cited_ids,
            retrieved_chunk_ids=retrieved_ids,
            loop_count=result.iterations_used,
            tokens_used=result.token_usage_estimate,
            latency_ms=int((time.monotonic() - t0) * 1000),
            failed_calls=int(summary.get("failed_calls", 0)),
            verification_status=status,
            contradiction_detected=contradiction,
            warned=list(result.warnings),
            metadata={"stop_reason": result.stop_reason.value},
        )

    return pipeline


# ---------------------------------------------------------------- scoring & reports

def score_items(
    items: list[BenchmarkItem],
    outputs: list[BenchmarkRunOutput],
    corpus: CorpusContext,
) -> dict[str, Any]:
    """Score each run and return per-item records plus the aggregate report data."""
    per_item: list[tuple[BenchmarkItem, dict[str, float]]] = []
    records: list[dict[str, Any]] = []
    for item, output in zip(items, outputs):
        gold = set(corpus.gold_chunk_ids.get(item.id, []))
        distractors = set(corpus.distractor_chunk_ids.get(item.id, []))
        scores = compute_item_scores(item, output, gold, distractors, corpus.chunk_text_by_id)
        per_item.append((item, scores))
        records.append(
            {
                "item_id": item.id,
                "type": item.type,
                "adversarial": item.adversarial_type,
                "warnings": output.warned,
                **{k: (None if math.isnan(v) else round(float(v), 4)) for k, v in scores.items()},
                "answer": output.answer[:400],
                "verification_status": output.verification_status,
            }
        )
    return {"metrics": aggregate_scores(per_item, outputs), "by_type": by_type_breakdown(per_item), "per_item": records}


def build_report(
    *,
    name: str,
    items: list[BenchmarkItem],
    outputs: list[BenchmarkRunOutput],
    corpus: CorpusContext,
    pipeline_label: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble the full benchmark report (JSON-serializable)."""
    scored = score_items(items, outputs, corpus)
    return {
        "name": name,
        "generated_at": datetime.now(UTC).isoformat(),
        "pipeline": pipeline_label,
        "item_count": len(items),
        "type_counts": {t: sum(1 for i in items if i.type == t) for t in sorted({i.type for i in items})},
        "corpus": {
            "build_duration_ms": corpus.build_duration_ms,
            "gold_chunks": sum(len(v) for v in corpus.gold_chunk_ids.values()),
        },
        **scored,
        "v3": {
            "vault_personalization_gain": None,  # computed only when a live vault is mounted
            "reindex_cost_ms": corpus.build_duration_ms,
            "write_back_usefulness": None,
        },
        "extra": extra or {},
    }


def write_reports(report: dict[str, Any], out_dir: Path) -> tuple[Path, Path]:
    """Write JSON + Markdown report files; returns their paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "benchmark_report.json"
    md_path = out_dir / "benchmark_report.md"
    json_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    md_path.write_text(markdown_report(report), encoding="utf-8")
    return json_path, md_path


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        f = float(value)
    except (TypeError, ValueError):
        return str(value)
    return "n/a" if math.isnan(f) else f"{f:.4f}"


def markdown_report(report: dict[str, Any]) -> str:
    """Render the aggregate report as a Markdown table (V2 §13 metric surface)."""
    lines: list[str] = [
        f"# {report['name']}",
        "",
        f"- pipeline: `{report['pipeline']}`",
        f"- generated: {report['generated_at']}",
        f"- items: {report['item_count']}",
        "",
        "## Metrics",
        "",
        "| Metric | value | applicable |",
        "|---|---|---|",
    ]
    for name, info in sorted((report.get("metrics") or {}).items()):
        lines.append(f"| {name} | {_fmt(info.get('value'))} | {info.get('applicable')} |")
    lines += ["", "## By question type (headline)", "", "| type | recall@10 | evidence precision | faithfulness |", "|---|---|---|---|"]
    for t, vals in sorted((report.get("by_type") or {}).items()):
        lines.append(
            f"| {t} | {_fmt(vals.get('recall_at_10'))} | {_fmt(vals.get('evidence_precision'))} | {_fmt(vals.get('answer_faithfulness'))} |"
        )
    v3 = report.get("v3") or {}
    lines += [
        "",
        "## V3-specific",
        "",
        f"- vault personalization gain: {v3.get('vault_personalization_gain')}",
        f"- reindex cost: {v3.get('reindex_cost_ms')} ms",
        f"- write-back usefulness: {v3.get('write_back_usefulness')}",
    ]
    return "\n".join(lines) + "\n"


async def run_benchmark(
    *,
    pipeline: Pipeline,
    question_path: Path = DEFAULT_QUESTION_PATH,
    limit: int | None = None,
    working_dir: Path | None = None,
    out_dir: Path | None = None,
    name: str = "ARGUS benchmark v1",
    pipeline_label: str = "full_argus",
) -> dict[str, Any]:
    """End-to-end: load items → build corpus → run → score → report.

    `working_dir` defaults to a temp dir (the corpus is disposable; the real
    KB/corpus is never touched). `out_dir` is where reports land.
    """
    items = load_items(question_path)
    if limit is not None:
        items = items[:limit]

    if working_dir is None:
        work = Path(tempfile.mkdtemp(prefix="argus_bench_"))
    else:
        work = working_dir
    corpus = build_corpus(items, work)

    outputs: list[BenchmarkRunOutput] = []
    for item in items:
        outputs.append(await pipeline(item, corpus))

    report = build_report(
        name=name,
        items=items,
        outputs=outputs,
        corpus=corpus,
        pipeline_label=pipeline_label,
    )
    if out_dir is not None:
        write_reports(report, out_dir)
    return report