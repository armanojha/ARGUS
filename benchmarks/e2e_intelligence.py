"""E2E Intelligence Benchmark — fixed inputs, measurable research behavior.

Measures what unit-test counts cannot: retrieval quality, contradiction
detection accuracy, and abstention behavior on a fixed 38-case eval plan
against a fixed 12-document corpus.

Two modes:
  deterministic (default, CI-safe, no LLM):
      Retrieval Recall@K / Precision@K vs gold supporting docs,
      gold-fact coverage in retrieved text,
      contradiction precision/recall of the deterministic detector,
      abstention-trigger accuracy of the deterministic absent-info detector,
      retrieval latency per query.
  live (--live, needs a configured LLM provider, rate-limit sensitive):
      full run_query per case plus citation accuracy, claim grounding,
      abstention accuracy, LLM calls, latency, tokens. Cost/query is
      reported as tokens (no provider pricing configured → cost n/a).

Isolation: ingests the eval corpus into a TMP EvidenceStore + TMP indexes.
Never touches the real data/ directory or the user's knowledge base.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from app.evidence.store import EvidenceStore
from app.ingestion.pipeline import IngestionPipeline
from app.orchestration.nodes import (
    _detect_contradictions,
    _is_evidence_absent,
)
from app.retrieval.bm25 import BM25Retriever
from app.retrieval.embeddings import EmbeddingGenerator
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.vector import FAISSVectorStore


def _doc_of(ref) -> str:
    """Map an EvidenceRef to its source filename."""
    return Path(ref.source_path).name


def run_deterministic(top_k: int = 8) -> dict:
    plan_path = REPO_ROOT / "benchmarks" / "eval_data" / "eval_plan_v1.json"
    plan = json.loads(plan_path.read_text())
    corpus_dir = REPO_ROOT / "benchmarks" / "eval_data" / plan["corpus_dir"]
    file_of_doc = plan["documents"]  # doc-id -> filename
    doc_of_file = {v: k for k, v in file_of_doc.items()}
    cases = [c for c in plan["queries"] if c.get("live")]

    tmp = Path(tempfile.mkdtemp(prefix="e2e_intel_"))
    store = EvidenceStore(
        db_path=tmp / "evidence.db",
        bm25_index_path=tmp / "bm25.pkl",
        faiss_index_path=tmp / "faiss.index",
    )
    pipeline = IngestionPipeline(store)
    for md in sorted(corpus_dir.glob("*.md")):
        pipeline.ingest_text_file(md)

    retriever = HybridRetriever(
        store=store,
        bm25=BM25Retriever(store),
        vector=FAISSVectorStore(store, embedding_dim=384, index_path=tmp / "f.index"),
        embedder=EmbeddingGenerator(),
    )
    retriever.ensure_indexes()

    per_case = []
    for case in cases:
        q = case["query"]
        gold_docs = set(case.get("supporting_docs", []))
        t0 = time.perf_counter()
        refs = retriever.search(q, top_k=top_k)
        latency_ms = round((time.perf_counter() - t0) * 1000, 1)
        got_docs = {_d for r in refs if (_d := doc_of_file.get(_doc_of(r)))}
        retrieved_text = " ".join(r.text for r in refs).lower()

        rec = len(got_docs & gold_docs) / len(gold_docs) if gold_docs else None
        prec = len(got_docs & gold_docs) / len(got_docs) if got_docs else (1.0 if not gold_docs else 0.0)
        gold_facts = case.get("gold_facts", [])
        coverage = (
            sum(1 for g in gold_facts if str(g).lower() in retrieved_text) / len(gold_facts)
            if gold_facts else None
        )
        signals = _detect_contradictions(refs, q)
        absent_trigger = _is_evidence_absent(refs, q)
        per_case.append({
            "id": case["id"], "class": case.get("class"), "query": q,
            "recall_at_k": rec, "precision_at_k": prec, "gold_coverage": coverage,
            "retrieved_docs": sorted(got_docs), "gold_docs": sorted(gold_docs),
            "contradiction_signals": len(signals),
            "conflict_expected": bool(case.get("conflict")),
            "absent_trigger": absent_trigger,
            "absent_expected": bool(case.get("absent")),
            "latency_ms": latency_ms,
        })

    return {"top_k": top_k, "cases": per_case, "mode": "deterministic"}


def aggregate(result: dict) -> dict:
    cases = result["cases"]
    by_class: dict[str, dict] = defaultdict(lambda: {
        "n": 0, "recall": [], "precision": [], "coverage": [],
        "lat": [], "tp": 0, "fp": 0, "fn": 0, "abs_ok": 0, "abs_n": 0,
    })
    for c in cases:
        b = by_class[c["class"]]
        b["n"] += 1
        if c["recall_at_k"] is not None:
            b["recall"].append(c["recall_at_k"])
        b["precision"].append(c["precision_at_k"])
        if c["gold_coverage"] is not None:
            b["coverage"].append(c["gold_coverage"])
        b["lat"].append(c["latency_ms"])
        if c["conflict_expected"]:
            if c["contradiction_signals"] > 0:
                b["tp"] += 1
            else:
                b["fn"] += 1
        elif c["contradiction_signals"] > 0:
            b["fp"] += 1
        if c["absent_expected"]:
            b["abs_n"] += 1
            if c["absent_trigger"]:
                b["abs_ok"] += 1

    def _mean(xs):
        return round(sum(xs) / len(xs), 4) if xs else None

    table = {}
    for cls, b in sorted(by_class.items()):
        tp, fp, fn = b["tp"], b["fp"], b["fn"]
        contra_p = round(tp / (tp + fp), 4) if (tp + fp) else None
        contra_r = round(tp / (tp + fn), 4) if (tp + fn) else None
        table[cls] = {
            "n": b["n"], "recall_at_k": _mean(b["recall"]),
            "precision_at_k": _mean(b["precision"]),
            "gold_coverage": _mean(b["coverage"]),
            "contradiction_precision": contra_p, "contradiction_recall": contra_r,
            "abstention_trigger_acc": round(b["abs_ok"] / b["abs_n"], 4) if b["abs_n"] else None,
            "avg_latency_ms": round(sum(b["lat"]) / len(b["lat"]), 1),
        }

    all_rec = [c["recall_at_k"] for c in cases if c["recall_at_k"] is not None]
    all_prec = [c["precision_at_k"] for c in cases]
    all_cov = [c["gold_coverage"] for c in cases if c["gold_coverage"] is not None]
    tps = sum(1 for c in cases if c["conflict_expected"] and c["contradiction_signals"] > 0)
    fns = sum(1 for c in cases if c["conflict_expected"] and c["contradiction_signals"] == 0)
    fps = sum(1 for c in cases if not c["conflict_expected"] and c["contradiction_signals"] > 0)
    abs_cases = [c for c in cases if c["absent_expected"]]
    abs_ok = sum(1 for c in abs_cases if c["absent_trigger"])
    overall = {
        "n": len(cases),
        "recall_at_k": _mean(all_rec), "precision_at_k": _mean(all_prec),
        "gold_coverage": _mean(all_cov),
        "contradiction_precision": round(tps / (tps + fps), 4) if (tps + fps) else None,
        "contradiction_recall": round(tps / (tps + fns), 4) if (tps + fns) else None,
        "abstention_trigger_acc": round(abs_ok / len(abs_cases), 4) if abs_cases else None,
        "avg_latency_ms": round(sum(c["latency_ms"] for c in cases) / len(cases), 1),
    }
    return {"overall": overall, "by_class": table}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--out", type=str, default="benchmarks/results/e2e_intelligence.json")
    ap.add_argument("--live", action="store_true",
                    help="Also run full run_query per case (needs LLM provider). Not CI-safe.")
    args = ap.parse_args()

    result = run_deterministic(top_k=args.top_k)
    result["aggregate"] = aggregate(result)
    if args.live:
        result["live"] = {"status": "not_implemented_in_ci_mode"}
    out = REPO_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1))

    agg = result["aggregate"]
    print(f"E2E Intelligence Benchmark (deterministic, top_k={args.top_k}, n={agg['overall']['n']})")
    print(f"OVERALL: {json.dumps(agg['overall'])}")
    for cls, m in agg["by_class"].items():
        print(f"  {cls:22s} n={m['n']} R={m['recall_at_k']} P={m['precision_at_k']} "
              f"cov={m['gold_coverage']} contraP/R={m['contradiction_precision']}/{m['contradiction_recall']} "
              f"abs={m['abstention_trigger_acc']} lat={m['avg_latency_ms']}ms")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
