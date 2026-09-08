"""Reasoning-trace materialization (run trace → reasoning graph).

The knowledge graph stores WHAT the corpus contains (entities, claims,
events). The reasoning trace records WHY a specific answer emerged:
query → tasks → hypothesis → evidence → claims → inference → decision →
answer, plus rejected counterclaims. It is derived purely from a finished
:class:`OrchestrationResult` — no LLM calls, no store access, fully
deterministic — so the Brain UI can render "why did ARGUS conclude this"
as a traversable graph instead of asserting it.
"""

from __future__ import annotations

import re
from typing import Any

_CIT_RE = re.compile(r"\[(\d+)\]")
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])")

NODE_TYPES = (
    "QUERY",
    "RESEARCH_TASK",
    "HYPOTHESIS",
    "EVIDENCE",
    "CLAIM",
    "COUNTERCLAIM",
    "INFERENCE",
    "DECISION",
    "ANSWER",
)


def _sentences(text: str) -> list[str]:
    parts = _SENT_SPLIT_RE.split((text or "").strip())
    return [p.strip() for p in parts if p.strip()]


def build_reasoning_trace(result: dict[str, Any]) -> dict[str, Any]:
    """Build a reasoning graph from a finished orchestration result.

    Accepts an ``OrchestrationResult`` (or its ``model_dump()`` dict).
    Never raises on missing/empty fields — degenerate runs still yield a
    QUERY/HYPOTHESIS/DECISION/ANSWER skeleton. Uncited answer sentences
    never become CLAIM nodes (no fabricated grounding).
    """
    if not isinstance(result, dict):
        result = result.model_dump(mode="json")  # type: ignore[union-attr]

    query = str(result.get("query") or "")
    plan = result.get("plan") or {}
    answer = str(result.get("answer") or "")
    citations = list(result.get("citations") or [])
    sub_queries = list(result.get("sub_queries_issued") or [])
    signals = list(result.get("contradiction_signals") or [])
    verification = result.get("verification") or {}

    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []

    def _add(node_id: str, node_type: str, label: str, **data: Any) -> None:
        nodes.append({"id": node_id, "node_type": node_type, "label": label, **data})

    def _edge(source: str, target: str, edge_type: str, **data: Any) -> None:
        edges.append({"source": source, "target": target, "edge_type": edge_type, **data})

    # QUERY
    _add("query:0", "QUERY", query[:120] or "(empty query)", text=query)

    # HYPOTHESIS (plan objective; entities/time as context)
    objective = str(plan.get("objective") or query)
    _add("hypothesis:0", "HYPOTHESIS", objective[:120],
         entities=list(plan.get("entities") or []),
         time_window=plan.get("time_window"))
    _edge("query:0", "hypothesis:0", "poses")

    # RESEARCH_TASKs (issued subqueries preferred; fall back to plan list)
    tasks: list[str] = sub_queries or list(plan.get("subquestions") or [])
    for i, tq in enumerate(tasks):
        tid = f"task:{i}"
        _add(tid, "RESEARCH_TASK", str(tq)[:120], text=str(tq))
        _edge("query:0", tid, "decomposes_to")
        _edge(tid, "hypothesis:0", "tests")

    # EVIDENCE (citations; ref_id is the 1-based citation index)
    ev_ids: dict[int, str] = {}
    for cit in citations:
        ref_id = int(cit.get("ref_id", 0) or 0)
        eid = f"evidence:{ref_id}"
        ev_ids[ref_id] = eid
        _add(eid, "EVIDENCE",
             str(cit.get("text") or "")[:120],
             text=str(cit.get("text") or ""),
             source_path=cit.get("source_path"),
             score=cit.get("score"),
             section_path=cit.get("section_path"))
        _edge(eid, "query:0", "retrieved_because")

    # CLAIMs (only answer sentences carrying [n] citations)
    claim_idx = 0
    for sent in _sentences(answer):
        refs = sorted({int(n) for n in _CIT_RE.findall(sent)})
        if not refs:
            continue
        cid = f"claim:{claim_idx}"
        claim_idx += 1
        _add(cid, "CLAIM", sent[:140], text=sent, evidence_refs=refs)
        for n in refs:
            if n in ev_ids:
                _edge(cid, ev_ids[n], "supported_by")

    # COUNTERCLAIMs (contradiction signals; rejected, not chosen)
    for i, sig in enumerate(signals):
        if not isinstance(sig, dict):
            continue
        cid = f"counter:{i}"
        _add(cid, "COUNTERCLAIM",
             str(sig.get("description") or sig.get("conflict_type") or "conflict")[:140],
             conflict_type=sig.get("conflict_type") or sig.get("type"),
             confidence=sig.get("confidence"),
             semantic_verified=sig.get("semantic_verified"))
        for n in sig.get("evidence_indices", []) or []:
            try:
                ni = int(n)
            except (TypeError, ValueError):
                continue
            if ni in ev_ids:
                _edge(cid, ev_ids[ni], "contradicted_by")

    # INFERENCE (what the loop concluded from the evidence)
    stop_reason = result.get("stop_reason")
    outcome = result.get("outcome")
    ver_status = verification.get("status") if isinstance(verification, dict) else None
    inference_text = (
        f"Evidence judged sufficient ({stop_reason}); "
        f"outcome {outcome}"
        + (f"; verification {ver_status}" if ver_status else "")
    )
    _add("inference:0", "INFERENCE", inference_text[:140],
         stop_reason=stop_reason, outcome=outcome,
         iterations_used=result.get("iterations_used"))
    for n in range(claim_idx):
        _edge("inference:0", f"claim:{n}", "produces")

    # DECISION
    _add("decision:0", "DECISION",
         f"{stop_reason} → {outcome}"[:140],
         stop_reason=stop_reason, outcome=outcome,
         stop_condition=result.get("stop_condition"))
    _edge("inference:0", "decision:0", "leads_to")

    # ANSWER
    _add("answer:0", "ANSWER", answer[:140], text=answer,
         outcome=outcome, warnings=list(result.get("warnings") or []))
    _edge("decision:0", "answer:0", "produces")
    _edge("hypothesis:0", "answer:0", "answered_by")
    for n in range(claim_idx):
        _edge("answer:0", f"claim:{n}", "supported_by")
    for i in range(len([s for s in signals if isinstance(s, dict)])):
        _edge("answer:0", f"counter:{i}", "rejected_alternative")

    return {"nodes": nodes, "edges": edges}
