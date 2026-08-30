"""ARGUS Evidence Explorer (Phase 12.1) — Streamlit MVP UI.

Renders a live research run transparently:
  - research plan
  - evidence (citations)
  - conflicts (verifier contradictions)
  - source trail
  - verifier result
  - loop count / stop reason
  - confidence

The UI is a presentation/control layer only: everything it shows comes
from the existing ARGUS API, and every backend call (`/api/v1/query`,
`/api/v1/verify`) runs the existing orchestration and verification
services. React upgrade path noted but not required (V3 §18).

Run:
    uvicorn app.api.main:app --reload        # API (separate terminal)
    streamlit run app/ui/streamlit_app.py    # this UI
"""

from __future__ import annotations

from typing import Any

import streamlit as st

from app.ui.api_client import ARGUSAPIClient, ARGUSAPIClientError

st.set_page_config(page_title="ARGUS Evidence Explorer", page_icon=":material/science:", layout="wide")

# React upgrade path (V3 §18): a future React client can reuse the exact
# same `/api/v1/query` + `/api/v1/verify` contract; nothing here is
# Streamlit-specific except the rendering layer.


def _fail(text: str) -> None:
    st.error(text)


def _section(title: str) -> None:
    st.subheader(title)


def _render_plan(plan: dict[str, Any]) -> None:
    """Research plan (V2 §5.1)."""
    _section("Research Plan")
    st.markdown(f"**Objective:** {plan.get('objective') or '—'}")
    cols = st.columns(3)
    cols[0].metric("Risk", plan.get("risk_level") or "low")
    cols[0].metric("Evidence type", plan.get("evidence_type"))
    cols[1].metric("Token budget", plan.get("token_budget"))
    cols[1].metric("Iterations budget", plan.get("iteration_budget"))
    cols[2].metric("Time window", plan.get("time_window") or "n/a")

    if plan.get("entities"):
        st.markdown("**Entities:** " + ", ".join(plan["entities"]))
    if plan.get("subquestions"):
        st.markdown("**Sub-questions:**")
        for i, sq in enumerate(plan["subquestions"], 1):
            st.markdown(f"{i}. {sq}")
    if plan.get("preferred_retrieval_methods"):
        st.markdown("**Preferred retrieval:** " + ", ".join(plan["preferred_retrieval_methods"]))
    if plan.get("stopping_condition"):
        st.caption(f"Stopping condition: {plan['stopping_condition']}")


def _render_loop_stats(result: dict[str, Any]) -> None:
    """Loop count, stop reason, sub-queries, warnings."""
    _section("Run Summary")
    cols = st.columns(4)
    cols[0].metric("Loop count", result.get("iterations_used"))
    cols[1].metric("Stop reason", result.get("stop_reason") or result.get("stop_condition") or "n/a")
    cols[2].metric("Citations", len(result.get("citations") or []))
    cols[3].metric("Token estimate", result.get("token_usage_estimate"))

    if result.get("question_pattern"):
        st.caption(f"Question pattern: {result['question_pattern']}")
    if result.get("agent_round") is not None:
        st.caption(f"Multi-agent debate rounds: {result['agent_round']}  |  disagreement: {result.get('disagreement_detected')}")
    if result.get("evidence_tasks"):
        st.caption(f"Evidence-seeking tasks: {len(result['evidence_tasks'])}")
    if result.get("warnings"):
        st.warning("; ".join(result["warnings"]))


def _render_evidence(citations: list[dict[str, Any]]) -> None:
    """Evidence chunks returned by the loop (citations)."""
    _section("Evidence")
    if not citations:
        st.info("No evidence was retrieved. Check the API corpus has been ingested.")
        return
    for cit in citations:
        loc = ""
        if cit.get("page_start"):
            loc += f"p{int(cit['page_start'])}"
        if cit.get("section_path"):
            loc += (", " if loc else "") + str(cit["section_path"])
        title = f"[{cit.get('ref_id')}] {cit.get('source_path')} — score {cit.get('score', 0):.3f}"
        if loc:
            title += f" ({loc})"
        with st.expander(title):
            st.markdown(cit.get("text") or "")
            meta = cit.get("metadata") or {}
            if meta:
                st.caption("Metadata: " + ", ".join(f"{k}={v}" for k, v in meta.items()))


def _render_source_trail(citations: list[dict[str, Any]]) -> None:
    """Source trail: unique sources actually cited, in citation order."""
    seen: list[str] = []
    for cit in citations or []:
        path = cit.get("source_path")
        if path and path not in seen:
            seen.append(path)
    _section("Source Trail")
    if not seen:
        st.caption("No sources yet.")
        return
    for i, path in enumerate(seen, 1):
        st.markdown(f"{i}. `{path}`")


def _render_verification(verification: dict[str, Any]) -> None:
    """Verifier result: status, confidence, conflicts, supporting/contradicting evidence."""
    _section("Verifier Result")
    status = verification.get("status", "error")
    confidence = verification.get("confidence", 0.0)
    codemap = {
        "supported": ("success", "Supported"),
        "partial": ("warning", "Partially supported"),
        "contradicted": ("error", "Contradicted"),
        "unsupported": ("info", "Unsupported"),
        "error": ("error", "Verifier unavailable"),
    }
    kind, label = codemap.get(status, ("info", status))
    cols = st.columns(4)
    cols[0].metric("Status", label)
    cols[0].markdown(f"[{kind}]")
    cols[1].metric("Confidence", f"{confidence:.2f}")
    cols[2].metric("Contradictions", len(verification.get("contradictions") or []))
    cols[3].metric("Evidence used", len(verification.get("supporting_evidence") or []) + len(verification.get("contradicting_evidence") or []))

    if verification.get("reasoning"):
        st.markdown(f"**Reasoning:** {verification['reasoning']}")

    _render_confidence_breakdown(verification)

    conflicts = verification.get("contradictions") or []
    if conflicts:
        st.markdown("**Conflicts detected:**")
        for c in conflicts:
            st.warning(
                f"{c.get('contradiction_type', 'source_conflict')} — {c.get('description')}"
                f"{'  (severity {:.2f})'.format(c.get('severity', 0))}"
            )
            if c.get("resolution_suggestion"):
                st.caption(f"Suggestion: {c['resolution_suggestion']}")
    else:
        st.caption("No contradictions detected.")

    if verification.get("supporting_evidence"):
        with st.expander(f"Supporting evidence ({len(verification['supporting_evidence'])})"):
            for ev in verification["supporting_evidence"]:
                st.markdown(f"- `{ev.get('source_path')}` — {ev.get('text', '')[:300]}")
    if verification.get("contradicting_evidence"):
        with st.expander(f"Contradicting evidence ({len(verification['contradicting_evidence'])})"):
            for ev in verification["contradicting_evidence"]:
                st.markdown(f"- `{ev.get('source_path')}` — {ev.get('text', '')[:300]}")


def _render_confidence_breakdown(verification: dict[str, Any]) -> None:
    """Diagnostic confidence components (V2 §9.3), when present."""
    comp = [
        ("Evidence coverage", verification.get("evidence_coverage")),
        ("Source quality", verification.get("source_quality")),
        ("Cross-source agreement", verification.get("cross_source_agreement")),
        ("Temporal relevance", verification.get("temporal_relevance")),
        ("Retrieval rank", verification.get("retrieval_rank")),
        ("Verifier judgment", verification.get("verifier_judgment")),
    ]
    present = [(k, v) for k, v in comp if v is not None]
    if present:
        with st.expander("Confidence components"):
            st.progress(verification.get("confidence", 0.0), text=f"Overall confidence {verification.get('confidence', 0.0):.2f}")
            for label, value in present:
                st.markdown(f"{label}: {value:.2f}")


def _render_telemetry(telemetry: dict[str, Any] | None) -> None:
    """Phase 12.2 run trace: latency, tokens, provider/model, failures."""
    if not telemetry:
        return
    _section("Run Trace")
    cols = st.columns(4)
    cols[0].metric("Total calls", telemetry.get("total_calls"))
    cols[1].metric("Failed calls", telemetry.get("failed_calls"))
    cols[2].metric("Tokens", telemetry.get("total_tokens"))
    cols[3].metric("Duration", f"{telemetry.get('duration_ms', 0)} ms")
    decisions = telemetry.get("routing_decisions") or []
    if decisions:
        with st.expander(f"Routing decisions ({len(decisions)})"):
            for d in decisions:
                st.markdown(
                    f"- `{d.get('call_type')}` → `{d.get('provider')}/{d.get('model')}` "
                    f"ok={d.get('success')} {d.get('latency_ms')} ms "
                    f"{('err=' + str(d.get('error_code'))) if not d.get('success') else ''}"
                )


def _render_run_traces(client: ARGUSAPIClient, limit: int = 10) -> None:
    """Phase 12.2: recent telemetry runs surfaced from the API."""
    try:
        runs = client.list_run_traces(limit=limit)
    except ARGUSAPIClientError:
        return
    if not runs:
        return
    _section("Recent Run Traces")
    for r in runs:
        with st.expander(f"{r.get('run_id')} — {r.get('duration_ms')} ms, {r.get('total_calls')} calls"):
            st.markdown(
                f"tokens={r.get('total_tokens')}, failed={r.get('failed_calls')}, "
                f"ceiling={r.get('call_ceiling')}"
            )


def run() -> None:
    """Evidence explorer main entry point."""
    st.title("ARGUS Evidence Explorer")
    st.caption("Phase 12.1 — research process transparency over the existing ARGUS system.")

    with st.sidebar:
        st.header("Control")
        base_url = st.text_input("API base URL", value="http://localhost:8000")
        user_early_stop = st.checkbox("Early stop", help="Stop after current evidence (Phase 06 user_early_stop).")
        run_verification = st.checkbox("Verify answer", value=True, help="Run the Phase 04 verifier over the answer.")
        st.caption("Model selection stays explicit server-side via `configs/model_policy.yaml` — the UI never selects models.")

    query_text = st.text_input("Research question", key="query", placeholder="e.g. Which evidence supports the claim that Acme acquired Beta?")
    go = st.button("Run research", type="primary")

    if not go:
        st.info("Enter a question and press **Run research**. The API must be running first:")
        st.code("uvicorn app.api.main:app --reload", language="bash")
        return

    if not query_text.strip():
        _fail("Please enter a question.")
        return

    client = ARGUSAPIClient(base_url=base_url)

    with st.spinner("Running the research loop…"):
        try:
            result = client.query(query_text.strip(), user_early_stop=user_early_stop)
        except ARGUSAPIClientError as exc:
            _fail(f"API call failed — is the API running? {exc}")
            return

    _render_loop_stats(result)
    _render_telemetry(result.get("telemetry"))
    _render_plan(result.get("plan") or {})
    _render_evidence(result.get("citations") or [])
    _render_source_trail(result.get("citations") or [])

    if run_verification:
        citations = result.get("citations") or []
        claim_text = result.get("answer") or result.get("query") or ""
        with st.spinner("Verifying the answer against evidence…"):
            try:
                verification = client.verify(
                    claim_text,
                    [str(c.get("chunk_id")) for c in citations],
                    entity_names=(result.get("plan") or {}).get("entities") or [],
                    temporal_context=(result.get("plan") or {}).get("time_window"),
                )
            except ARGUSAPIClientError as exc:
                _fail(f"Verification failed: {exc}")
                verification = None
        if verification is not None:
            _render_verification(verification)

    _render_run_traces(client)


run()