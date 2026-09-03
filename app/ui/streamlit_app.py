"""ARGUS Knowledge System UI (Streamlit) — the front door to all three layers.

The user-facing control layer over ARGUS, with four conceptual areas:

  * **Chat / Research** — ask questions, see grounded cited answers + sources,
    evidence, confidence, research process, and whether persistent memory was
    consulted (distinct from user-document evidence).
  * **Knowledge Base** — the user's document corpus (E:/KNOWLEDGE BASE): status,
    document/source/chunk counts, recently ingested documents, upload files,
    and a "resync" action. Uploads feed the SAME ingestion pipeline as the
    filesystem corpus.
  * **ARGUS Brain** — ARGUS's persistent machine memory: layer/promotion/
    scope counts, confidence, and recent memory records (provenance-preserving).
  * **Obsidian Brain** — ARGUS's dedicated Obsidian vault: path, note count,
    recent notes, and a selective memory->vault promotion action.

The UI is a pure presentation/control layer: every backend action runs the
existing ARGUS services (`/api/v1/query`, `/api/v1/verify`, and the new
knowledge-base / brain / obsidian endpoints). No business logic lives here.

Run:
    uvicorn app.api.main:app --reload        # API (separate terminal)
    streamlit run app/ui/streamlit_app.py    # this UI
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import streamlit as st

from app.ui.api_client import ARGUSAPIClient, ARGUSAPIClientError

st.set_page_config(page_title="ARGUS", page_icon=":material/science:", layout="wide")

# File types the modern chat attach accepts (mirrors the server's supported set).
_SUPPORTED_UI_EXTENSIONS: set[str] = {
    "pdf",
    "txt",
    "md",
    "markdown",
    "csv",
    "xlsx",
    "xls",
    "xlsm",
}

# React upgrade path (V3 §18): a future React client can reuse the exact same
# `/api/v1/query` + `/api/v1/verify` + knowledge-base/brain/obsidian contract;
# nothing here is Streamlit-specific except the rendering layer.


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


def _render_knowledge_base(client: ARGUSAPIClient) -> None:
    """Knowledge Base: user document corpus status / upload / resync."""
    _section("Knowledge Base")
    try:
        status = client.get_knowledge_base_status()
    except ARGUSAPIClientError as exc:
        _fail(f"Unable to reach the API: {exc}")
        return

    cols = st.columns(4)
    cols[0].metric("Documents", status.get("document_count"))
    cols[1].metric("Sources", status.get("source_count"))
    cols[2].metric("Chunks", status.get("chunk_count"))
    cols[3].metric("Indexed", "Yes" if status.get("indexed") else "No")

    st.markdown(f"**Corpus path:** `{status.get('knowledge_base_path')}`")
    st.caption("Supported types: " + (", ".join(status.get("supported_types") or [])))

    recents = status.get("recently_ingested") or []
    if recents:
        with st.expander(f"Recently ingested ({len(recents)})"):
            for r in recents:
                st.markdown(f"- `{r.get('source_path')}` — v{r.get('version')} · {r.get('created_at')}")

    with st.container(border=True):
        st.markdown("**Upload / re-sync**")
        up_c1, up_c2 = st.columns([2, 1])
        uploaded = up_c1.file_uploader(
            "Upload documents into the Knowledge Base",
            accept_multiple_files=True,
            key="kb_uploader",
        )
        up_c2.markdown("&nbsp;")
        up_c2.markdown("&nbsp;")
        upload_go = up_c2.button("Upload", key="kb_upload_go", type="primary")
        ingest_go = st.button("Resync knowledge base (idempotent)", key="kb_ingest_go")

    if upload_go and uploaded:
        # Persist uploads to a temp dir, then send paths to the upload endpoint.
        import tempfile

        file_paths: list[str] = []
        with tempfile.TemporaryDirectory() as tmpdir:
            for u in uploaded:
                dest = Path(tmpdir) / Path(u.name).name
                dest.write_bytes(u.getvalue())
                file_paths.append(str(dest))
            with st.spinner("Uploading and ingesting…"):
                try:
                    result = client.upload_files(file_paths)
                except ARGUSAPIClientError as exc:
                    _fail(f"Upload failed: {exc}")
                    result = None
        if result:
            ups = result.get("uploaded") or []
            rejs = result.get("rejected") or []
            st.success(f"Uploaded {len(ups)} document(s), rejected {len(rejs)}.")
            for r in rejs:
                st.warning(f"`{r.get('filename')}`: {r.get('reason')}")
            st.caption(f"Index refreshed: {result.get('indexed')} (chunks: {result.get('indexed_chunks')})")

    if ingest_go:
        with st.spinner("Re-syncing the corpus (no LLM calls)…"):
            try:
                result = client.ingest_knowledge_base()
            except ARGUSAPIClientError as exc:
                _fail(f"Resync failed: {exc}")
                result = None
        if result:
            st.success(
                f"Ingested {result.get('ingested')}, unchanged {result.get('unchanged')}, "
                f"errors {result.get('errors')} in {result.get('duration_s', 0):.1f}s."
            )
            for pth in result.get("error_paths") or []:
                st.warning(f"Error processing `{pth}`")


def _render_brain(client: ARGUSAPIClient) -> None:
    """ARGUS Brain: persistent machine memory status + recent records."""
    _section("ARGUS Brain")
    try:
        status = client.get_brain_status()
    except ARGUSAPIClientError as exc:
        _fail(f"Unable to reach the API: {exc}")
        return

    if not status.get("enabled"):
        st.info("ARGUS memory is disabled (`ARGUS_MEMORY_ENABLED=false`). No machine memory is being recorded.")
        return

    cols = st.columns(4)
    cols[0].metric("Memory records", status.get("total_records"))
    cols[1].metric("Avg confidence", f"{status.get('avg_confidence', 0.0):.2f}")
    cols[2].metric("DB size", f"{status.get('db_size_bytes', 0) / 1024:.1f} KiB")
    cols[3].metric("Promoted", sum((status.get("promotion_counts") or {}).values()))

    with st.expander("By layer"):
        for layer, count in (status.get("layer_counts") or {}).items():
            st.markdown(f"- **{layer}**: {count}")
    with st.expander("By promotion status"):
        for ps, count in (status.get("promotion_counts") or {}).items():
            st.markdown(f"- **{ps}**: {count}")

    recents = status.get("recent_records") or []
    if recents:
        _section("Recent memory")
        for rec in recents:
            layer = rec.get("layer")
            with st.expander(f"[{layer}] {str(rec.get('content'))[:120]} — conf {rec.get('confidence', 0):.2f}"):
                st.markdown(str(rec.get("content")))
                st.caption(
                    f"id={rec.get('id')} · promotion={rec.get('promotion_status')} · "
                    f"created_at={rec.get('created_at')}"
                )
            if rec.get("source_query") or rec.get("supporting_chunk_ids"):
                st.caption(
                    f"source_query: {rec.get('source_query')}  |  supporting chunks: "
                    + (", ".join(map(str, rec.get("supporting_chunk_ids") or [])))
                )


def _render_obsidian_brain(client: ARGUSAPIClient) -> None:
    """Obsidian Brain: dedicated vault status + selective memory promotion."""
    _section("Obsidian Brain")
    try:
        status = client.get_obsidian_brain_status()
    except ARGUSAPIClientError as exc:
        _fail(f"Unable to reach the API: {exc}")
        return

    st.markdown(f"**Vault path:** `{status.get('vault_path') or '—'}`")
    cols = st.columns(3)
    cols[0].metric("Configured", "Yes" if status.get("configured") else "No")
    cols[1].metric("Vault exists", "Yes" if status.get("exists") else "No")
    cols[2].metric("Notes", status.get("note_count"))

    if not status.get("configured") or not status.get("exists"):
        st.info(
            "The ARGUS Obsidian brain vault is not configured or does not exist. "
            "Set `ARGUS_BRAIN_VAULT_PATH` to a real Obsidian vault directory to enable this layer."
        )
        return

    st.markdown(f"**Write-back root:** `{status.get('write_back_root')}`")

    notes = status.get("recent_notes") or []
    if notes:
        with st.expander(f"Recent notes ({len(notes)})"):
            for n in notes:
                st.markdown(f"- `{n.get('path')}` · {n.get('modified_iso')}")

    if st.button("Promote eligible memories into the vault (selective, provenance-preserving)", key="obsidian_promote_go"):
        with st.spinner("Sweeping eligible PROMOTED long-term memories…"):
            try:
                result = client.promote_knowledge()
            except ARGUSAPIClientError as exc:
                _fail(f"Promotion failed: {exc}")
                result = None
        if result:
            st.success(
                f"Created {result.get('notes_created')}, skipped {result.get('notes_skipped')}, "
                f"failed {result.get('notes_failed')}."
            )
            for pth in result.get("created_paths") or []:
                st.markdown(f"- `{pth}`")
            if result.get("failed"):
                st.warning("; ".join(result.get("failed") or []))


def _run_query(client: ARGUSAPIClient, question: str) -> dict[str, Any] | None:
    """Run one question through the research loop and return the result (or None)."""
    with st.spinner("Researching and grounding the answer…"):
        try:
            return client.query(question, user_early_stop=False)
        except ARGUSAPIClientError as exc:
            st.error(f"API call failed — is the API running? {exc}")
            return None


def _citation_lines(citations: list[dict[str, Any]]) -> list[str]:
    """Short, user-friendly source references for an answer."""
    lines: list[str] = []
    for i, cit in enumerate(citations, 1):
        loc = ""
        if cit.get("page_start"):
            loc = f" p.{int(cit['page_start'])}"
        path = cit.get("source_path") or "source"
        lines.append(f"{i}. `{path}`{loc}")
    return lines


def _render_chat(client: ARGUSAPIClient) -> None:
    """Unified, ChatGPT-like chat: ask a question OR attach a file, in one thread."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "processing" not in st.session_state:
        st.session_state.processing = False

    chat_title = st.container()
    with chat_title:
        st.title("ARGUS")
        st.caption("Ask a question or drop a file. Answers are grounded in your Knowledge Base.")

    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                for block in msg.get("blocks", [msg]):
                    kind = block.get("kind", "text")
                    if kind == "text":
                        st.markdown(block.get("text", ""))
                    elif kind == "sources":
                        st.caption(block.get("label", "Sources"))
                        for line in block.get("lines", []):
                            st.markdown(line)
                    elif kind == "pill":
                        st.caption(block.get("text", ""))

    with st.container(border=True):
        attach_col, dispatch_col = st.columns([1, 1])
        with attach_col:
            uploaded = st.file_uploader(
                "Attach a document",
                type=list(_SUPPORTED_UI_EXTENSIONS),
                key="chat_attach",
                label_visibility="collapsed",
            )
        with dispatch_col, st.container():
            st.markdown("&nbsp;")
            st.markdown("**Ask below, or attach and then ask.**")

    prompt = st.chat_input("Ask ARGUS a question…")

    if uploaded is not None and not st.session_state.processing:
        st.session_state.processing = True
        filename = uploaded.name or "document"
        st.session_state.messages.append({"role": "user", "blocks": [{"kind": "text", "text": f"Uploaded: **{filename}**"}]})
        with chat_container, st.chat_message("user"):
            st.markdown(f"Uploaded: **{filename}**")
        _ingest_upload(client, uploaded, filename, chat_container)
        question = f"Summarize the document '{filename}' in a clear, well-structured way."
        _ask(client, question, chat_container, from_upload=True)
        st.session_state.processing = False
        st.rerun()

    if prompt and not st.session_state.processing:
        st.session_state.processing = True
        st.session_state.messages.append({"role": "user", "blocks": [{"kind": "text", "text": prompt}]})
        with chat_container, st.chat_message("user"):
            st.markdown(prompt)
        _ask(client, prompt, chat_container)
        st.session_state.processing = False
        st.rerun()


def _ingest_upload(
    client: ARGUSAPIClient,
    uploaded: Any,
    filename: str,
    container: Any,
) -> None:
    """Ingest an attached file into the Knowledge Base and reflect the outcome in chat."""
    import tempfile

    file_paths: list[str] = []
    with tempfile.TemporaryDirectory() as tmpdir:
        dest = Path(tmpdir) / Path(filename).name
        dest.write_bytes(uploaded.getvalue())
        file_paths.append(str(dest))
        with st.spinner("Ingesting the attached file…"):
            try:
                result = client.upload_files(file_paths)
            except ARGUSAPIClientError as exc:
                with container, st.chat_message("assistant"):
                    st.error(f"Upload failed: {exc}")
                st.session_state.messages.append(
                    {"role": "assistant", "blocks": [{"kind": "text", "text": f"Upload failed: {exc}"}]}
                )
                return

    uploaded_count = len(result.get("uploaded") or [])
    rejected = result.get("rejected") or []
    if uploaded_count == 0:
        reason = rejected[0].get("reason", "unsupported file") if rejected else "unknown"
        with container, st.chat_message("assistant"):
            st.error(f"Could not ingest `{filename}`: {reason}")
        st.session_state.messages.append(
            {"role": "assistant", "blocks": [{"kind": "text", "text": f"Could not ingest `{filename}`: {reason}"}]}
        )
        return

    with container, st.chat_message("assistant"):
        st.markdown(f"`{filename}` ingested ({uploaded_count} file(s)). Index refreshed — you can ask about it now.")
    st.session_state.messages.append(
        {"role": "assistant", "blocks": [{"kind": "text", "text": f"`{filename}` ingested and indexed."}]}
    )


def _ask(
    client: ARGUSAPIClient,
    question: str,
    container: Any,
    *,
    from_upload: bool = False,
) -> None:
    """Run the question, render the assistant answer and sources, remember it."""
    result = _run_query(client, question)
    if result is None:
        return

    answer = (result.get("answer") or "").strip()
    citations = result.get("citations") or []
    memory_consulted = result.get("memory_consulted") or []

    blocks: list[dict[str, Any]] = []
    if answer:
        blocks.append({"kind": "text", "text": answer})
    elif not from_upload:
        blocks.append({"kind": "text", "text": "_No answer returned._"})

    if memory_consulted:
        blocks.append(
            {
                "kind": "pill",
                "text": "Consulted ARGUS memory (derived knowledge, kept distinct from your document evidence).",
            }
        )

    if citations:
        blocks.append({"kind": "sources", "label": "Sources", "lines": _citation_lines(citations)})
    else:
        blocks.append({"kind": "pill", "text": "_No sources found — check that the Knowledge Base has been ingested._"})

    with container, st.chat_message("assistant"):
        for block in blocks:
            if block["kind"] == "text":
                st.markdown(block["text"])
            elif block["kind"] == "pill":
                st.caption(block["text"])
            elif block["kind"] == "sources":
                st.markdown(f"**{block['label']}**")
                for line in block["lines"]:
                    st.markdown(line)

    st.session_state.messages.append({"role": "assistant", "blocks": blocks})


def run() -> None:
    """ARGUS knowledge system: ChatGPT-like unified chat + management sidebar."""
    chat_col, side_col = st.columns([3, 1], gap="large")

    with side_col:
        st.markdown("### System")
        with st.expander("Connect", expanded=True):
            base_url = st.text_input("API base URL", value="http://localhost:8000")
        client = ARGUSAPIClient(base_url=base_url)
        with st.expander("Knowledge Base", expanded=False):
            _render_knowledge_base(client)
        with st.expander("ARGUS Brain (memory)", expanded=False):
            _render_brain(client)
        with st.expander("Obsidian Brain", expanded=False):
            _render_obsidian_brain(client)
        st.caption("Management views for operators. The chat above is the end-user surface.")

    with chat_col:
        _render_chat(client)


run()