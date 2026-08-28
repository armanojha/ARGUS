"""Phase 09.2: Hypothesis research runner and coordinator.

Turns Hypothesis / Task-Question notes into research objectives and runs
them through the Phase 02 agentic loop (with Phase 04 verification exposed
as an optional cross-check), producing a verified / contradicted /
undetermined outcome with citations.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field

from app.config import Settings, get_settings
from app.integrations.obsidian.classifier import (
    RuleBasedHypothesisConverter,
    RuleBasedObsidianClassifier,
    extract_hypothesis_text,
)
from app.integrations.obsidian.contracts import HypothesisResearchObjective
from app.integrations.obsidian.parser import parse_obsidian_note
from app.integrations.obsidian.scanner import VaultScanner
from app.integrations.obsidian.writer import ObsidianWriter
from app.logging_config import get_logger

logger = get_logger("argus.obsidian.research")


def _citation_to_dict(c: object) -> dict[str, Any]:
    """Best-effort serialize a citation to a plain dict for report files."""
    dump = getattr(c, "model_dump", None)
    if callable(dump):
        result = dump()
        if isinstance(result, dict):
            return result
        return {"value": result}
    return {"value": c}


class HypothesisResearchOutcome(BaseModel):
    """Outcome of running research on a hypothesis."""

    model_config = ConfigDict(extra="forbid")

    research_id: str
    hypothesis_text: str
    research_objective: str
    status: str = "undetermined"  # verified | contradicted | partial | undetermined | error
    answer: str | None = None
    citations: list[Any] = Field(default_factory=list)
    confidence: float | None = None
    validation: str | None = None
    caveats: list[str] = Field(default_factory=list)
    source_note_path: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class HypothesisResearchRunner:
    """Runs a research objective through the Phase 02 orchestration loop.

    Components (router / retriever / reranker / Phase 06 policy) may be
    injected for deterministic testing; otherwise they are resolved from
    the usual Phase 00.3 / Phase 01 singletons inside `run_query`.
    """

    def __init__(
        self,
        *,
        router: Any | None = None,
        retriever: Any | None = None,
        reranker: Any | None = None,
        settings: Settings | None = None,
        policy_router: Any | None = None,
        gap_detector: Any | None = None,
        stopping_logic: Any | None = None,
        request_id_prefix: str = "obsidian-hyp",
    ) -> None:
        self.router = router
        self.retriever = retriever
        self.reranker = reranker
        self.settings = settings
        self.policy_router = policy_router
        self.gap_detector = gap_detector
        self.stopping_logic = stopping_logic
        self.request_id_prefix = request_id_prefix

    async def run(self, objective: HypothesisResearchObjective) -> HypothesisResearchOutcome:
        """Run the research objective and produce a deterministic outcome."""
        from app.orchestration.graph import run_query

        outcome = HypothesisResearchOutcome(
            research_id=f"research-{objective.hypothesis_id}",
            hypothesis_text=objective.hypothesis_text,
            research_objective=objective.research_objective,
            source_note_path=objective.source_note_path,
        )
        request_id = f"{self.request_id_prefix}-{objective.hypothesis_id}-{uuid4().hex[:6]}"

        try:
            result = await run_query(
                objective.research_objective,
                request_id=request_id,
                router=self.router,
                retriever=self.retriever,
                reranker=self.reranker,
                settings=self.settings,
                policy_router=self.policy_router,
                gap_detector=self.gap_detector,
                stopping_logic=self.stopping_logic,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("hypothesis_research_failed", hypothesis_id=objective.hypothesis_id, error=str(exc))
            outcome.status = "error"
            outcome.caveats.append(f"research run failed: {exc}")
            return outcome

        outcome.answer = result.answer
        outcome.citations = list(result.citations or [])

        scores = [
            float(c.score)
            for c in outcome.citations
            if isinstance(c.score, (int, float))
        ]
        if scores:
            outcome.confidence = round(sum(scores) / len(scores), 4)

        if not result.answer and not outcome.citations:
            outcome.status = "undetermined"
            outcome.caveats.append("no answer or citations returned")
        elif not outcome.citations:
            outcome.status = "undetermined"
            outcome.caveats.append("answer returned but no citations to ground it")
        elif result.answer:
            outcome.status = "verified"
        else:
            outcome.status = "partial"
        return outcome

    async def verify_result(
        self,
        outcome: HypothesisResearchOutcome,
        evidence_store: Any,
        router: Any | None = None,
        graph_store: Any | None = None,
    ) -> HypothesisResearchOutcome:
        """Optional Phase 04 cross-check of the hypothesis against citations.

        Non-fatal: if verification is unavailable, the heuristic outcome is
        preserved and `validation` explains why it was skipped.
        """
        chunk_ids: list[UUID] = []
        for citation in outcome.citations:
            raw = getattr(citation, "chunk_id", None)
            if raw is None:
                continue
            try:
                chunk_ids.append(UUID(str(raw).split(":")[-1]))
            except ValueError:
                continue
        if not chunk_ids or not outcome.answer:
            outcome.validation = "skipped; no citation chunk ids to verify"
            return outcome

        verifier_router = router or self.router
        if verifier_router is None:
            outcome.validation = "skipped; no LLM router available for verification"
            return outcome

        try:
            from app.verification.engine import verify_claim
            from app.verification.models import VerificationRequest

            request = VerificationRequest(
                claim_id=UUID(int=0),
                claim_text=outcome.hypothesis_text,
                supporting_chunk_ids=chunk_ids,
                entity_names=[outcome.hypothesis_text] if outcome.hypothesis_text else [],
                max_evidence_items=10,
            )
            verification = await verify_claim(
                request,
                router=verifier_router,
                evidence_store=evidence_store,
                graph_store=graph_store,
                settings=self.settings or get_settings(),
                request_id=outcome.research_id,
            )
            outcome.validation = verification.status.value if getattr(verification.status, "value", None) else str(verification.status)
            if outcome.validation in {"supported", "partial"} and outcome.status in {"undetermined", "partial"}:
                outcome.status = "verified" if outcome.validation == "supported" else "partial"
            elif outcome.validation == "contradicted":
                outcome.status = "contradicted"
            if verification.reasoning:
                outcome.caveats.append(verification.reasoning[:300])
        except Exception as exc:  # noqa: BLE001
            logger.warning("hypothesis_verification_skipped", research_id=outcome.research_id, error=str(exc))
            outcome.validation = f"verification unavailable: {exc}"
        return outcome


class ObsidianResearchCoordinator:
    """End-to-end research loop for hypothesis notes.

    scan/parse -> classify (7-class) -> convert -> research -> write back the
    research capture + evidence report + trace into 90_ARGUS.
    """

    def __init__(
        self,
        vault_root: Path,
        *,
        settings: Settings | None = None,
        classifier: Any | None = None,
        converter: Any | None = None,
        runner: HypothesisResearchRunner | None = None,
        writer: ObsidianWriter | None = None,
        write_back_root: str = "90_ARGUS",
    ) -> None:
        self.vault_root = Path(vault_root).resolve()
        self.settings = settings or get_settings()
        self.classifier = classifier or RuleBasedObsidianClassifier()
        self.converter = converter or RuleBasedHypothesisConverter()
        self.runner = runner or HypothesisResearchRunner(settings=self.settings)
        self.writer = writer or ObsidianWriter(self.vault_root, write_back_root)
        self.scanner = VaultScanner(self.vault_root)

    async def process_hypothesis_note(self, note_path: Path | str) -> HypothesisResearchOutcome | None:
        """Process a single note: classify, convert, research, write outputs."""
        abs_path = Path(note_path)
        if not abs_path.is_absolute():
            abs_path = self.vault_root / abs_path
        if not abs_path.exists():
            logger.warning("hypothesis_note_not_found", path=str(abs_path))
            return None
        note = parse_obsidian_note(abs_path, self.vault_root)
        return await self._process_note(note)

    async def process_vault(
        self,
        exclude_patterns: list[str] | None = None,
    ) -> list[HypothesisResearchOutcome]:
        """Scan the vault and run research for every hypothesis note."""
        outcomes: list[HypothesisResearchOutcome] = []
        for note in self.scanner.scan(exclude_patterns=exclude_patterns):
            outcome = await self._process_note(note, write_outputs=False)
            if outcome is not None:
                outcomes.append(outcome)
        return outcomes

    async def _process_note(self, note: Any, write_outputs: bool = True) -> HypothesisResearchOutcome | None:
        classification = await self.classifier.classify_note(
            str(note.file_path),
            note.raw_content,
            note.frontmatter,
            note.sections,
        )
        frontmatter = note.frontmatter.model_dump() if hasattr(note.frontmatter, "model_dump") else {}
        if not self.converter.should_convert(classification.knowledge_class, frontmatter):
            logger.info(
                "hypothesis_not_eligible",
                note=note.vault_relative_path,
                knowledge_class=classification.knowledge_class,
            )
            return None

        objective = await self.converter.convert_hypothesis(
            extract_hypothesis_text(note),
            note.vault_relative_path,
            context=frontmatter,
        )
        outcome = await self.runner.run(objective)
        await self._verify_outcome(outcome)
        if write_outputs:
            self._write_research_outputs(outcome, title=note.file_stem)
        return outcome

    async def _verify_outcome(self, outcome: HypothesisResearchOutcome) -> None:
        """Cross-check the heuristic research outcome against Phase 04 verification.

        The Phase 09.2 ``run()`` heuristic marks a grounded answer as
        "verified" purely from retrieval results; calling ``verify_result``
        here ensures a hypothesis only stays/moves to "verified" after an
        actual claim-to-evidence verification pass (and can be downgraded to
        "contradicted"). Non-fatal: if no router is available the heuristic
        outcome is preserved and ``validation`` explains why.
        """
        router = getattr(self.runner, "router", None)
        if router is None:
            try:
                from app.llm_gateway import get_router

                router = get_router()
            except Exception as exc:  # noqa: BLE001 - verification is a cross-check
                logger.warning("hypothesis_verifier_unavailable", error=str(exc))
                router = None
        if router is None:
            outcome.validation = "skipped; no LLM router available for verification"
            return
        try:
            from app.evidence.store import get_evidence_store
            from app.graph.store import get_graph_store

            await self.runner.verify_result(
                outcome,
                evidence_store=get_evidence_store(),
                router=router,
                graph_store=get_graph_store(),
            )
        except Exception as exc:  # noqa: BLE001 - verification is non-fatal
            logger.warning("hypothesis_verification_skipped", research_id=outcome.research_id, error=str(exc))
            outcome.validation = f"verification unavailable: {exc}"

    def _write_research_outputs(self, outcome: HypothesisResearchOutcome, title: str) -> None:
        """Write research capture / evidence report / trace to 90_ARGUS."""
        from app.integrations.obsidian.models import ResearchCaptureNote

        capture = ResearchCaptureNote(
            research_id=outcome.research_id,
            title=title,
            status="completed" if outcome.status in {"verified", "partial"} else outcome.status,
            query=outcome.research_objective,
            answer=outcome.answer or "No answer was produced.",
            citations=[_citation_to_dict(c) for c in outcome.citations],
            confidence=outcome.confidence,
            caveats=outcome.caveats,
            sources=[outcome.source_note_path],
            claims=[outcome.hypothesis_text],
        )
        capture_path = self.writer.write_research_capture(capture)
        evidence_summary = self._evidence_summary(outcome, capture_path)
        self.writer.write_evidence_report(
            research_id=outcome.research_id,
            title=f"Evidence for {title}",
            evidence_summary=evidence_summary,
            citations=[_citation_to_dict(c) for c in outcome.citations],
        )
        self.writer.write_research_trace(
            research_id=outcome.research_id,
            trace_data={
                "research_id": outcome.research_id,
                "status": outcome.status,
                "confidence": outcome.confidence,
                "validation": outcome.validation,
                "hypothesis_text": outcome.hypothesis_text,
                "source_note": outcome.source_note_path,
                "citation_count": len(outcome.citations),
            },
        )
        logger.info("hypothesis_research_written", research_id=outcome.research_id, capture=str(capture_path))

    @staticmethod
    def _evidence_summary(outcome: HypothesisResearchOutcome, capture_path: Path) -> str:
        """Build a short evidence summary with wikilinks back to the capture."""
        lines = [f"Research outcome: **{outcome.status}**."]
        if outcome.confidence is not None:
            lines.append(f"Confidence: {outcome.confidence:.1%}.")
        if outcome.caveats:
            lines.append("Caveats: " + "; ".join(outcome.caveats[:3]))
        lines.append(f"See capture: [[{capture_path.stem}|Research capture]]")
        return "\n".join(lines)