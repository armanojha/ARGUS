"""Active Evidence Seeking (Phase 06).

Implements the `EvidenceGapDetectorInterface` shared contract from
`app/retrieval/policy.py` plus a vault-seeded evidence-task seeker
(V3 §5.1 subset). Together they turn "the evidence does not answer the
question" into an explicit, targeted retrieval action instead of an
unexplained loop.

Everything here is deterministic — no LLM call — so the detection and
the follow-up action remain traceable to the evidence actually in scope
and keep working when the gateway is down.

Obsidian hypotheses: per V3, user hypotheses are personal claims, NOT
automatically trusted evidence. `ObsidianHypothesisSeeker` therefore
only *seeds retrieval tasks* (queries to run), never facts to cite.
Full hypothesis-classification/write-back is deferred to Phase 09.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from app.config import Settings, get_settings
from app.evidence.models import EvidenceRef
from app.logging_config import get_logger
from app.retrieval.policy import EvidenceGapDetectorInterface

if TYPE_CHECKING:
    from app.orchestration.state import OrchestrationState

logger = get_logger("argus.retrieval.seeking")

_HYPOTHESIS_MARKER_RE = re.compile(r"(?im)^[:#*-]?\s*(hypothesis|assumption|conjecture)\s*[:#]?\s*(.*)$")
_STOPWORDS = {
    "what", "which", "when", "where", "who", "how", "why", "the", "a", "an", "and", "or", "of",
    "to", "in", "on", "for", "with", "is", "are", "was", "were", "does", "do", "did", "not",
    "this", "that", "these", "those", "from", "by", "at", "it", "its", "about", "between",
}


def _terms(text: str) -> set[str]:
    # Min length 3 keeps short but meaningful domain terms (e.g. "fox")
    # relevant to overlap detection; the stopword set still filters noise.
    return {t for t in re.findall(r"[a-zA-Z0-9_]{3,}", text.lower()) if t not in _STOPWORDS}


def _overlap(a: str, b: str) -> float:
    ta, tb = _terms(a), _terms(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


class AdaptiveEvidenceGapDetector(EvidenceGapDetectorInterface):
    """Deterministic gap detection + targeted retrieval-action formulation.

    Gap types produced:
    - ``no_evidence``        — nothing retrieved at all.
    - ``missing_evidence``   — a plan subquestion has no supporting evidence.
    - ``low_coverage``       — overall subquestion coverage is poor.
    - ``evidence_quality``   — top retrieved scores are weak.
    - ``contradiction_unresolved`` — an unresolved contradiction signal exists.
    - ``obsidian_hypothesis`` — a vault hypothesis seeds a follow-up task.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        hypothesis_seeker: Any | None = None,  # ObsidianHypothesisSeeker
    ) -> None:
        self.settings = settings or get_settings()
        self.hypothesis_seeker = hypothesis_seeker

    def detect_gaps(
        self,
        state: OrchestrationState,
        plan: Any,
        evidence: list[EvidenceRef],
    ) -> list[dict[str, Any]]:
        gaps: list[dict[str, Any]] = []

        if not evidence:
            gaps.append(self._make_gap(
                "no_evidence",
                "No evidence was retrieved for the question.",
                suggested_query=state["query"],
                priority=0.9,
            ))

        subquestions = list(getattr(plan, "subquestions", None) or [])
        issued = list(state.get("issued_subqueries", []))

        covered = [sq for sq in subquestions if any(_overlap(sq, issued_q) > 0.0 for issued_q in issued)]
        missing = [sq for sq in subquestions if sq not in covered]
        for sq in missing:
            gaps.append(self._make_gap(
                "missing_evidence",
                f"The plan subquestion has no supporting evidence: {sq[:200]}",
                suggested_query=sq,
                priority=0.8,
            ))

        if subquestions and len(covered) / len(subquestions) < 0.5:
            gaps.append(self._make_gap(
                "low_coverage",
                f"Overall coverage of the plan's subquestions is {len(covered)}/{len(subquestions)}.",
                suggested_query=plan.objective or state["query"],
                priority=0.7,
            ))

        if evidence:
            top_score = max(r.score for r in evidence)
            if top_score < self.settings.active_seeking_quality_threshold:
                gaps.append(self._make_gap(
                    "evidence_quality",
                    f"Top retrieved evidence score ({top_score:.2f}) is below the quality threshold.",
                    suggested_query=self._refine_for_quality(state["query"]),
                    priority=0.6,
                ))

        unresolved = [
            signal for signal in (state.get("contradiction_signals") or [])
            if not signal.get("resolved", False) and signal.get("critical", True)
        ]
        if unresolved:
            gaps.append(self._make_gap(
                "contradiction_unresolved",
                f"{len(unresolved)} unresolved critical contradiction(s) remain.",
                suggested_query=self._resolve_contradiction_query(state["query"]),
                priority=0.85,
            ))

        if self.hypothesis_seeker is not None:
            for task in self.hypothesis_seeker.seed_evidence_tasks(state["query"]):
                task["source"] = "obsidian_hypothesis"
                gaps.append(task)

        gaps = self._dedupe(gaps)
        logger.info(
            "evidence_gaps_detected",
            gaps=len(gaps),
            types=[g["gap_type"] for g in gaps],
        )
        return gaps

    def should_re_retrieve(self, gaps: list[dict[str, Any]]) -> bool:
        """Re-retrieve only when a gap is above the configured priority bar."""
        if not gaps:
            return False
        return any(g.get("priority", 0.0) >= self.settings.active_seeking_min_priority for g in gaps)

    # -- helpers -------------------------------------------------------------

    @staticmethod
    def _make_gap(
        gap_type: str,
        description: str,
        suggested_query: str | None,
        priority: float,
    ) -> dict[str, Any]:
        return {
            "gap_type": gap_type,
            "description": description,
            "suggested_query": suggested_query,
            "priority": priority,
            "source": "policy",
        }

    @staticmethod
    def _refine_for_quality(query: str) -> str:
        low = query.lower()
        if any(word in low for word in ("define", "definition")):
            return query
        return f"{query} definition and supporting evidence"

    @staticmethod
    def _resolve_contradiction_query(query: str) -> str:
        return f"{query} - resolution of conflicting evidence, authoritative sources"

    @staticmethod
    def _dedupe(gaps: list[dict[str, Any]]) -> list[dict[str, Any]]:
        seen: set[tuple[str, str]] = set()
        out: list[dict[str, Any]] = []
        for gap in gaps:
            key = (gap.get("gap_type", ""), (gap.get("suggested_query") or "").lower())
            if key in seen:
                continue
            seen.add(key)
            out.append(gap)
        return out


class ObsidianHypothesisSeeker:
    """Vault-seeded evidence tasks (V3 §5.1 subset).

    Finds personal-context notes in the ingested Obsidian vault that look
    like hypotheses (marker headings/text or a ``#hypothesis``-style tag)
    and turns them into *research tasks* when they overlap the query.

    Hypotheses are untrusted leads: the seeker only proposes queries to
    run against the evidence store — it never injects hypothesis text as
    evidence. Full taxonomy/classification is Phase 09.
    """

    def __init__(self, store: Any | None = None, task_priority: float = 0.6) -> None:
        self._store = store
        # Lazily built list of hypothesis dicts baked from the store.
        self._hypotheses: list[dict[str, Any]] | None = None
        self.task_priority = task_priority

    def find_hypotheses(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        """Return hypothesis statements from the vault relevant to the query."""
        if self._hypotheses is None:
            self._hypotheses = self._load_hypotheses()
        scored = sorted(
            (
                {**h, "relevance": round(_overlap(query, h["text"]), 4)}
                for h in self._hypotheses
            ),
            key=lambda h: h["relevance"],
            reverse=True,
        )
        return [h for h in scored if h["relevance"] > 0][:limit]

    def seed_evidence_tasks(self, query: str, limit: int = 3) -> list[dict[str, Any]]:
        """Turn relevant vault hypotheses into targeted evidence tasks."""
        tasks = []
        for hypothesis in self.find_hypotheses(query, limit=limit):
            tasks.append({
                "gap_type": "obsidian_hypothesis",
                "description": f"Vault hypothesis to investigate: {hypothesis['text'][:200]}",
                "suggested_query": self._task_query(hypothesis["text"]),
                "priority": self.task_priority,
                "hypothesis_text": hypothesis["text"],
                "note_path": hypothesis.get("note_path"),
            })
        return tasks

    def _load_hypotheses(self) -> list[dict[str, Any]]:
        from app.config import get_settings
        from app.evidence.store import get_evidence_store

        settings = get_settings()
        if not settings.obsidian_enabled and self._store is None:
            return []
        store = self._store or get_evidence_store()
        found: list[dict[str, Any]] = []
        for chunk in store.get_chunks_by_document_metadata("note_type", "personal_context"):
            text = chunk.text.strip()
            match = _HYPOTHESIS_MARKER_RE.search(text)
            hypothesis = (match.group(2) or match.group(1)).strip() if match else None
            if not hypothesis:
                continue
            tags = chunk.metadata.get("tags") or []
            if match or any("hypothesis" in str(t).lower() for t in tags):
                found.append({
                    "text": hypothesis,
                    "chunk_id": chunk.id,
                    "note_path": chunk.metadata.get("vault_relative_path"),
                })
            if len(found) >= 200:
                break
        return found

    @staticmethod
    def _task_query(hypothesis_text: str) -> str:
        return f"Investigate whether: {hypothesis_text}"


def get_adaptive_gap_detector(
    settings: Settings | None = None,
    hypothesis_seeker: Any | None = None,
) -> AdaptiveEvidenceGapDetector:
    """Create the Phase 06 gap detector (with Obsidian seeker when enabled)."""
    settings = settings or get_settings()
    if hypothesis_seeker is None and settings.active_evidence_seeking_enabled and settings.obsidian_enabled:
        hypothesis_seeker = ObsidianHypothesisSeeker()
    return AdaptiveEvidenceGapDetector(settings=settings, hypothesis_seeker=hypothesis_seeker)


__all__ = [
    "AdaptiveEvidenceGapDetector",
    "ObsidianHypothesisSeeker",
    "get_adaptive_gap_detector",
]