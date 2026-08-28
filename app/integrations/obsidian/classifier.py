"""Phase 09.1: 7-class Obsidian knowledge-taxonomy classifier.

Deterministic, feature-driven classification over the V3 §4.2 taxonomy.
No LLM calls are involved: classification must be reproducible and cheap
during ingestion. The hypothesis converter (Phase 09.2) turns Hypothesis /
Task-Question notes into research objectives that drive the evidence loop.
"""

from __future__ import annotations

import re
from typing import Any
from uuid import uuid4

from app.integrations.obsidian.contracts import (
    CLASSIFICATION_RULES,
    ClassificationResult,
    HypothesisConverterInterface,
    HypothesisResearchObjective,
    KnowledgeClass,
    ObsidianClassifierInterface,
)
from app.logging_config import get_logger

logger = get_logger("argus.obsidian.classifier")

# Frontmatter keys that carry an explicit knowledge class.
_CLASS_KEYS = ("knowledge_class", "argus_class", "note_class", "note_type")
_VALID_TYPE_VALUES = {cls.value for cls in KnowledgeClass}

# Hypothesis signal patterns.
_HYPOTHESIS_STRONG = re.compile(r"\b(hypothesi[sz]e|hypothesis|suspect)\b", re.IGNORECASE)
_HYPOTHESIS_WEAK = re.compile(
    r"\b(i (wonder|believe|guess))\b"
    r"|\b(possibly|maybe|perhaps)\b"
    r"|\bmight (be|explain|indicate|cause)\b"
    r"|\blikely (caused by|due to|linked|explains)\b",
    re.IGNORECASE,
)
_URL = re.compile(r"https?://\S+")
_QUESTION_LINE = re.compile(r"\?\s*$")
_WIKILINK = re.compile(r"\[\[")

# Frontmatter keys for source / project detection.
_SOURCE_FM_KEYS = {"source", "author", "url", "doi", "isbn", "publication", "journal", "publisher", "citation"}
_PROJECT_FM_KEYS = {"status", "deadline", "owner", "priority", "milestone", "team", "epic"}

_INDEX_HEADINGS = {"index", "moc", "map of content", "table of contents", "toc", "overview"}
_SOURCE_HEADINGS = {"references", "source", "sources", "bibliography"}
_PROJECT_HEADINGS = ("project", "milestones", "action items", "goals")

# Tag -> class mapping (order matters: first match wins). Supports nested
# tags like `#class/hypothesis`.
_TAG_CLASS_MAP: list[tuple[str, set[str]]] = [
    ("research_capture", {"research-capture", "argus", "argus-research"}),
    ("source_note", {"source", "source-note", "source_note", "sourcenote"}),
    ("hypothesis", {"hypothesis", "hyp"}),
    ("task_question", {"question", "task", "todo", "issue"}),
    ("reference_index", {"index", "moc", "reference", "references"}),
    ("project_note", {"project", "project-note", "project_note"}),
]


def _normalize_frontmatter(frontmatter: Any) -> dict[str, Any]:
    """Normalize a frontmatter object (dict or Pydantic model) to a dict."""
    if frontmatter is None:
        return {}
    if isinstance(frontmatter, dict):
        return frontmatter
    if hasattr(frontmatter, "model_dump"):
        return frontmatter.model_dump()
    return {}


def _collect_tags(frontmatter: dict[str, Any]) -> list[str]:
    """Collect lowercase tag names from frontmatter."""
    tags: list[str] = []
    for value in frontmatter.get("tags", []) or []:
        if isinstance(value, str) and value.strip():
            tags.append(value.strip().lstrip("#").lower())
    return tags


def _collect_section_headings(sections: list[Any]) -> list[str]:
    """Collect section headings (objects or dicts with a `heading` key)."""
    headings: list[str] = []
    for section in sections or []:
        if hasattr(section, "heading"):
            headings.append(section.heading)
        elif isinstance(section, dict):
            headings.append(str(section.get("heading", "")))
    return [h for h in headings if h]


def _class_from_tags(tags: list[str]) -> KnowledgeClass | None:
    """Map tags to a knowledge class using the first matching signal."""
    for class_value, tag_set in _TAG_CLASS_MAP:
        for tag in tags:
            if tag in tag_set or any(tag.endswith(f"/{cand}") for cand in tag_set):
                return KnowledgeClass(class_value)
    return None


def _explicit_class(frontmatter: dict[str, Any]) -> KnowledgeClass | None:
    """Return the explicit knowledge class from frontmatter, if present."""
    custom = frontmatter.get("custom", {})
    if not isinstance(custom, dict):
        custom = {}
    for key in _CLASS_KEYS:
        raw = custom.get(key, frontmatter.get(key))
        if isinstance(raw, str) and raw.strip():
            try:
                return KnowledgeClass(raw.strip().lower())
            except ValueError:
                logger.warning("obsidian_unknown_knowledge_class", value=raw)
    # A `type` override with a valid taxonomy value is also honored.
    raw_type = custom.get("type", frontmatter.get("type"))
    if isinstance(raw_type, str) and raw_type.strip().lower() in _VALID_TYPE_VALUES:
        return KnowledgeClass(raw_type.strip().lower())
    return None


def extract_hypothesis_text(note: Any) -> str:
    """Extract a concise hypothesis statement from a parsed note.

    Uses the title (frontmatter or first H1) or the first non-empty line
    of the body. Deterministic and bounded.
    """
    title = getattr(getattr(note, "frontmatter", None), "title", None)
    if isinstance(title, str) and title.strip():
        return " ".join(title.split())[:300]
    body = (note.content_without_frontmatter or "").strip()
    lines = [ln.strip().lstrip("#").strip() for ln in body.splitlines() if ln.strip()]
    line = lines[0] if lines else ""
    if line:
        return " ".join(line.split())[:300]
    stem = getattr(note, "file_stem", None) or ""
    return " ".join(stem.split())[:300]


class RuleBasedObsidianClassifier(ObsidianClassifierInterface):
    """Deterministic 7-class classifier with explicit frontmatter override.

    Priority order:
      1. Explicit frontmatter knowledge_class / note_type / type
      2. Note located inside the ARGUS write-back area (90_ARGUS)
      3. Tag signals
      4. Content heuristics (source -> index -> question -> hypothesis ->
         project -> default knowledge note)
    """

    def __init__(self, default_class: str = "knowledge_note") -> None:
        self._default_class = KnowledgeClass(default_class)

    # Interface (async; implemented on top of the synchronous core).
    async def classify_note(
        self,
        note_path: str,
        content: str,
        frontmatter: dict[str, Any],
        sections: list[Any],
    ) -> ClassificationResult:
        return self.classify_sync(note_path, content, frontmatter, sections)

    # Synchronous core used by the ingestion pipeline.
    def classify_sync(
        self,
        note_path: str,
        content: str,
        frontmatter: dict[str, Any],
        sections: list[Any],
    ) -> ClassificationResult:
        fm = _normalize_frontmatter(frontmatter)
        custom = fm.get("custom", {})
        if not isinstance(custom, dict):
            custom = {}
        tags = _collect_tags(fm)
        features: dict[str, Any] = {}

        # 1. Explicit frontmatter override.
        explicit = _explicit_class(fm)
        if explicit is not None:
            features["explicit_frontmatter"] = explicit.value
            return self._result(explicit, 0.97, "Explicit frontmatter knowledge_class", features)

        # 2. ARGUS write-back area.
        path_lower = str(note_path).replace("\\", "/").lower()
        if "90_argus" in path_lower or any(
            marker in path_lower
            for marker in ("research_output", "research_traces", "evidence_reports", "sync_logs")
        ):
            features["argus_area"] = True
            return self._result(KnowledgeClass.RESEARCH_CAPTURE, 0.95, "Note inside 90_ARGUS write-back area", features)

        # 3. Tag signals.
        tag_class = _class_from_tags(tags)
        if tag_class is not None:
            features["tags"] = sorted(tags)
            features["tag_signal"] = tag_class.value
            return self._result(tag_class, 0.9, f"Tagged with {tag_class.value}", features)

        # 4. Content heuristics.
        headings_lower = [h.strip().lower() for h in _collect_section_headings(sections)]
        body = content or ""
        lines = [ln.strip() for ln in body.splitlines() if ln.strip()]

        # 4a. Source note: provenance fields / URL density.
        if (set(custom) & _SOURCE_FM_KEYS) or (len(_URL.findall(body)) >= 2) or (
            any(h in headings_lower for h in _SOURCE_HEADINGS) and _URL.search(body)
        ):
            features["provenance_fields"] = sorted(set(custom) & _SOURCE_FM_KEYS) or "urls"
            return self._result(KnowledgeClass.SOURCE_NOTE, 0.85, "Provenance fields / URL density present", features)

        # 4b. Reference index: wikilink hub / index heading.
        wikilink_count = len(_WIKILINK.findall(body))
        word_count = len(re.split(r"\s+", body))
        if any(h in headings_lower for h in _INDEX_HEADINGS) or (wikilink_count >= 4 and word_count <= 250):
            features["wikilink_count"] = wikilink_count
            return self._result(KnowledgeClass.REFERENCE_INDEX, 0.85, "Wikilink hub / index structure", features)

        # 4c. Task / question.
        question_lines = [ln for ln in lines if _QUESTION_LINE.search(ln)]
        if len(question_lines) >= 2 and len(question_lines) / max(len(lines), 1) >= 0.5:
            features["question_lines"] = len(question_lines)
            return self._result(KnowledgeClass.TASK_QUESTION, 0.85, "Content is question-heavy", features)

        # 4d. Hypothesis.
        if _HYPOTHESIS_STRONG.search(body) or any("hypothes" in h for h in headings_lower):
            features["hypothesis_marker"] = True
            return self._result(KnowledgeClass.HYPOTHESIS, 0.9, "Explicit hypothesis markers", features)
        if _HYPOTHESIS_WEAK.search(body):
            features["hypothesis_marker"] = "weak"
            return self._result(KnowledgeClass.HYPOTHESIS, 0.75, "Speculative phrasing detected", features)

        # 4e. Project note.
        if (set(custom) & _PROJECT_FM_KEYS) or any(
            any(marker in h for marker in _PROJECT_HEADINGS) for h in headings_lower
        ):
            features["project_signals"] = sorted(set(custom) & _PROJECT_FM_KEYS) or "headings"
            return self._result(KnowledgeClass.PROJECT_NOTE, 0.8, "Project tracking fields present", features)

        # 5. Default.
        features["default"] = True
        return self._result(self._default_class, 0.7, "No strong signals; default knowledge note", features)

    def get_treatment_rule(self, knowledge_class: str) -> str:
        """Get the treatment rule for a knowledge class."""
        try:
            return CLASSIFICATION_RULES[KnowledgeClass(knowledge_class)].treatment.value
        except (ValueError, KeyError):
            logger.warning("obsidian_unknown_knowledge_class_lookup", knowledge_class=knowledge_class)
            return "personalization_only"

    def get_all_rules(self) -> dict[str, Any]:
        """Get all classification rules as a plain dict."""
        return {
            rule.knowledge_class.value: {
                "treatment": rule.treatment.value,
                "requires_provenance": rule.requires_provenance,
                "drives_research": rule.drives_research,
                "is_personal": rule.is_personal,
                "is_argus_generated": rule.is_argus_generated,
            }
            for rule in CLASSIFICATION_RULES.values()
        }

    def _result(
        self,
        knowledge_class: KnowledgeClass,
        confidence: float,
        reasoning: str,
        features: dict[str, Any],
    ) -> ClassificationResult:
        rule = CLASSIFICATION_RULES[knowledge_class]
        return ClassificationResult(
            knowledge_class=rule.knowledge_class.value,
            confidence=confidence,
            reasoning=reasoning,
            treatment_rule=rule.treatment.value,
            features=features,
        )


class RuleBasedHypothesisConverter(HypothesisConverterInterface):
    """Deterministic Hypothesis / Task-Question -> research objective converter.

    Honors frontmatter opt-outs (e.g. `argus_research: false`). Research
    objectives are derived from the hypothesis text and active evidence
    seeking is enabled via the runner (Phase 09.2).
    """

    def __init__(self, opt_out_keys: tuple[str, ...] = ("argus_research", "research", "convert_to_research")) -> None:
        self._opt_out_keys = opt_out_keys

    def should_convert(self, note_class: str, frontmatter: dict[str, Any]) -> bool:
        """True when the note class drives research and is not opted out."""
        try:
            rule = CLASSIFICATION_RULES[KnowledgeClass(note_class)]
        except (ValueError, KeyError):
            return False
        if not rule.drives_research:
            return False
        fm = _normalize_frontmatter(frontmatter)
        custom = fm.get("custom", {})
        if not isinstance(custom, dict):
            custom = {}
        for key in self._opt_out_keys:
            raw = custom.get(key, fm.get(key))
            if raw is False:
                return False
            if isinstance(raw, str) and raw.strip().lower() in {"false", "no", "skip", "off"}:
                return False
        return True

    async def convert_hypothesis(
        self,
        hypothesis_text: str,
        note_path: str,
        context: dict[str, Any] | None = None,
    ) -> HypothesisResearchObjective:
        text = " ".join((hypothesis_text or "").split())
        if not text:
            text = "the topic described in the note"
        topic = self._extract_topic(text)
        objective = f"Determine, using external evidence, whether the following hypothesis holds: {text}"
        subquestions = [
            f"What evidence supports: {text}?",
            f"What evidence contradicts: {text}?",
            f"What do reliable sources say about: {topic}?",
        ]
        priority = 0.8
        ctx = _normalize_frontmatter(context)
        custom = ctx.get("custom", {})
        if not isinstance(custom, dict):
            custom = {}
        raw_priority = custom.get("priority", ctx.get("priority"))
        if isinstance(raw_priority, (int, float)):
            priority = max(0.0, min(1.0, float(raw_priority)))

        return HypothesisResearchObjective(
            hypothesis_id=f"hyp-{uuid4().hex[:8]}",
            hypothesis_text=text,
            research_objective=objective,
            subquestions=subquestions,
            suggested_patterns=["hypothesis_research", "verification"],
            priority=priority,
            source_note_path=note_path,
        )

    @staticmethod
    def _extract_topic(text: str) -> str:
        """Extract a short topic phrase from the hypothesis text."""
        topic = text
        lowered = text.lower()
        for marker in ("whether", "that", "if "):
            idx = lowered.find(marker)
            if idx >= 0:
                topic = text[idx + len(marker):].strip().strip(":,")
                break
        topic = topic.split(",", 1)[0].strip(" .")
        words = topic.split()
        if len(words) > 12:
            topic = " ".join(words[:12])
        return topic or text