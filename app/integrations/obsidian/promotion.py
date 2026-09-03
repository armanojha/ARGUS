"""Memory -> Obsidian Brain selective promotion (Phase 08/09 bridge).

Connects ARGUS's persistent machine memory (``MemoryStore``) to its dedicated,
human-readable Obsidian brain vault (``settings.argus_brain_vault_path``).

Promotion is SELECTIVE and never automatic-trust:
  * Only records with ``promotion_status == MemoryPromotionStatus.PROMOTED``
    and in eligible long-term layers are considered.
  * Only records that carry provenance (supporting chunk IDs / source query /
    structured subject-predicate-object) are eligible — a memory with no
    provenance is never turned into a knowledge note.
  * A minimum confidence gate applies (``memory_confidence_threshold``).
  * Idempotent: a note already written for a given memory record id is skipped.

This preserves the "Obsidian brain stays distinct from both the Knowledge Base
and machine memory" boundary: notes are structured, provenance-bearing, and
created only for knowledge that qualified for long-term retention.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from app.config import get_settings
from app.integrations.obsidian.writer import _escape_yaml, _safe_md_name
from app.logging_config import get_logger
from app.memory.interfaces import (
    MemoryLayer,
    MemoryPromotionStatus,
    MemoryQuery,
    MemoryStoreInterface,
)

logger = get_logger("argus.obsidian.promotion")

# Layers whose records are eligible to become long-term knowledge notes.
_ELIGIBLE_LAYERS = {MemoryLayer.LONG_TERM_KNOWLEDGE.value}

# Marker key stored in a note's frontmatter to identify ARGUS-generated brain notes.
_NOTE_KIND = "argus-knowledge"


@dataclass
class PromotionResult:
    """Outcome of a selective promotion sweep."""

    notes_created: int = 0
    notes_skipped: int = 0
    notes_failed: int = 0
    created_paths: list[str] = field(default_factory=list)
    skipped_ids: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)


def promote_eligible_memories(
    store: MemoryStoreInterface,
    vault_root: Path | None = None,
    *,
    rebuild_indexes: bool = False,
    limit: int = 50,
) -> PromotionResult:
    """Promote eligible PROMOTED long-term memories into the brain vault.

    Args:
        store: The active memory store.
        vault_root: Brain vault root; defaults to ``settings.argus_brain_vault_path``.
        rebuild_indexes: Unused placeholder reserved for consistency (notes are
            not part of the EvidenceStore index).
        limit: Maximum number of candidate memories to consider per sweep.
    """
    import asyncio

    settings = get_settings()
    vault_root = vault_root or settings.argus_brain_vault_path
    vault_root = Path(vault_root)
    result = PromotionResult()

    if not vault_root or str(vault_root).strip() in ("", "."):
        logger.info("brain_promotion_skipped", reason="brain_vault_not_configured")
        return result
    if not vault_root.exists() or not vault_root.is_dir():
        logger.info("brain_promotion_skipped", reason="brain_vault_missing")
        return result

    min_conf = settings.memory_confidence_threshold
    query = MemoryQuery(
        query_text="",
        layers=[MemoryLayer.LONG_TERM_KNOWLEDGE],
        limit=limit,
        min_confidence=min_conf,
    )

    try:
        records = asyncio.run(store.retrieve(query)) or []
    except RuntimeError:
        records = []
    except Exception as exc:  # noqa: BLE001 - promotion is best-effort
        logger.warning("brain_promotion_retrieve_failed", error=str(exc))
        return result

    notes_dir = vault_root / settings.argus_brain_write_back_root / "Knowledge"
    notes_dir.mkdir(parents=True, exist_ok=True)

    for record in records:
        status = getattr(record, "promotion_status", None)
        if status != MemoryPromotionStatus.PROMOTED:
            continue
        if not _has_provenance(record):
            result.notes_skipped += 1
            result.skipped_ids.append(str(getattr(record, "id", "")))
            continue

        record_id = str(getattr(record, "id", ""))
        note_path = notes_dir / f"{_note_slug(record)}_{record_id[:8]}.md"
        if note_path.exists():
            result.notes_skipped += 1
            result.skipped_ids.append(record_id)
            continue

        try:
            note_path.write_text(_render_note(record), encoding="utf-8")
            result.notes_created += 1
            result.created_paths.append(str(note_path))
            logger.info("brain_note_created", record=str(record_id), path=str(note_path))
        except Exception as exc:  # noqa: BLE001 - per-note failure must not abort sweep
            result.notes_failed += 1
            result.failed.append(str(exc))

    logger.info(
        "brain_promotion_complete",
        created=result.notes_created,
        skipped=result.notes_skipped,
        failed=result.notes_failed,
    )
    return result


def _has_provenance(record) -> bool:
    chunks = getattr(record, "supporting_chunk_ids", None) or []
    source_query = getattr(record, "source_query", None)
    subject = getattr(record, "subject", None)
    return bool(chunks or source_query or subject)


def _note_slug(record) -> str:
    subject = getattr(record, "subject", None)
    if subject:
        return _safe_md_name(str(subject))
    content = getattr(record, "content", "") or ""
    first_line = content.strip().splitlines()[0] if content.strip() else "knowledge"
    return _safe_md_name(first_line[:60])


def _render_note(record) -> str:
    subject = getattr(record, "subject", None)
    predicate = getattr(record, "predicate", None)
    object_value = getattr(record, "object", None)
    content = getattr(record, "content", "") or ""
    confidence = float(getattr(record, "confidence", 0.0))
    source_query = getattr(record, "source_query", None)
    supporting_chunk_ids = list(getattr(record, "supporting_chunk_ids", []) or [])
    tags = list(getattr(record, "tags", []) or [])

    yaml_lines = [
        "---",
        f"type: {_NOTE_KIND}",
        f"confidence: {confidence:.2f}",
        "created_by: argus",
        "promotion_status: promoted",
        f"memory_id: {getattr(record, 'id', '')}",
    ]
    if subject:
        yaml_lines.append(f"subject: \"{_escape_yaml(str(subject))}\"")
    if predicate:
        yaml_lines.append(f"predicate: \"{_escape_yaml(str(predicate))}\"")
    if object_value:
        yaml_lines.append(f"object: \"{_escape_yaml(str(object_value))}\"")
    if source_query:
        yaml_lines.append(f"source_query: \"{_escape_yaml(str(source_query))}\"")
    if supporting_chunk_ids:
        yaml_lines.append("sources:")
        for cid in supporting_chunk_ids:
            yaml_lines.append(f"  - {cid}")
    if tags:
        yaml_lines.append(f"tags: {[str(t) for t in tags]}")
    yaml_lines.append("---")
    yaml_lines.append("")

    title = subject if subject else (_note_slug(record)).replace("_", " ")
    body = [f"# {title}", ""]

    if subject and predicate:
        body.append(f"**{subject}** *{predicate}* **{object_value}**" if object_value else f"**{subject}** *{predicate}*")
        body.append("")

    body.append(content.strip())
    body.append("")

    if supporting_chunk_ids:
        body.append("## Evidence")
        body.append("")
        for cid in supporting_chunk_ids:
            body.append(f"- `{cid}`")
        body.append("")

    body.append("---")
    body.append("_Generated by ARGUS. Provenance preserved; treat as derived knowledge, not raw source evidence._")

    return "\n".join(yaml_lines + body).strip() + "\n"