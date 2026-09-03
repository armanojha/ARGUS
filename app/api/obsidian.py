"""Obsidian Brain API surface (control layer).

Thin HTTP seam over the ARGUS Obsidian "brain" vault (``argus_brain_vault_path``).
This is ARGUS's dedicated, human-readable, structured long-term knowledge
layer — distinct from the user document corpus and from ARGUS machine memory.

The endpoint reports vault state (path, note count, recent notes) and is purely
read/deterministic on the control plane (no LLM calls).
"""

from __future__ import annotations

from datetime import UTC
from pathlib import Path

from fastapi import APIRouter, Depends
from pydantic import BaseModel, ConfigDict, Field

from app.config import Settings, get_settings
from app.memory import get_memory_factory_instance

router = APIRouter(prefix="/api/v1", tags=["obsidian-brain"])


def _is_configured(vault: Path) -> bool:
    """A vault counts as configured only when a real path is set.

    The default ``Path("")`` coerces to the process CWD (``.``), which must
    NOT be treated as an intentionally-configured brain vault.
    """
    raw = str(vault).strip()
    return bool(vault) and raw != "" and raw != "."


class ObsidianBrainStatus(BaseModel):
    """Status snapshot of ARGUS's dedicated Obsidian brain vault."""

    model_config = ConfigDict(extra="forbid")

    vault_path: str
    exists: bool
    configured: bool
    note_count: int
    write_back_root: str
    recent_notes: list[dict[str, object]] = Field(default_factory=list)


class ObsidianPromotionResponse(BaseModel):
    """Result of a selective memory->Obsidian brain promotion sweep."""

    model_config = ConfigDict(extra="forbid")

    notes_created: int
    notes_skipped: int
    notes_failed: int
    created_paths: list[str]
    skipped_ids: list[str]
    failed: list[str]


@router.get("/obsidian-brain/status", response_model=ObsidianBrainStatus)
def get_obsidian_brain_status(
    settings: Settings = Depends(get_settings),
) -> ObsidianBrainStatus:
    """Report the state of ARGUS's dedicated Obsidian brain vault."""
    vault = settings.argus_brain_vault_path
    configured = _is_configured(vault)
    exists = configured and vault.exists() and vault.is_dir()

    note_count = 0
    recent_notes: list[dict[str, object]] = []
    if exists:
        notes = sorted(vault.rglob("*.md"), key=lambda p: p.stat().st_mtime, reverse=True)
        note_count = len(notes)
        for note in notes[:15]:
            mtime = note.stat().st_mtime
            recent_notes.append(
                {
                    "path": str(note.relative_to(vault)),
                    "modified_at": note.stat().st_mtime,
                    "modified_iso": _iso_from_epoch(mtime),
                }
            )

    return ObsidianBrainStatus(
        vault_path=str(vault),
        exists=exists,
        configured=configured,
        note_count=note_count,
        write_back_root=settings.argus_brain_write_back_root,
        recent_notes=recent_notes,
    )


def _iso_from_epoch(epoch: float) -> str:
    from datetime import datetime

    return datetime.fromtimestamp(epoch, tz=UTC).isoformat()


@router.post("/obsidian-brain/promote", response_model=ObsidianPromotionResponse)
def promote_endpoint(
    settings: Settings = Depends(get_settings),
    limit: int = 50,
) -> ObsidianPromotionResponse:
    """Selectively promote eligible PROMOTED long-term memories into the brain vault."""
    from app.integrations.obsidian.promotion import promote_eligible_memories

    factory = get_memory_factory_instance()
    store = factory.create_memory_store() if factory is not None else None
    if store is None or not settings.memory_enabled:
        return ObsidianPromotionResponse(
            notes_created=0, notes_skipped=0, notes_failed=0,
            created_paths=[], skipped_ids=[], failed=["memory_disabled"],
        )

    result = promote_eligible_memories(store, vault_root=settings.argus_brain_vault_path, limit=limit)
    return ObsidianPromotionResponse(
        notes_created=result.notes_created,
        notes_skipped=result.notes_skipped,
        notes_failed=result.notes_failed,
        created_paths=result.created_paths,
        skipped_ids=result.skipped_ids,
        failed=result.failed,
    )