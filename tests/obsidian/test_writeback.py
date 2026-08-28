"""Phase 09.4 safe write-back tests (propose-don't-mutate + promotion).

Verifies the acceptance criteria: proposals never mutate the target note,
accept applies the change, reject leaves it untouched, traversal is refused,
and the user-promotion path writes wikilink pointers (not content copies).
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from app.integrations.obsidian.models import ResearchCaptureNote
from app.integrations.obsidian.proposals import WriteBackProposalManager
from app.integrations.obsidian.writer import ObsidianWriter


@pytest.fixture
def vault() -> Path:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / "Personal").mkdir(parents=True)
        (root / "Personal" / "Note.md").write_text(
            "# Personal Note\n\nExisting content that the user wrote.\n\n## Status\n\nDraft.\n",
            encoding="utf-8",
        )
        yield root


class TestWriteBackProposalManager:
    async def test_create_proposal_never_mutates_target(self, vault: Path):
        target = vault / "Personal" / "Note.md"
        original = target.read_text(encoding="utf-8")

        manager = WriteBackProposalManager(vault)
        proposal = await manager.create_proposal(
            target_note_path="Personal/Note.md",
            change_type="append",
            proposed_content="ARGUS suggests adding a verified subsection.",
            research_id="research-abc",
            evidence_citations=[{"source": "x.md", "text": "quote"}],
            confidence=0.8,
        )

        assert proposal.status == "pending"
        assert target.read_text(encoding="utf-8") == original  # untouched
        proposal_file = vault / "90_ARGUS" / "Proposals" / f"{proposal.proposal_id}.md"
        assert proposal_file.exists()
        assert "ARGUS suggests" in proposal_file.read_text(encoding="utf-8")

    async def test_accept_applies_change_under_section(self, vault: Path):
        manager = WriteBackProposalManager(vault)
        proposal = await manager.create_proposal(
            target_note_path="Personal/Note.md",
            change_type="update_section",
            proposed_content="- new verified finding",
            section_heading="Status",
        )
        assert await manager.accept_proposal(proposal.proposal_id) is True

        updated = (vault / "Personal" / "Note.md").read_text(encoding="utf-8")
        assert "- new verified finding" in updated
        # Found inside the Status section, not appended elsewhere.
        assert "## Status\n\n- new verified finding" in updated
        accepted = await manager.get_proposal(proposal.proposal_id)
        assert accepted.status == "accepted"
        assert accepted.reviewed_by == "user"

    async def test_accept_creates_missing_section(self, vault: Path):
        manager = WriteBackProposalManager(vault)
        proposal = await manager.create_proposal(
            target_note_path="Personal/Note.md",
            change_type="update_section",
            proposed_content="Fresh section body.",
            section_heading="Research Notes",
        )
        assert await manager.accept_proposal(proposal.proposal_id) is True
        updated = (vault / "Personal" / "Note.md").read_text(encoding="utf-8")
        assert "## Research Notes" in updated

    async def test_reject_leaves_target_untouched(self, vault: Path):
        target = vault / "Personal" / "Note.md"
        original = target.read_text(encoding="utf-8")

        manager = WriteBackProposalManager(vault)
        proposal = await manager.create_proposal(
            target_note_path="Personal/Note.md",
            change_type="append",
            proposed_content="This should not be written.",
        )
        assert await manager.reject_proposal(proposal.proposal_id) is True
        assert target.read_text(encoding="utf-8") == original
        rejected = await manager.get_proposal(proposal.proposal_id)
        assert rejected.status == "rejected"

    async def test_list_proposals_filters_by_status(self, vault: Path):
        manager = WriteBackProposalManager(vault)
        proposal = await manager.create_proposal(
            target_note_path="Personal/Note.md",
            change_type="append",
            proposed_content="One.",
        )
        await manager.accept_proposal(proposal.proposal_id)
        pending = await manager.list_proposals(status="pending")
        accepted = await manager.list_proposals(status="accepted")
        assert not pending
        assert [p.proposal_id for p in accepted] == [proposal.proposal_id]

    async def test_target_outside_vault_is_refused(self, vault: Path):
        manager = WriteBackProposalManager(vault)
        with pytest.raises(ValueError):
            await manager.create_proposal(
                target_note_path="../../outside.md",
                change_type="append",
                proposed_content="boom",
            )

    async def test_proposal_for_missing_target_is_refused(self, vault: Path):
        manager = WriteBackProposalManager(vault)
        with pytest.raises(FileNotFoundError):
            await manager.create_proposal(
                target_note_path="Personal/Missing.md",
                change_type="append",
                proposed_content="boom",
            )


class TestPromotionPath:
    def test_promotion_requires_user_decision(self, vault: Path):
        writer = ObsidianWriter(vault)
        capture = ResearchCaptureNote(
            research_id="research-x",
            title="Fox findings",
            query="q",
            answer="The long answer body that must not be copied.",
            sources=["Sources/fox-source.md"],
            claims=["Claim 1"],
        )
        assert writer.promote_research_capture(capture, vault / "Knowledge", user_decision=False) is None

    def test_promotion_writes_wikilink_pointer_not_content_copy(self, vault: Path):
        writer = ObsidianWriter(vault)
        capture = ResearchCaptureNote(
            research_id="research-x",
            title="Fox findings",
            query="q",
            answer="The long answer body that must not be copied.",
            confidence=0.9,
            sources=["Sources/fox-source.md"],
            claims=["Foxes are canids."],
        )
        out = writer.promote_research_capture(capture, vault / "Knowledge")
        assert out is not None
        assert out.exists()
        assert out.parent.name == "Knowledge"
        content = out.read_text(encoding="utf-8")
        # Wikilink back to the source note.
        assert "[[fox-source|Sources/fox-source.md]]" in content
        # Claims survive as pointers; the full answer body is not copied.
        assert "Foxes are canids." in content
        assert "The long answer body" not in content

    def test_promotion_refuses_write_back_area(self, vault: Path):
        writer = ObsidianWriter(vault)
        capture = ResearchCaptureNote(
            research_id="research-y",
            title="Y",
            query="q",
            answer="a",
            sources=[],
            claims=[],
        )
        with pytest.raises(ValueError):
            writer.promote_research_capture(capture, vault / "90_ARGUS" / "Research_Output")