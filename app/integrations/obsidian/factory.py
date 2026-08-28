"""Concrete Phase 09 extension factory.

Implements ``ObsidianExtensionFactoryInterface`` from
``app/integrations/obsidian/contracts.py`` with the real Phase 09
components, so callers of ``get_obsidian_extension_factory()`` receive a
working classifier / converter / proposal manager / writer when full
Obsidian integration is enabled.

Registered at runtime via ``set_obsidian_extension_factory`` — by default
the app keeps ``DefaultObsidianExtensionFactory`` (no-op) so Phase 09
stays off until explicitly enabled.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from app.integrations.obsidian.contracts import ObsidianExtensionFactoryInterface


class ObsidianExtensionFactory(ObsidianExtensionFactoryInterface):
    """Returns concrete Phase 09 components (lazily imported)."""

    def __init__(
        self,
        vault_root: Path | str | None = None,
        write_back_root: str = "90_ARGUS",
    ) -> None:
        self.vault_root = Path(vault_root).resolve() if vault_root else None
        self.write_back_root = write_back_root

    def create_classifier(self) -> Any | None:
        from app.integrations.obsidian.classifier import RuleBasedObsidianClassifier

        return RuleBasedObsidianClassifier()

    def create_hypothesis_converter(self) -> Any | None:
        from app.integrations.obsidian.classifier import RuleBasedHypothesisConverter

        return RuleBasedHypothesisConverter()

    def create_writeback_proposal(self) -> Any | None:
        if self.vault_root is None:
            return None
        from app.integrations.obsidian.proposals import WriteBackProposalManager

        return WriteBackProposalManager(self.vault_root, self.write_back_root)

    def create_research_writer(self) -> Any | None:
        if self.vault_root is None:
            return None
        from app.integrations.obsidian.writer import ObsidianWriter

        return ObsidianWriter(self.vault_root, self.write_back_root)


__all__ = ["ObsidianExtensionFactory"]