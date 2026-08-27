"""Memory Factory and Integration (Phase 08).

Provides factory for creating memory components and integrates
with the existing Phase 00-05 architecture.
"""

from __future__ import annotations

from app.config import get_settings
from app.logging_config import get_logger
from app.memory.interfaces import (
    DefaultMemoryFactory,
    MemoryFactoryInterface,
    MemoryStoreInterface,
    VaultMemoryInterface,
    get_memory_factory,
    set_memory_factory,
)
from app.memory.store import MemoryStore, get_memory_store
from app.memory.versioning import GraphVersionManager, get_version_manager

logger = get_logger("argus.memory.factory")


class MemoryFactory(MemoryFactoryInterface):
    """Factory for creating memory components (Phase 08 implementation)."""

    def __init__(self):
        self._memory_store: MemoryStore | None = None
        self._version_manager: GraphVersionManager | None = None

    def create_memory_store(self) -> MemoryStoreInterface:
        """Create the memory store."""
        settings = get_settings()
        if not settings.memory_enabled:
            logger.info("memory_factory_disabled", reason="memory_enabled=false")
            return NullMemoryStore()

        if self._memory_store is None:
            self._memory_store = MemoryStore()
        return self._memory_store

    def create_vault_memory(self) -> VaultMemoryInterface | None:
        """Create vault memory coordinator (Phase 09), or None."""
        settings = get_settings()
        if not settings.obsidian_full_enabled:
            return None
        # Phase 09 will implement this
        logger.debug("vault_memory_not_implemented_yet")
        return None

    def get_version_manager(self) -> GraphVersionManager:
        """Get the graph version manager."""
        if self._version_manager is None:
            self._version_manager = GraphVersionManager()
        return self._version_manager

    def close(self) -> None:
        """Close all components."""
        self._memory_store = None
        self._version_manager = None


class NullMemoryStore(MemoryStoreInterface):
    """Null object implementation when memory is disabled."""

    async def store(self, record) -> None:
        pass

    async def retrieve(self, query) -> list:
        return []

    async def get_by_id(self, record_id: str):
        return None

    async def update(self, record) -> None:
        pass

    async def delete(self, record_id: str) -> bool:
        return False

    async def get_stats(self) -> dict:
        return {"enabled": False, "total_records": 0}


def initialize_memory_system() -> MemoryFactory:
    """Initialize the memory system and register the factory."""
    settings = get_settings()
    if not settings.memory_enabled:
        logger.info("memory_system_disabled")
        factory = DefaultMemoryFactory()
    else:
        factory = MemoryFactory()
        # Verify store can be created
        store = factory.create_memory_store()
        stats = store.get_stats() if hasattr(store, 'get_stats') else {}
        logger.info("memory_system_initialized", stats=stats)

    set_memory_factory(factory)
    return factory


def get_memory_factory_instance() -> MemoryFactory | DefaultMemoryFactory:
    """Get the current memory factory."""
    return get_memory_factory()


def shutdown_memory_system() -> None:
    """Shutdown the memory system."""
    factory = get_memory_factory()
    if hasattr(factory, 'close'):
        factory.close()
    set_memory_factory(DefaultMemoryFactory())
    logger.info("memory_system_shutdown")