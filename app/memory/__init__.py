"""Memory Package (Phase 08).

Persistent multi-layer memory and versioned graph updates.
"""

from app.memory.factory import (
    MemoryFactory,
    NullMemoryStore,
    get_memory_factory_instance,
    initialize_memory_system,
    shutdown_memory_system,
)
from app.memory.interfaces import (
    DefaultMemoryFactory,
    MemoryAwarePlannerInterface,
    MemoryFactoryInterface,
    # Enums
    MemoryLayer,
    MemoryQuery,
    # Models
    MemoryRecord,
    MemoryScope,
    MemorySearchResult,
    # Interfaces
    MemoryStoreInterface,
    VaultMemoryInterface,
    VaultMemoryRecord,
    get_memory_factory,
    set_memory_factory,
)
from app.memory.planner_integration import (
    MemoryAwarePlanner,
    create_memory_aware_planner,
    inject_memory_into_planning_prompt,
)
from app.memory.store import (
    MemoryStore,
    close_memory_store,
    get_memory_store,
)
from app.memory.versioning import (
    DeltaStatus,
    DeltaType,
    GraphDelta,
    GraphVersionManager,
    close_version_manager,
    get_version_manager,
)

__all__ = [
    "DefaultMemoryFactory",
    "DeltaStatus",
    "DeltaType",
    "GraphDelta",
    "GraphVersionManager",
    "MemoryAwarePlanner",
    "MemoryAwarePlannerInterface",
    "MemoryFactory",
    "MemoryFactoryInterface",
    "MemoryLayer",
    "MemoryQuery",
    "MemoryRecord",
    "MemoryScope",
    "MemorySearchResult",
    "MemoryStore",
    "MemoryStoreInterface",
    "NullMemoryStore",
    "VaultMemoryInterface",
    "VaultMemoryRecord",
    "close_memory_store",
    "close_version_manager",
    "create_memory_aware_planner",
    "get_memory_factory",
    "get_memory_factory_instance",
    "get_memory_store",
    "get_version_manager",
    "initialize_memory_system",
    "inject_memory_into_planning_prompt",
    "set_memory_factory",
    "shutdown_memory_system",
]