"""Memory Package (Phase 08).

Persistent multi-layer memory and versioned graph updates.
"""

from app.memory.interfaces import (
    # Enums
    MemoryLayer,
    MemoryScope,
    # Models
    MemoryRecord,
    MemoryQuery,
    MemorySearchResult,
    VaultMemoryRecord,
    # Interfaces
    MemoryStoreInterface,
    MemoryAwarePlannerInterface,
    VaultMemoryInterface,
    MemoryFactoryInterface,
    DefaultMemoryFactory,
    get_memory_factory,
    set_memory_factory,
)
from app.memory.store import (
    SQLiteMemoryStore,
    get_memory_store,
    close_memory_store,
)
from app.memory.versioning import (
    DeltaType,
    DeltaStatus,
    GraphDelta,
    GraphVersionManager,
    get_version_manager,
    close_version_manager,
)
from app.memory.planner_integration import (
    MemoryAwarePlanner,
    create_memory_aware_planner,
    inject_memory_into_planning_prompt,
)
from app.memory.factory import (
    MemoryFactory,
    NullMemoryStore,
    initialize_memory_system,
    get_memory_factory_instance,
    shutdown_memory_system,
)

__all__ = [
    # Enums
    "MemoryLayer",
    "MemoryScope",
    "DeltaType",
    "DeltaStatus",
    # Models
    "MemoryRecord",
    "MemoryQuery",
    "MemorySearchResult",
    "VaultMemoryRecord",
    "GraphDelta",
    # Interfaces
    "MemoryStoreInterface",
    "MemoryAwarePlannerInterface",
    "VaultMemoryInterface",
    "MemoryFactoryInterface",
    "DefaultMemoryFactory",
    # Implementations
    "SQLiteMemoryStore",
    "GraphVersionManager",
    "MemoryAwarePlanner",
    "MemoryFactory",
    "NullMemoryStore",
    # Factories/Accessors
    "get_memory_factory",
    "set_memory_factory",
    "get_memory_store",
    "close_memory_store",
    "get_version_manager",
    "close_version_manager",
    "create_memory_aware_planner",
    "inject_memory_into_planning_prompt",
    "initialize_memory_system",
    "get_memory_factory_instance",
    "shutdown_memory_system",
]