"""Memory-Aware Planner Integration (Phase 08.3).

Enhances the Phase 02 planner with memory consultation:
- Retrieves relevant long-term knowledge for query entities/topics
- Surfaces relevant research history to avoid repeated research
- Injects source reliability context from source memory
- Incorporates user preferences from user memory
"""

from __future__ import annotations

from app.config import get_settings
from app.logging_config import get_logger
from app.memory.interfaces import (
    MemoryAwarePlannerInterface,
    MemoryLayer,
    MemoryQuery,
    MemoryRecord,
    MemoryStoreInterface,
)
from app.memory.store import get_memory_store
from app.orchestration.models import ResearchPlan

logger = get_logger("argus.memory.planner")


class MemoryAwarePlanner(MemoryAwarePlannerInterface):
    """Enhances research plans using persistent memory."""

    def __init__(
        self,
        memory_store: MemoryStoreInterface | None = None,
        max_memory_results: int = 5,
        min_confidence: float = 0.6,
    ):
        self.memory_store = memory_store
        self.max_memory_results = max_memory_results
        self.min_confidence = min_confidence
        self.settings = get_settings()

    async def enhance_plan_with_memory(
        self,
        plan: ResearchPlan,
        query: str,
        memory_store: MemoryStoreInterface,
    ) -> ResearchPlan:
        """Enhance a research plan using relevant memories."""
        # If memory_store is provided, use it regardless of global settings
        # This allows tests to work without enabling global memory
        if memory_store is None:
            return plan

        self.memory_store = memory_store
        enhancements = []

        # 1. Consult long-term knowledge for entities in the plan
        if plan.entities:
            entity_memories = await self._retrieve_entity_knowledge(plan.entities)
            if entity_memories:
                enhancements.append({
                    "type": "long_term_knowledge",
                    "entities": plan.entities,
                    "memories": [m.content for m in entity_memories],
                    "confidence": max(m.confidence for m in entity_memories),
                })

        # 2. Check research history for similar queries
        history_memories = await self._retrieve_research_history(query, plan.entities)
        if history_memories:
            enhancements.append({
                "type": "research_history",
                "previous_queries": [m.source_query for m in history_memories if m.source_query],
                "memories": [m.content for m in history_memories],
                "confidence": max(m.confidence for m in history_memories),
            })

        # 3. Get source reliability context if plan mentions specific sources
        if plan.required_sources:
            source_memories = await self._retrieve_source_memory(plan.required_sources)
            if source_memories:
                enhancements.append({
                    "type": "source_memory",
                    "sources": plan.required_sources,
                    "memories": [m.content for m in source_memories],
                    "confidence": max(m.confidence for m in source_memories),
                })

        # 4. Apply user preferences from user memory
        user_memories = await self._retrieve_user_preferences(query)
        if user_memories:
            enhancements.append({
                "type": "user_preferences",
                "preferences": [m.content for m in user_memories],
                "confidence": max(m.confidence for m in user_memories),
            })

        # 5. Build enhanced plan with memory context
        if enhancements:
            enhanced_plan = self._apply_enhancements(plan, enhancements)
            logger.info(
                "plan_enhanced_with_memory",
                query=query[:80],
                enhancement_types=[e["type"] for e in enhancements],
            )
            return enhanced_plan

        return plan

    async def _retrieve_entity_knowledge(self, entities: list[str]) -> list:
        """Retrieve long-term knowledge for plan entities."""
        assert self.memory_store is not None
        all_memories: list[MemoryRecord] = []
        for entity in entities[:5]:  # Limit to top 5 entities
            memories = await self.memory_store.retrieve(
                MemoryQuery(
                    query_text=entity,
                    layers=[MemoryLayer.LONG_TERM_KNOWLEDGE],
                    limit=self.max_memory_results,
                    min_confidence=self.min_confidence,
                )
            )
            all_memories.extend(memories)
        return all_memories

    async def _retrieve_research_history(self, query: str, entities: list[str]) -> list:
        """Retrieve relevant research history."""
        # Search by query text and entities
        assert self.memory_store is not None
        search_terms = [query] + entities[:3]
        all_memories: list[MemoryRecord] = []
        for term in search_terms:
            memories = await self.memory_store.retrieve(
                MemoryQuery(
                    query_text=term,
                    layers=[MemoryLayer.RESEARCH_HISTORY],
                    limit=self.max_memory_results,
                    min_confidence=self.min_confidence,
                )
            )
            all_memories.extend(memories)
        # Deduplicate by ID
        seen = set()
        unique = []
        for m in all_memories:
            if m.id not in seen:
                seen.add(m.id)
                unique.append(m)
        return unique

    async def _retrieve_source_memory(self, sources: list[str]) -> list:
        """Retrieve source reliability/bias memories."""
        assert self.memory_store is not None
        all_memories: list[MemoryRecord] = []
        for source in sources[:5]:
            memories = await self.memory_store.retrieve(
                MemoryQuery(
                    query_text=source,
                    layers=[MemoryLayer.SOURCE_MEMORY],
                    limit=self.max_memory_results,
                    min_confidence=self.min_confidence,
                )
            )
            all_memories.extend(memories)
        return all_memories

    async def _retrieve_user_preferences(self, query: str) -> list:
        """Retrieve user preferences relevant to query."""
        assert self.memory_store is not None
        return await self.memory_store.retrieve(
            MemoryQuery(
                query_text=query,
                layers=[MemoryLayer.USER_MEMORY],
                limit=self.max_memory_results,
                min_confidence=0.5,  # Lower threshold for preferences
            )
        )

    def _apply_enhancements(self, plan: ResearchPlan, enhancements: list[dict]) -> ResearchPlan:
        """Apply memory enhancements to the research plan."""
        # Build memory context string for the planner
        memory_context_parts = []
        for enh in enhancements:
            if enh["type"] == "long_term_knowledge" and enh["memories"]:
                memory_context_parts.append(
                    f"Known facts about entities {enh['entities']}: " + "; ".join(enh["memories"][:3])
                )
            elif enh["type"] == "research_history" and enh["memories"]:
                memory_context_parts.append(
                    "Previous research on similar topics: " + "; ".join(enh["memories"][:3])
                )
            elif enh["type"] == "source_memory" and enh["memories"]:
                memory_context_parts.append(
                    "Source context: " + "; ".join(enh["memories"][:2])
                )
            elif enh["type"] == "user_preferences" and enh["preferences"]:
                memory_context_parts.append(
                    "User preferences: " + "; ".join(enh["preferences"][:2])
                )

        if not memory_context_parts:
            return plan

        memory_context = "\n\n".join(memory_context_parts)

        # Enhance the plan by adding memory context to objective and subquestions
        enhanced_objective = plan.objective
        if len(memory_context) > 0:
            enhanced_objective = (
                f"{plan.objective}\n\n[Memory Context]\n{memory_context}"
            )

        # Add memory-informed subquestions if history suggests gaps
        enhanced_subquestions = list(plan.subquestions)
        for enh in enhancements:
            if enh["type"] == "research_history" and enh["previous_queries"]:
                # Could add follow-up questions based on past research gaps
                pass

        return plan.model_copy(
            update={
                "objective": enhanced_objective,
                "subquestions": enhanced_subquestions,
            }
        )


async def create_memory_aware_planner(
    memory_store: MemoryStoreInterface | None = None,
) -> MemoryAwarePlanner:
    """Factory function to create a memory-aware planner."""
    settings = get_settings()
    if not settings.memory_enabled:
        return MemoryAwarePlanner(memory_store=None)

    store = memory_store or get_memory_store()
    return MemoryAwarePlanner(
        memory_store=store,
        # Number of memory records to inject into the planning prompt per
        # category. This is a small prompt-budget cap, unrelated to the store's
        # per-layer capacity (`memory_max_records_per_layer`), so we use a sane
        # fixed value rather than deriving it from the store cap.
        max_memory_results=5,
        min_confidence=settings.memory_confidence_threshold,
    )


def inject_memory_into_planning_prompt(
    base_prompt: str,
    memory_context: str,
) -> str:
    """Inject memory context into a planner prompt template."""
    if not memory_context.strip():
        return base_prompt

    injection = (
        "\n\n=== RELEVANT MEMORY CONTEXT ===\n"
        f"{memory_context}\n"
        "=== END MEMORY CONTEXT ===\n"
        "Use the above memory context to inform your research plan. "
        "Avoid repeating research already done. Leverage known facts. "
        "Consider source reliability and user preferences.\n"
    )

    # Insert before the final instruction section if possible
    if "===" in base_prompt:
        parts = base_prompt.split("===")
        return "===".join(parts[:-1]) + injection + "===" + parts[-1]

    return base_prompt + injection