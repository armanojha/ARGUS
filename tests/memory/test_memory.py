"""Tests for Phase 08 Memory System."""

from __future__ import annotations

import tempfile
from pathlib import Path
from uuid import uuid4

import pytest

from app.memory import (
    DeltaStatus,
    DeltaType,
    GraphDelta,
    GraphVersionManager,
    MemoryAwarePlanner,
    MemoryFactory,
    MemoryLayer,
    MemoryPromotionStatus,
    MemoryQuery,
    MemoryRecord,
    MemoryScope,
    MemoryStore,
)
from app.memory.factory import (
    NullMemoryStore,
    get_memory_factory,
    initialize_memory_system,
    shutdown_memory_system,
)
from app.memory.interfaces import DefaultMemoryFactory


class TestSQLiteMemoryStore:
    """Tests for the SQLite memory store."""

    @pytest.fixture
    def temp_db(self):
        """Create a temporary database for testing."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = Path(f.name)
        yield db_path
        # Cleanup
        if db_path.exists():
            try:
                db_path.unlink()
            except PermissionError:
                pass  # Windows may lock the file briefly

    @pytest.fixture
    def memory_store(self, temp_db):
        """Create a memory store with temp database."""
        store = MemoryStore(db_path=temp_db, max_records_per_layer=100)
        yield store
        # Close the connection pool to release the file lock
        store.close()

    @pytest.fixture
    def sample_record(self):
        """Create a sample memory record."""
        return MemoryRecord(
            id=uuid4(),
            layer=MemoryLayer.LONG_TERM_KNOWLEDGE,
            scope=MemoryScope.GLOBAL,
            content="The capital of France is Paris.",
            subject="France",
            predicate="capital",
            object="Paris",
            supporting_chunk_ids=[str(uuid4()), str(uuid4())],
            source_query="What is the capital of France?",
            confidence=0.9,
            tags=["geography", "fact"],
            metadata={"source": "wikipedia"},
        )

    @pytest.mark.asyncio
    async def test_store_and_retrieve(self, memory_store, sample_record):
        """Test storing and retrieving a memory record."""
        await memory_store.store(sample_record)

        results = await memory_store.retrieve(
            MemoryQuery(
                query_text="France",
                layers=[MemoryLayer.LONG_TERM_KNOWLEDGE],
                limit=10,
            )
        )

        assert len(results) == 1
        assert results[0].id == sample_record.id
        assert results[0].content == sample_record.content
        assert results[0].confidence == sample_record.confidence

    @pytest.mark.asyncio
    async def test_retrieve_by_layer_filter(self, memory_store):
        """Test retrieval filtered by layer."""
        record1 = MemoryRecord(
            id=uuid4(),
            layer=MemoryLayer.LONG_TERM_KNOWLEDGE,
            content="Fact 1",
            confidence=0.9,
        )
        record2 = MemoryRecord(
            id=uuid4(),
            layer=MemoryLayer.RESEARCH_HISTORY,
            content="Previous query about X",
            confidence=0.8,
        )

        await memory_store.store(record1)
        await memory_store.store(record2)

        # Query only long-term knowledge
        results = await memory_store.retrieve(
            MemoryQuery(
                query_text="",
                layers=[MemoryLayer.LONG_TERM_KNOWLEDGE],
                limit=10,
            )
        )
        assert len(results) == 1
        assert results[0].layer == MemoryLayer.LONG_TERM_KNOWLEDGE

    @pytest.mark.asyncio
    async def test_retrieve_by_confidence_threshold(self, memory_store):
        """Test retrieval with minimum confidence filter."""
        record1 = MemoryRecord(id=uuid4(), layer=MemoryLayer.LONG_TERM_KNOWLEDGE, content="High confidence", confidence=0.9)
        record2 = MemoryRecord(id=uuid4(), layer=MemoryLayer.LONG_TERM_KNOWLEDGE, content="Low confidence", confidence=0.3)

        await memory_store.store(record1)
        await memory_store.store(record2)

        results = await memory_store.retrieve(
            MemoryQuery(
                query_text="",
                layers=[MemoryLayer.LONG_TERM_KNOWLEDGE],
                min_confidence=0.5,
                limit=10,
            )
        )
        assert len(results) == 1
        assert results[0].confidence >= 0.5

    @pytest.mark.asyncio
    async def test_get_by_id(self, memory_store, sample_record):
        """Test retrieving a record by ID."""
        await memory_store.store(sample_record)

        retrieved = await memory_store.get_by_id(str(sample_record.id))
        assert retrieved is not None
        assert retrieved.id == sample_record.id

        # Non-existent ID
        missing = await memory_store.get_by_id(str(uuid4()))
        assert missing is None

    @pytest.mark.asyncio
    async def test_update_record(self, memory_store, sample_record):
        """Test updating a memory record."""
        await memory_store.store(sample_record)

        # Update confidence - create new record with updated values
        updated = MemoryRecord(
            **sample_record.model_dump(exclude={"confidence"}),
            confidence=0.95,
        )
        await memory_store.update(updated)

        retrieved = await memory_store.get_by_id(str(sample_record.id))
        assert retrieved.confidence == 0.95

    @pytest.mark.asyncio
    async def test_delete_record(self, memory_store, sample_record):
        """Test deleting a memory record."""
        await memory_store.store(sample_record)

        deleted = await memory_store.delete(str(sample_record.id))
        assert deleted is True

        retrieved = await memory_store.get_by_id(str(sample_record.id))
        assert retrieved is None

        # Delete non-existent
        deleted = await memory_store.delete(str(uuid4()))
        assert deleted is False

    @pytest.mark.asyncio
    async def test_layer_limit_enforcement(self, temp_db):
        """Test that layer limits are enforced."""
        store = MemoryStore(db_path=temp_db, max_records_per_layer=3)

        # Store 5 records in the same layer
        for i in range(5):
            record = MemoryRecord(
                id=uuid4(),
                layer=MemoryLayer.WORKING,
                content=f"Working memory {i}",
                confidence=0.5 + i * 0.1,
            )
            await store.store(record)

        stats = await store.get_stats()
        # Should have pruned to limit + batch
        assert stats["layer_counts"].get("working", 0) <= 3

    @pytest.mark.asyncio
    async def test_promote_memory(self, memory_store, sample_record):
        """Test promoting a memory to higher confidence."""
        # Create a new record with lower confidence (can't modify frozen record)
        low_conf_record = MemoryRecord(
            **sample_record.model_dump(exclude={"confidence"}),
            confidence=0.6,
        )
        await memory_store.store(low_conf_record)

        promoted = await memory_store.promote_memory(str(low_conf_record.id), 0.9, "Verified by multiple sources")
        assert promoted is True

        retrieved = await memory_store.get_by_id(str(low_conf_record.id))
        assert retrieved.confidence == 0.9

    @pytest.mark.asyncio
    async def test_get_stats(self, memory_store):
        """Test getting memory store statistics."""
        # Add some records
        for layer in [MemoryLayer.LONG_TERM_KNOWLEDGE, MemoryLayer.RESEARCH_HISTORY, MemoryLayer.USER_MEMORY]:
            for i in range(2):
                record = MemoryRecord(
                    id=uuid4(),
                    layer=layer,
                    content=f"Content {i}",
                    confidence=0.7,
                )
                await memory_store.store(record)

        stats = await memory_store.get_stats()
        assert stats["total_records"] == 6
        assert len(stats["layer_counts"]) == 3
        assert stats["avg_confidence"] > 0

    @pytest.mark.asyncio
    async def test_text_search(self, memory_store):
        """Test text-based retrieval."""
        record1 = MemoryRecord(
            id=uuid4(),
            layer=MemoryLayer.LONG_TERM_KNOWLEDGE,
            content="The mitochondria is the powerhouse of the cell",
            confidence=0.9,
        )
        record2 = MemoryRecord(
            id=uuid4(),
            layer=MemoryLayer.LONG_TERM_KNOWLEDGE,
            content="The nucleus contains DNA",
            confidence=0.8,
        )

        await memory_store.store(record1)
        await memory_store.store(record2)

        results = await memory_store.retrieve(
            MemoryQuery(
                query_text="mitochondria",
                layers=[MemoryLayer.LONG_TERM_KNOWLEDGE],
                limit=10,
            )
        )
        assert len(results) == 1
        assert "mitochondria" in results[0].content.lower()

    @pytest.mark.asyncio
    async def test_fresh_evidence_wins_supersedes_older(self, memory_store):
        """Test that newer contradicting evidence supersedes (archives) the older record."""
        older = MemoryRecord(
            id=uuid4(),
            layer=MemoryLayer.LONG_TERM_KNOWLEDGE,
            content="The CEO of Acme is A.",
            subject="Acme",
            predicate="ceo",
            object="A",
            confidence=0.8,
        )
        newer = MemoryRecord(
            id=uuid4(),
            layer=MemoryLayer.LONG_TERM_KNOWLEDGE,
            content="The CEO of Acme is B.",
            subject="Acme",
            predicate="ceo",
            object="B",
            confidence=0.9,
        )
        await memory_store.store(older)
        await memory_store.store(newer)

        archived = await memory_store.get_by_id(str(older.id))
        assert archived is not None
        assert archived.promotion_status == MemoryPromotionStatus.ARCHIVED
        assert archived.superseded_by_id == newer.id

        current = await memory_store.get_by_id(str(newer.id))
        assert current is not None
        assert current.supersedes_id == older.id

    @pytest.mark.asyncio
    async def test_fresh_evidence_keeps_old_when_newer_has_lower_confidence_age(self, memory_store):
        """Older contradicting evidence is not superseded by a lower-confidence record."""
        older = MemoryRecord(
            id=uuid4(),
            layer=MemoryLayer.LONG_TERM_KNOWLEDGE,
            content="The capital is A.",
            subject="Country",
            predicate="capital",
            object="A",
            confidence=0.7,
        )
        weaker = MemoryRecord(
            id=uuid4(),
            layer=MemoryLayer.LONG_TERM_KNOWLEDGE,
            content="The capital is B.",
            subject="Country",
            predicate="capital",
            object="B",
            confidence=0.3,
        )
        await memory_store.store(older)
        await memory_store.store(weaker)

        kept = await memory_store.get_by_id(str(older.id))
        assert kept.promotion_status != MemoryPromotionStatus.ARCHIVED
        assert kept.superseded_by_id is None


class TestGraphVersionManager:
    """Tests for the graph versioning system."""

    @pytest.fixture
    def temp_db(self):
        """Create a temporary database for testing."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = Path(f.name)
        yield db_path
        if db_path.exists():
            db_path.unlink()

    @pytest.fixture
    def version_manager(self, temp_db):
        """Create a version manager with temp database."""
        return GraphVersionManager(db_path=temp_db, confidence_threshold=0.7)

    def test_record_delta(self, version_manager):
        """Test recording a graph delta."""
        target_id = uuid4()
        delta = GraphDelta(
            id=uuid4(),
            delta_type=DeltaType.CLAIM_CREATED,
            status=DeltaStatus.PROVISIONAL,
            target_id=target_id,
            target_type="claim",
            previous_state=None,
            new_state={"text": "Test claim", "confidence": 0.8},
            supporting_chunk_ids=[uuid4()],
            source_query="Test query",
            confidence=0.8,
        )

        recorded = version_manager.record_delta(delta)
        assert recorded.id == delta.id
        # High confidence (0.8 > 0.7 threshold) gets auto-promoted
        assert recorded.status == DeltaStatus.PROMOTED

    def test_auto_promote_high_confidence(self, version_manager):
        """Test that high-confidence deltas are auto-promoted."""
        target_id = uuid4()
        delta = GraphDelta(
            id=uuid4(),
            delta_type=DeltaType.CLAIM_CREATED,
            status=DeltaStatus.PROVISIONAL,
            target_id=target_id,
            target_type="claim",
            new_state={"text": "High confidence claim", "confidence": 0.9},
            confidence=0.9,  # Above threshold of 0.7
        )

        recorded = version_manager.record_delta(delta)
        # Should be auto-promoted
        assert recorded.status == DeltaStatus.PROMOTED

    def test_get_delta(self, version_manager):
        """Test retrieving a delta by ID."""
        target_id = uuid4()
        delta = GraphDelta(
            id=uuid4(),
            delta_type=DeltaType.ENTITY_CREATED,
            status=DeltaStatus.PROVISIONAL,
            target_id=target_id,
            target_type="entity",
            new_state={"canonical_name": "Test Entity"},
            confidence=0.6,
        )

        version_manager.record_delta(delta)
        retrieved = version_manager.get_delta(delta.id)
        assert retrieved is not None
        assert retrieved.target_id == target_id

    def test_get_deltas_for_target(self, version_manager):
        """Test getting all deltas for a target."""
        target_id = uuid4()

        # Record multiple deltas for same target
        for i in range(3):
            delta = GraphDelta(
                id=uuid4(),
                delta_type=DeltaType.CLAIM_UPDATED,
                status=DeltaStatus.PROVISIONAL,
                target_id=target_id,
                target_type="claim",
                new_state={"version": i, "confidence": 0.5 + i * 0.1},
                confidence=0.5 + i * 0.1,
            )
            version_manager.record_delta(delta)

        deltas = version_manager.get_deltas_for_target(target_id, "claim")
        assert len(deltas) == 3
        # Should be ordered by sequence
        assert deltas[0].new_state["version"] == 0
        assert deltas[2].new_state["version"] == 2

    def test_promote_delta(self, version_manager):
        """Test manually promoting a delta."""
        target_id = uuid4()
        delta = GraphDelta(
            id=uuid4(),
            delta_type=DeltaType.CLAIM_CREATED,
            status=DeltaStatus.PROVISIONAL,
            target_id=target_id,
            target_type="claim",
            new_state={"text": "Claim"},
            confidence=0.6,
        )
        version_manager.record_delta(delta)

        promoted = version_manager.promote_delta(delta.id, "manual")
        assert promoted is True

        retrieved = version_manager.get_delta(delta.id)
        assert retrieved.status == DeltaStatus.PROMOTED
        assert retrieved.promoted_by == "manual"

    def test_reject_delta(self, version_manager):
        """Test rejecting a provisional delta."""
        target_id = uuid4()
        delta = GraphDelta(
            id=uuid4(),
            delta_type=DeltaType.CLAIM_CREATED,
            status=DeltaStatus.PROVISIONAL,
            target_id=target_id,
            target_type="claim",
            new_state={"text": "Claim"},
            confidence=0.6,
        )
        version_manager.record_delta(delta)

        rejected = version_manager.reject_delta(delta.id, "Contradicted by evidence")
        assert rejected is True

        retrieved = version_manager.get_delta(delta.id)
        assert retrieved.status == DeltaStatus.REJECTED

    def test_claim_version_history(self, version_manager):
        """Test claim version history tracking."""
        claim_id = uuid4()

        # Record multiple versions
        for i in range(3):
            delta = GraphDelta(
                id=uuid4(),
                delta_type=DeltaType.CLAIM_UPDATED if i > 0 else DeltaType.CLAIM_CREATED,
                status=DeltaStatus.PROMOTED,
                target_id=claim_id,
                target_type="claim",
                new_state={"text": f"Claim version {i}", "confidence": 0.7 + i * 0.1},
                confidence=0.7 + i * 0.1,
            )
            version_manager.record_delta(delta)

        history = version_manager.get_claim_history(claim_id)
        assert len(history) == 3
        assert history[0]["version"] == 1
        assert history[2]["is_current"] == 1
        assert history[0]["is_current"] == 0

    def test_auto_promote_eligible(self, version_manager):
        """Test batch auto-promotion."""
        # Add several provisional deltas with varying confidence
        # Note: 0.8 and 0.9 will be auto-promoted during record_delta
        for i, conf in enumerate([0.6, 0.5, 0.4, 0.3]):
            delta = GraphDelta(
                id=uuid4(),
                delta_type=DeltaType.CLAIM_CREATED,
                status=DeltaStatus.PROVISIONAL,
                target_id=uuid4(),
                target_type="claim",
                new_state={"text": f"Claim {i}"},
                confidence=conf,
            )
            version_manager.record_delta(delta)

        # Add some more with high confidence but they won't be auto-promoted
        # because they were already promoted during record_delta
        # Now test auto_promote_eligible on the remaining provisional ones
        promoted_count = version_manager.auto_promote_eligible()
        # All are below threshold, so 0
        assert promoted_count == 0

    def test_get_stats(self, version_manager):
        """Test versioning statistics."""
        # Add deltas - some with high confidence (auto-promoted), some low
        for i in range(5):
            conf = 0.5 + i * 0.1  # 0.5, 0.6, 0.7, 0.8, 0.9
            status = DeltaStatus.PROMOTED if i >= 3 else DeltaStatus.PROVISIONAL
            delta = GraphDelta(
                id=uuid4(),
                delta_type=DeltaType.CLAIM_CREATED,
                status=status,
                target_id=uuid4(),
                target_type="claim",
                new_state={"text": f"Claim {i}"},
                confidence=conf,
            )
            version_manager.record_delta(delta)

        stats = version_manager.get_stats()
        assert stats["total_deltas"] == 5
        # 0.7, 0.8, 0.9 are >= 0.7 threshold, so they get auto-promoted
        # That's 3 promoted, 2 provisional
        assert stats["by_status"]["provisional"] == 2
        assert stats["by_status"]["promoted"] == 3


class TestMemoryAwarePlanner:
    """Tests for the memory-aware planner integration."""

    @pytest.fixture
    def memory_store(self):
        """Create a memory store for testing."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = Path(f.name)
        store = MemoryStore(db_path=db_path, max_records_per_layer=100)
        yield store
        # Close the connection pool to release the file lock
        store.close()
        if db_path.exists():
            try:
                db_path.unlink()
            except PermissionError:
                pass

    @pytest.fixture
    def planner(self, memory_store):
        """Create a memory-aware planner."""
        return MemoryAwarePlanner(
            memory_store=memory_store,
            max_memory_results=3,
            min_confidence=0.5,
        )

    @pytest.mark.asyncio
    async def test_enhance_plan_with_entity_knowledge(self, planner, memory_store):
        """Test plan enhancement with long-term knowledge."""
        from app.orchestration.models import ResearchPlan

        # Add some long-term knowledge
        entity_record = MemoryRecord(
            id=uuid4(),
            layer=MemoryLayer.LONG_TERM_KNOWLEDGE,
            content="Apple Inc. was founded in 1976 by Steve Jobs and Steve Wozniak.",
            subject="Apple Inc.",
            predicate="founded_in",
            object="1976",
            confidence=0.9,
        )
        await memory_store.store(entity_record)

        plan = ResearchPlan(
            objective="When was Apple founded?",
            entities=["Apple Inc."],
            subquestions=["When was Apple founded?", "Who founded Apple?"],
        )

        enhanced = await planner.enhance_plan_with_memory(plan, "When was Apple founded?", memory_store)
        assert "1976" in enhanced.objective or "1976" in str(enhanced.subquestions)

    @pytest.mark.asyncio
    async def test_enhance_plan_with_research_history(self, planner, memory_store):
        """Test plan enhancement with research history."""
        from app.orchestration.models import ResearchPlan

        # Add research history
        history_record = MemoryRecord(
            id=uuid4(),
            layer=MemoryLayer.RESEARCH_HISTORY,
            content="Previously researched: Apple founding date is well-documented as April 1, 1976.",
            source_query="When was Apple founded?",
            confidence=0.8,
        )
        await memory_store.store(history_record)

        plan = ResearchPlan(
            objective="When was Apple founded?",
            entities=["Apple Inc."],
            subquestions=["When was Apple founded?"],
        )

        enhanced = await planner.enhance_plan_with_memory(plan, "When was Apple founded?", memory_store)
        assert "Previously researched" in enhanced.objective or "April 1, 1976" in enhanced.objective


class TestMemoryFactory:
    """Tests for the memory factory."""

    def test_default_factory_returns_none(self):
        """Test DefaultMemoryFactory returns None."""
        factory = DefaultMemoryFactory()
        assert factory.create_memory_store() is None
        assert factory.create_vault_memory() is None

    def test_memory_factory_creates_store(self):
        """Test MemoryFactory creates memory store."""
        factory = MemoryFactory()
        store = factory.create_memory_store()
        # May return NullMemoryStore if disabled
        assert store is not None
        assert hasattr(store, 'store')
        assert hasattr(store, 'retrieve')

    @pytest.mark.asyncio
    async def test_initialize_and_shutdown(self):
        """Test memory system initialization and shutdown."""
        # This uses global state, so just test it doesn't crash
        factory = await initialize_memory_system()
        assert factory is not None
        await shutdown_memory_system()
        factory2 = get_memory_factory()
        assert isinstance(factory2, DefaultMemoryFactory)


class TestNullMemoryStore:
    """Tests for the NullMemoryStore (disabled memory)."""

    @pytest.mark.asyncio
    async def test_null_store_operations(self):
        """Test that null store operations are no-ops."""
        store = NullMemoryStore()

        record = MemoryRecord(
            id=uuid4(),
            layer=MemoryLayer.LONG_TERM_KNOWLEDGE,
            content="Test",
            confidence=0.5,
        )

        await store.store(record)
        results = await store.retrieve(MemoryQuery(query_text="test", limit=10))
        assert results == []

        retrieved = await store.get_by_id(str(record.id))
        assert retrieved is None

        deleted = await store.delete(str(record.id))
        assert deleted is False

        stats = await store.get_stats()
        assert stats["enabled"] is False