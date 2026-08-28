"""Phase 09 classifier + hypothesis research tests (deterministic, minimal).

Covers the Phase 09 acceptance surface without live LLMs: the 7-class
taxonomy, treatment rules, hypothesis conversion, classification-drives-
ingestion, and the hypothesis research runner mapped through the Phase 02
loop with a scripted provider.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from app.config import Settings
from app.evidence.models import Chunk, Document, Source, SourceType
from app.evidence.store import EvidenceStore
from app.integrations.obsidian.classifier import (
    RuleBasedHypothesisConverter,
    RuleBasedObsidianClassifier,
)
from app.integrations.obsidian.contracts import CLASSIFICATION_RULES, HypothesisResearchObjective
from app.integrations.obsidian.ingestion import ObsidianIngestionPipeline
from app.llm_gateway.capabilities import ProviderCapabilities
from app.llm_gateway.providers.models import CompletionResponse, Usage
from app.llm_gateway.quota import reset_quota_tracker
from app.llm_gateway.routing.router import LLMRouter
from app.reranking.reranker import NoOpReranker
from app.retrieval.bm25 import assign_bm25_doc_ids
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.vector import assign_embedding_indices


def _classify(content: str, note_path: str = "notes/note.md", frontmatter: dict | None = None):
    classifier = RuleBasedObsidianClassifier()
    return classifier.classify_sync(note_path, content, frontmatter or {}, [])


@pytest.mark.parametrize(
    ("note_path", "frontmatter", "content", "expected"),
    [
        ("notes/source.md", {"custom": {"author": "Alice", "url": "https://example.com"}}, "Some sourced content.", "source_note"),
        ("notes/source.md", {}, "See https://example.com/a and https://example.com/b.", "source_note"),
        ("notes/knowledge.md", {}, "Foxes are small-to-medium sized omnivores.", "knowledge_note"),
        ("notes/idea.md", {}, "I hypothesize that ARGUS scales with evidence density.", "hypothesis"),
        ("notes/project.md", {"custom": {"status": "active", "owner": "team"}}, "Working on the vault migration.", "project_note"),
        (
            "notes/question.md",
            {},
            "How does the planner work?\nWhy does the loop stop early?\nWho synthesized this?",
            "task_question",
        ),
        ("90_ARGUS/Research_Output/cap.md", {}, "Generated research output.", "research_capture"),
        (
            "notes/index.md",
            {},
            "# MOC\n\n- [[Alpha]]\n- [[Beta]]\n- [[Gamma]]\n- [[Delta]]\n- [[Epsilon]]",
            "reference_index",
        ),
    ],
)
def test_classifies_all_seven_classes(note_path, frontmatter, content, expected):
    result = _classify(content, note_path=note_path, frontmatter=frontmatter)
    assert result.knowledge_class == expected
    assert result.confidence > 0.5


def test_explicit_frontmatter_overrides_content_signals():
    result = _classify(
        "I hypothesize that scales matter.",
        note_path="notes/idea.md",
        frontmatter={"custom": {"knowledge_class": "knowledge_note"}},
    )
    assert result.knowledge_class == "knowledge_note"
    assert result.features.get("explicit_frontmatter") == "knowledge_note"


def test_treatment_rules_match_taxonomy():
    classifier = RuleBasedObsidianClassifier()
    for cls, rule in CLASSIFICATION_RULES.items():
        assert classifier.get_treatment_rule(cls.value) == rule.treatment.value
    assert set(classifier.get_all_rules()) == {cls.value for cls in CLASSIFICATION_RULES}


async def test_async_interface_matches_sync_core():
    classifier = RuleBasedObsidianClassifier()
    sync_result = classifier.classify_sync("notes/h.md", "I suspect this is true.", {}, [])
    async_result = await classifier.classify_note("notes/h.md", "I suspect this is true.", {}, [])
    assert sync_result.model_dump() == async_result.model_dump()


class TestHypothesisConverter:
    async def test_convert_hypothesis_produces_objective(self):
        converter = RuleBasedHypothesisConverter()
        objective = await converter.convert_hypothesis(
            "ARGUS becomes faster with cheaper retrieval windows.",
            "notes/idea.md",
        )
        assert objective.hypothesis_text.startswith("ARGUS becomes")
        assert "whether" in objective.research_objective
        assert len(objective.subquestions) == 3
        assert objective.suggested_patterns == ["hypothesis_research", "verification"]
        assert objective.priority == 0.8
        assert objective.source_note_path == "notes/idea.md"

    async def test_priority_taken_from_frontmatter(self):
        converter = RuleBasedHypothesisConverter()
        objective = await converter.convert_hypothesis(
            "X is y.",
            "notes/idea.md",
            context={"custom": {"priority": 0.3}},
        )
        assert objective.priority == 0.3

    def test_should_convert(self):
        converter = RuleBasedHypothesisConverter()
        assert converter.should_convert("hypothesis", {}) is True
        assert converter.should_convert("task_question", {}) is True
        assert converter.should_convert("knowledge_note", {}) is False
        assert converter.should_convert("hypothesis", {"custom": {"argus_research": False}}) is False
        assert converter.should_convert("hypothesis", {"custom": {"research": "false"}}) is False


class TestIngestionClassification:
    def test_classification_flows_into_records_and_chunks(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            vault = Path(tmpdir) / "vault"
            (vault / "Personal").mkdir(parents=True)
            (vault / "Personal" / "hypothesis.md").write_text(
                "# Hunch\n\nI hypothesize that evidence density accelerates insight.",
                encoding="utf-8",
            )
            (vault / "Personal" / "plain.md").write_text(
                "# Notes\n\nJust a personal note about the vault.",
                encoding="utf-8",
            )

            store = EvidenceStore(
                db_path=Path(tmpdir) / "evidence.db",
                bm25_index_path=Path(tmpdir) / "bm25.pkl",
                faiss_index_path=Path(tmpdir) / "faiss.index",
            )
            pipeline = ObsidianIngestionPipeline(
                vault,
                store=store,
                manifest_path=Path(tmpdir) / "manifest.pkl",
                classifier=RuleBasedObsidianClassifier(),
                enable_hypothesis_objectives=True,
            )
            result = pipeline.ingest_vault(incremental=False)

            assert result.notes_discovered == 2
            assert result.notes_classified == 2
            assert result.completed_at is not None

            records = pipeline.sync_manager.manifest.notes

            def record_for(suffix: str):
                return next(r for r in records.values() if r.vault_relative_path.endswith(suffix))

            assert record_for("hypothesis.md").knowledge_class == "hypothesis"
            assert record_for("plain.md").knowledge_class == "knowledge_note"

            hyp_claim = result.hypothesis_objectives[0]
            assert "whether" in hyp_claim.research_objective

            # Manifest + evidence document + chunk metadata all carry the class.
            hyp_record = record_for("hypothesis.md")
            document = store.get_latest_document_for_source(hyp_record.source_id)
            assert document.metadata["knowledge_class"] == "hypothesis"
            chunks = store.get_chunks_by_ids(hyp_record.chunk_ids)
            assert any(chunk.metadata.get("knowledge_class") == "hypothesis" for chunk in chunks)

            store.close()


# ---- scripted research run (mirrors cross-phase harness) ----

ANALYSIS_OK = {"complexity": "simple", "reasoning": "single fact lookup", "suggested_subquestion_count": 1}
FULL_SCRIPT = {
    "query_analysis": [ANALYSIS_OK],
    "research_planning": [
        {
            "objective": "Explain what the fox does.",
            "entities": ["fox"],
            "time_window": None,
            "subquestions": ["fox behavior"],
            "evidence_type": "factual",
            "preferred_retrieval_methods": ["hybrid"],
            "required_sources": [],
            "risk_level": "low",
            "token_budget": 6000,
            "iteration_budget": 5,
            "stopping_condition": "Stop once fox behavior is supported by evidence.",
        }
    ],
    "evidence_extraction": [{"sufficient": True, "reasoning": "test", "next_subquery": None}],
    "synthesis": ["Foxes jump over dogs [1]."],
}


class ScriptedProvider:
    """Fake provider returning scripted payloads per call type."""

    def __init__(self, script: dict) -> None:
        self._script = {k: list(v) for k, v in script.items()}
        self.name = "scripted"
        self.default_model = "scripted-model"
        self.calls: list[tuple[str, str]] = []

    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities()

    async def complete(
        self,
        messages,
        *,
        model=None,
        temperature=0.0,
        max_tokens=None,
        response_format=None,
        tools=None,
        tool_choice=None,
        timeout=30.0,
        call_type: str = "general",
        request_id=None,
    ) -> CompletionResponse:
        self.calls.append((call_type, request_id))
        queue = self._script.get(call_type)
        payload = queue.pop(0) if queue else {"fallback": True}
        content = __import__("json").dumps(payload) if isinstance(payload, dict) else payload
        return CompletionResponse(
            content=content,
            model=model or self.default_model,
            usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
            provider=self.name,
            request_id=request_id,
        )

    async def aclose(self) -> None:
        pass


@pytest.fixture(autouse=True)
def _reset_quota():
    reset_quota_tracker()
    yield
    reset_quota_tracker()


@pytest.fixture
def temp_dir() -> Path:
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def settings(temp_dir: Path) -> Settings:
    return Settings(
        _env_file=None,
        config_dir=temp_dir / "config",
        memory_db_path=temp_dir / "memory" / "memory.db",
        evidence_db_path=temp_dir / "evidence.db",
        bm25_index_path=temp_dir / "bm25.pkl",
        faiss_index_path=temp_dir / "faiss.index",
    )


@pytest.fixture
def populated_store(temp_dir: Path) -> EvidenceStore:
    store = EvidenceStore(
        db_path=temp_dir / "evidence.db",
        bm25_index_path=temp_dir / "bm25.pkl",
        faiss_index_path=temp_dir / "faiss.index",
    )
    source = Source(type=SourceType.TEXT, path="/test/corpus.txt", checksum="c1")
    store.upsert_source(source)
    doc = Document(source_id=source.id, version=1, checksum="d1", chunking_strategy="fixed")
    store.insert_document(doc)
    store.insert_chunks(
        [
            Chunk(document_id=doc.id, ordinal=0, text="The quick brown fox jumps over the lazy dog.", token_count=10),
            Chunk(document_id=doc.id, ordinal=1, text="Foxes are members of the Canidae family.", token_count=8),
        ]
    )
    assign_bm25_doc_ids(store)
    assign_embedding_indices(store)
    return store


class TestHypothesisResearchRunner:
    async def test_hypothesis_triggers_research_run(self, populated_store, settings):
        from app.integrations.obsidian.research import HypothesisResearchRunner

        provider = ScriptedProvider({k: list(v) for k, v in FULL_SCRIPT.items()})
        router = LLMRouter(provider)
        retriever = HybridRetriever(populated_store)

        objective = HypothesisResearchObjective(
            hypothesis_id="hyp-test123",
            hypothesis_text="Foxes jump over dogs.",
            research_objective="Determine whether foxes jump over dogs.",
            subquestions=["What do sources say about foxes?"],
            source_note_path="Personal/hypothesis.md",
        )
        runner = HypothesisResearchRunner(
            router=router,
            retriever=retriever,
            reranker=NoOpReranker(),
            settings=settings,
        )
        outcome = await runner.run(objective)

        assert outcome.status == "verified"
        assert outcome.answer
        assert len(outcome.citations) >= 1
        assert outcome.research_id == "research-hyp-test123"
        assert outcome.source_note_path == "Personal/hypothesis.md"
        populated_store.close()

    async def test_empty_corpus_degrades_to_undetermined(self, settings, temp_dir):
        """Research against an empty corpus must degrade to undetermined (no crash)."""
        from app.integrations.obsidian.research import HypothesisResearchRunner

        store = EvidenceStore(
            db_path=temp_dir / "evidence_empty.db",
            bm25_index_path=temp_dir / "bm25_empty.pkl",
            faiss_index_path=temp_dir / "faiss_empty.index",
        )
        provider = ScriptedProvider({k: list(v) for k, v in FULL_SCRIPT.items()})
        router = LLMRouter(provider)
        retriever = HybridRetriever(store)

        objective = HypothesisResearchObjective(
            hypothesis_id="hyp-empty",
            hypothesis_text="Foxes jump over dogs.",
            research_objective="Determine whether foxes jump over dogs.",
            source_note_path="Personal/hypothesis.md",
        )
        runner = HypothesisResearchRunner(
            router=router,
            retriever=retriever,
            reranker=NoOpReranker(),
            settings=settings,
        )
        outcome = await runner.run(objective)
        assert outcome.status in {"undetermined", "error"}
        store.close()

    async def test_runner_survives_provider_failure(self, settings, temp_dir):
        """Provider crashes propagate as an error outcome, not an exception."""
        from app.integrations.obsidian.research import HypothesisResearchRunner

        store = EvidenceStore(
            db_path=temp_dir / "evidence_fail.db",
            bm25_index_path=temp_dir / "bm25_fail.pkl",
            faiss_index_path=temp_dir / "faiss_fail.index",
        )
        retriever = HybridRetriever(store)

        class FailingProvider:
            @property
            def capabilities(self) -> ProviderCapabilities:
                return ProviderCapabilities()

            async def complete(self, messages, **kwargs) -> CompletionResponse:
                raise RuntimeError("boom")

        objective = HypothesisResearchObjective(
            hypothesis_id="hyp-fail",
            hypothesis_text="Foxes jump over dogs.",
            research_objective="Determine whether foxes jump over dogs.",
            source_note_path="Personal/hypothesis.md",
        )
        runner = HypothesisResearchRunner(
            router=LLMRouter(FailingProvider()),
            retriever=retriever,
            reranker=NoOpReranker(),
            settings=settings,
        )
        outcome = await runner.run(objective)
        assert outcome.status in {"error", "undetermined"}
        store.close()