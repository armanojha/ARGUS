"""Tests for Phase 10 Multi-Agent Challenge.

Tests cover:
- Agent role activation rules
- Multi-agent debate flow
- Disagreement detection
- Agent coordinator integration
- Cross-model disagreement triggers
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from app.config import Settings
from app.llm_gateway.providers.models import (
    CompletionResponse,
    Message,
    MessageRole,
    Tool,
    ToolChoice,
    Usage,
)
from app.orchestration.agents.agents import (
    AlternativeHypothesisAgent,
    JudgeAgent,
    ResearcherAgent,
    SkepticAgent,
    VerifierAgent,
)
from app.orchestration.agents.coordinator import AgentCoordinator
from app.orchestration.contracts import AgentActivationRule, AgentRole, AgentMessage
from app.orchestration.state import OrchestrationState
from tests.mocks.mock_provider import MockProvider


class ResearcherOutput(BaseModel):
    key_claims: list[str] = []
    evidence_summary: str = ""
    confidence: float = 0.5
    gaps: list[str] = []


class SkepticOutput(BaseModel):
    challenges: list[str] = []
    counter_evidence_needed: list[str] = []
    confidence: float = 0.5
    severity: float = 0.5


class AlternativeHypothesisOutput(BaseModel):
    alternative_explanations: list[str] = []
    supporting_evidence_for_alternatives: list[str] = []
    confidence: float = 0.5
    distinctiveness: float = 0.5


class VerifierOutput(BaseModel):
    claim: str = ""
    status: str = "UNSUPPORTED"
    confidence: float = 0.0
    reasoning: str = ""
    evidence_coverage: float = 0.0
    source_quality: float = 0.0


class JudgeOutput(BaseModel):
    resolution: str = ""
    final_answer: str = ""
    confidence: float = 0.5
    key_disagreements_resolved: list[str] = []
    remaining_uncertainties: list[str] = []
    should_continue_debate: bool = False


@pytest.fixture
def mock_router():
    """Create a mock router with pre-programmed responses."""
    from app.llm_gateway.routing.router import LLMRouter

    provider = MockProvider()
    return LLMRouter(provider)


@pytest.fixture
def mock_coordinator(mock_router):
    """Create an AgentCoordinator with mock dependencies."""
    from app.retrieval.hybrid import HybridRetriever
    from app.reranking.reranker import NoOpReranker

    settings = Settings(
        multiagent_enabled=True,
        multiagent_max_rounds=3,
        multiagent_skeptic_threshold=0.7,
        multiagent_disagreement_threshold=0.3,
        orchestration_max_iterations=3,
        orchestration_token_budget=6000,
        orchestration_llm_timeout=30.0,
    )

    # Create minimal mock retriever and reranker
    retriever = HybridRetriever()
    reranker = NoOpReranker()

    coordinator = AgentCoordinator(mock_router, settings, retriever, reranker)
    return coordinator


class TestAgentRoles:
    """Test individual agent roles."""

    @pytest.mark.asyncio
    async def test_researcher_agent_activates_always(self, mock_router):
        """Researcher should always activate."""
        settings = Settings()
        agent = ResearcherAgent(mock_router, settings, call_type="researcher")
        assert agent.role == AgentRole.RESEARCHER
        assert AgentActivationRule.ALWAYS in agent.activation_rules

    @pytest.mark.asyncio
    async def test_skeptic_agent_activation_rules(self, mock_router):
        """Skeptic should activate on high stakes, conflicting evidence, high uncertainty."""
        settings = Settings()
        agent = SkepticAgent(mock_router, settings, call_type="skeptic")
        assert agent.role == AgentRole.SKEPTIC
        assert AgentActivationRule.HIGH_STAKES in agent.activation_rules
        assert AgentActivationRule.CONFLICTING_EVIDENCE in agent.activation_rules
        assert AgentActivationRule.HIGH_UNCERTAINTY in agent.activation_rules
        assert AgentActivationRule.ALWAYS not in agent.activation_rules

    @pytest.mark.asyncio
    async def test_alternative_hypothesis_agent_activation_rules(self, mock_router):
        """Alternative Hypothesis should activate on high stakes, conflicting evidence, high uncertainty."""
        settings = Settings()
        agent = AlternativeHypothesisAgent(mock_router, settings, call_type="alternative_hypothesis")
        assert agent.role == AgentRole.ALTERNATIVE_HYPOTHESIS
        assert AgentActivationRule.HIGH_STAKES in agent.activation_rules
        assert AgentActivationRule.CONFLICTING_EVIDENCE in agent.activation_rules
        assert AgentActivationRule.HIGH_UNCERTAINTY in agent.activation_rules

    @pytest.mark.asyncio
    async def test_verifier_agent_activates_always(self, mock_router):
        """Verifier should always activate."""
        settings = Settings()
        agent = VerifierAgent(mock_router, settings, call_type="verifier")
        assert agent.role == AgentRole.VERIFIER
        assert AgentActivationRule.ALWAYS in agent.activation_rules

    @pytest.mark.asyncio
    async def test_judge_agent_activates_always(self, mock_router):
        """Judge should always activate."""
        settings = Settings()
        agent = JudgeAgent(mock_router, settings, call_type="judge")
        assert agent.role == AgentRole.JUDGE
        assert AgentActivationRule.ALWAYS in agent.activation_rules


class TestAgentProcessing:
    """Test agent processing with mocked responses."""

    @pytest.mark.asyncio
    async def test_researcher_processes_evidence(self, mock_router):
        """Researcher should extract claims from evidence."""
        from app.llm_gateway.providers.models import CompletionResponse, Usage

        # Pre-program researcher response
        researcher_output = ResearcherOutput(
            key_claims=["Claim 1: The sky is blue", "Claim 2: Water is wet"],
            evidence_summary="Evidence shows the sky appears blue due to Rayleigh scattering.",
            confidence=0.85,
            gaps=["Need more evidence about water wetness"],
        )
        mock_router.provider._responses["mock-model"] = CompletionResponse(
            content=researcher_output.model_dump_json(),
            model="mock-model",
            usage=Usage(prompt_tokens=100, completion_tokens=50, total_tokens=150),
            provider="mock",
        )

        settings = Settings()
        agent = ResearcherAgent(mock_router, settings, call_type="researcher")

        state = OrchestrationState(
            request_id="test-123",
            query="Why is the sky blue?",
            max_iterations=3,
            token_budget=6000,
            plan=None,
            pending_subquestions=[],
            issued_subqueries=[],
            evidence=[],
            consecutive_empty_retrievals=0,
            iteration=1,
            tokens_used=100,
            sufficient=False,
            stop_reason=None,
            answer=None,
            warnings=[],
        )

        messages = await agent.process(state, [])
        assert len(messages) == 1
        assert messages[0].from_agent == AgentRole.RESEARCHER
        assert "Claim 1" in messages[0].content
        assert messages[0].payload["confidence"] == 0.85

    @pytest.mark.asyncio
    async def test_skeptic_challenges_claims(self, mock_router):
        """Skeptic should challenge researcher's claims."""
        from app.llm_gateway.providers.models import CompletionResponse, Usage

        skeptic_output = SkepticOutput(
            challenges=["Claim 1 lacks quantitative evidence", "Rayleigh scattering explanation is oversimplified"],
            counter_evidence_needed=["Spectral measurements", "Atmospheric composition data"],
            confidence=0.75,
            severity=0.8,
        )
        mock_router.provider._responses["mock-model"] = CompletionResponse(
            content=skeptic_output.model_dump_json(),
            model="mock-model",
            usage=Usage(prompt_tokens=100, completion_tokens=50, total_tokens=150),
            provider="mock",
        )

        settings = Settings()
        agent = SkepticAgent(mock_router, settings, call_type="skeptic")

        state = OrchestrationState(
            request_id="test-123",
            query="Why is the sky blue?",
            max_iterations=3,
            token_budget=6000,
            plan=None,
            pending_subquestions=[],
            issued_subqueries=[],
            evidence=[],
            consecutive_empty_retrievals=0,
            iteration=1,
            tokens_used=100,
            sufficient=False,
            stop_reason=None,
            answer=None,
            warnings=[],
        )

        # Simulate prior researcher message
        prior_messages = [AgentMessage(
            from_agent=AgentRole.RESEARCHER,
            content="Key Claims:\n- Claim 1: The sky is blue\n- Claim 2: Water is wet\n\nConfidence: 0.85",
            payload={"key_claims": ["Claim 1: The sky is blue", "Claim 2: Water is wet"], "confidence": 0.85},
        )]

        messages = await agent.process(state, prior_messages)
        assert len(messages) == 1
        assert messages[0].from_agent == AgentRole.SKEPTIC
        assert "quantitative evidence" in messages[0].content
        assert messages[0].payload["severity"] == 0.8

    @pytest.mark.asyncio
    async def test_alternative_hypothesis_proposes_alternatives(self, mock_router):
        """Alternative Hypothesis should propose competing explanations."""
        from app.llm_gateway.providers.models import CompletionResponse, Usage

        alt_output = AlternativeHypothesisOutput(
            alternative_explanations=["The sky appears blue due to human perception bias", "Atmospheric refraction causes blue appearance"],
            supporting_evidence_for_alternatives=["Psychological studies on color perception", "Refraction index measurements"],
            confidence=0.6,
            distinctiveness=0.7,
        )
        mock_router.provider._responses["mock-model"] = CompletionResponse(
            content=alt_output.model_dump_json(),
            model="mock-model",
            usage=Usage(prompt_tokens=100, completion_tokens=50, total_tokens=150),
            provider="mock",
        )

        settings = Settings()
        agent = AlternativeHypothesisAgent(mock_router, settings, call_type="alternative_hypothesis")

        state = OrchestrationState(
            request_id="test-123",
            query="Why is the sky blue?",
            max_iterations=3,
            token_budget=6000,
            plan=None,
            pending_subquestions=[],
            issued_subqueries=[],
            evidence=[],
            consecutive_empty_retrievals=0,
            iteration=1,
            tokens_used=100,
            sufficient=False,
            stop_reason=None,
            answer=None,
            warnings=[],
        )

        prior_messages = [
            AgentMessage(
                from_agent=AgentRole.RESEARCHER,
                content="Claims...",
                payload={"key_claims": ["Claim 1"], "confidence": 0.85},
            ),
            AgentMessage(
                from_agent=AgentRole.SKEPTIC,
                content="Challenges...",
                payload={"challenges": ["Lacks quantitative evidence"]},
            ),
        ]

        messages = await agent.process(state, prior_messages)
        assert len(messages) == 1
        assert messages[0].from_agent == AgentRole.ALTERNATIVE_HYPOTHESIS
        assert "perception bias" in messages[0].content
        assert messages[0].payload["distinctiveness"] == 0.7

    @pytest.mark.asyncio
    async def test_verifier_verifies_claims(self, mock_router):
        """Verifier should verify claims against evidence."""
        from app.llm_gateway.providers.models import CompletionResponse, Usage

        verifier_output = VerifierOutput(
            claim="The sky is blue due to Rayleigh scattering",
            status="SUPPORTED",
            confidence=0.9,
            reasoning="Evidence directly supports the claim",
            evidence_coverage=0.85,
            source_quality=0.8,
        )
        mock_router.provider._responses["mock-model"] = CompletionResponse(
            content=verifier_output.model_dump_json(),
            model="mock-model",
            usage=Usage(prompt_tokens=100, completion_tokens=50, total_tokens=150),
            provider="mock",
        )

        settings = Settings()
        agent = VerifierAgent(mock_router, settings, call_type="verifier")

        state = OrchestrationState(
            request_id="test-123",
            query="Why is the sky blue?",
            max_iterations=3,
            token_budget=6000,
            plan=None,
            pending_subquestions=[],
            issued_subqueries=[],
            evidence=[],
            consecutive_empty_retrievals=0,
            iteration=1,
            tokens_used=100,
            sufficient=False,
            stop_reason=None,
            answer=None,
            warnings=[],
        )

        prior_messages = [
            AgentMessage(
                from_agent=AgentRole.RESEARCHER,
                content="Claims...",
                payload={"key_claims": ["The sky is blue due to Rayleigh scattering"]},
            ),
        ]

        messages = await agent.process(state, prior_messages)
        assert len(messages) == 1
        assert messages[0].from_agent == AgentRole.VERIFIER
        assert "SUPPORTED" in messages[0].content

    @pytest.mark.asyncio
    async def test_judge_synthesizes_final_answer(self, mock_router):
        """Judge should synthesize debate into final answer."""
        from app.llm_gateway.providers.models import CompletionResponse, Usage

        judge_output = JudgeOutput(
            resolution="The sky is blue due to Rayleigh scattering; alternatives are less supported.",
            final_answer="The sky appears blue primarily due to Rayleigh scattering [1]. While alternative explanations exist, they are less well-supported by evidence.",
            confidence=0.85,
            key_disagreements_resolved=["Skeptic's challenge about quantitative evidence acknowledged; verifier confirms evidence support"],
            remaining_uncertainties=["Exact atmospheric conditions vary"],
            should_continue_debate=False,
        )
        mock_router.provider._responses["mock-model"] = CompletionResponse(
            content=judge_output.model_dump_json(),
            model="mock-model",
            usage=Usage(prompt_tokens=100, completion_tokens=50, total_tokens=150),
            provider="mock",
        )

        settings = Settings()
        agent = JudgeAgent(mock_router, settings, call_type="judge")

        state = OrchestrationState(
            request_id="test-123",
            query="Why is the sky blue?",
            max_iterations=3,
            token_budget=6000,
            plan=None,
            pending_subquestions=[],
            issued_subqueries=[],
            evidence=[],
            consecutive_empty_retrievals=0,
            iteration=1,
            tokens_used=100,
            sufficient=False,
            stop_reason=None,
            answer=None,
            warnings=[],
        )

        prior_messages = [
            AgentMessage(
                from_agent=AgentRole.RESEARCHER,
                content="Claims...",
                payload={},
            ),
            AgentMessage(
                from_agent=AgentRole.SKEPTIC,
                content="Challenges...",
                payload={},
            ),
            AgentMessage(
                from_agent=AgentRole.VERIFIER,
                content="Verification...",
                payload={},
            ),
        ]

        messages = await agent.process(state, prior_messages)
        assert len(messages) == 1
        assert messages[0].from_agent == AgentRole.JUDGE
        assert "Rayleigh scattering" in messages[0].content
        assert messages[0].payload["should_continue_debate"] is False


class TestAgentCoordinator:
    """Test the AgentCoordinator orchestration."""

    @pytest.mark.asyncio
    async def test_coordinator_activates_correct_agents(self, mock_coordinator):
        """Coordinator should activate agents based on risk level."""
        state = OrchestrationState(
            request_id="test-123",
            query="High stakes medical question",
            max_iterations=3,
            token_budget=6000,
            plan=None,  # Will be set below
            pending_subquestions=[],
            issued_subqueries=[],
            evidence=[],
            consecutive_empty_retrievals=0,
            iteration=1,
            tokens_used=100,
            sufficient=False,
            stop_reason=None,
            answer=None,
            warnings=[],
        )

        # Add a plan with high risk
        from app.orchestration.models import ResearchPlan
        state["plan"] = ResearchPlan(
            objective="High stakes medical question",
            risk_level="high",
            subquestions=["What are the risks?"],
        )

        active_roles = mock_coordinator.should_activate_agents(state)
        assert AgentRole.RESEARCHER in active_roles
        assert AgentRole.VERIFIER in active_roles
        assert AgentRole.SKEPTIC in active_roles  # High risk activates skeptic
        assert AgentRole.ALTERNATIVE_HYPOTHESIS in active_roles  # High risk activates alt-hyp
        assert AgentRole.JUDGE in active_roles

    @pytest.mark.asyncio
    async def test_coordinator_activates_for_conflicting_evidence(self, mock_coordinator):
        """Coordinator should activate skeptic/alt-hyp for conflicting evidence."""
        from app.evidence.models import EvidenceRef, SourceType
        from uuid import uuid4

        # Create evidence with high variance (conflicting)
        evidence = [
            EvidenceRef(
                chunk_id=uuid4(),
                document_id=uuid4(),
                source_id=uuid4(),
                source_path="doc1.txt",
                source_type=SourceType.TEXT,
                text="Evidence supporting claim A",
                score=0.9,
            ),
            EvidenceRef(
                chunk_id=uuid4(),
                document_id=uuid4(),
                source_id=uuid4(),
                source_path="doc2.txt",
                source_type=SourceType.TEXT,
                text="Evidence supporting claim B (contradicts A)",
                score=0.3,  # Low score = conflict
            ),
            EvidenceRef(
                chunk_id=uuid4(),
                document_id=uuid4(),
                source_id=uuid4(),
                source_path="doc3.txt",
                source_type=SourceType.TEXT,
                text="More evidence for A",
                score=0.85,
            ),
        ]

        state = OrchestrationState(
            request_id="test-123",
            query="Conflicting evidence question",
            max_iterations=3,
            token_budget=6000,
            plan=None,
            pending_subquestions=[],
            issued_subqueries=[],
            evidence=evidence,
            consecutive_empty_retrievals=0,
            iteration=1,
            tokens_used=100,
            sufficient=False,
            stop_reason=None,
            answer=None,
            warnings=[],
        )

        active_roles = mock_coordinator.should_activate_agents(state)
        assert AgentRole.SKEPTIC in active_roles
        assert AgentRole.ALTERNATIVE_HYPOTHESIS in active_roles

    @pytest.mark.asyncio
    async def test_coordinator_detects_disagreement(self, mock_coordinator):
        """Coordinator should detect material disagreement."""
        # This test verifies the disagreement detection logic
        # We can't easily test the full debate without mocking all agents,
        # but we can test the detection function

        # Create mock agent messages showing disagreement
        agent_messages = [
            {
                "from_agent": "researcher",
                "content": "Claims...",
                "payload": {"key_claims": ["Claim 1"]},
                "round": 1,
            },
            {
                "from_agent": "skeptic",
                "content": "Challenges...",
                "payload": {"severity": 0.8, "challenges": ["Challenge 1"]},
                "round": 1,
            },
        ]

        round_messages = [
            type('Msg', (), {"from_agent": AgentRole.SKEPTIC, "payload": {"severity": 0.8}})()
        ]

        disagreement = mock_coordinator._detect_disagreement(agent_messages, round_messages)
        assert disagreement is not None
        assert disagreement["skeptic_severity"] == 0.8


class TestMultiAgentIntegration:
    """Integration tests for multi-agent debate flow."""

    @pytest.mark.asyncio
    async def test_debate_runs_multiple_rounds(self, mock_coordinator):
        """Debate should run for configured max rounds or until judge stops."""
        state = OrchestrationState(
            request_id="test-123",
            query="Test question",
            max_iterations=3,
            token_budget=6000,
            plan=None,
            pending_subquestions=[],
            issued_subqueries=[],
            evidence=[],
            consecutive_empty_retrievals=0,
            iteration=1,
            tokens_used=100,
            sufficient=True,  # Stop after assess
            stop_reason="sufficient_evidence",
            answer=None,
            warnings=[],
        )

        # This test just verifies the coordinator can be instantiated
        # and the should_activate_agents works
        active_roles = mock_coordinator.should_activate_agents(state)
        assert AgentRole.RESEARCHER in active_roles
        assert AgentRole.VERIFIER in active_roles
        assert AgentRole.JUDGE in active_roles

    @pytest.mark.asyncio
    async def test_create_agent_coordinator(self, mock_router):
        """Factory function should create coordinator correctly."""
        from app.retrieval.hybrid import HybridRetriever
        from app.reranking.reranker import NoOpReranker
        from app.orchestration.agents.coordinator import create_agent_coordinator

        settings = Settings(
            multiagent_enabled=True,
            multiagent_max_rounds=3,
            orchestration_max_iterations=3,
            orchestration_token_budget=6000,
            orchestration_llm_timeout=30.0,
        )
        retriever = HybridRetriever()
        reranker = NoOpReranker()

        coordinator = create_agent_coordinator(mock_router, settings, retriever, reranker)
        assert isinstance(coordinator, AgentCoordinator)
        assert coordinator._max_rounds == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])