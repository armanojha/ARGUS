"""Agent Coordinator for Phase 10 Multi-Agent Challenge.

Coordinates the multi-agent debate: activates appropriate agents based on
risk/uncertainty level, runs debate rounds, detects disagreement, and
triggers targeted retrieval when agents disagree materially.
"""

from __future__ import annotations

from typing import Any

from app.config import Settings
from app.llm_gateway.routing.multi_model_router import MultiModelRouter
from app.llm_gateway.routing.router import LLMRouter
from app.logging_config import get_logger
from app.orchestration.agents.agents import create_agents
from app.orchestration.contracts import (
    AgentCoordinatorInterface,
    AgentInterface,
    AgentMessage,
    AgentRole,
    OrchestrationState,
)
from app.reranking.reranker import NoOpReranker, Reranker
from app.retrieval.hybrid import HybridRetriever

logger = get_logger("argus.orchestration.agents.coordinator")


class AgentCoordinator(AgentCoordinatorInterface):
    """Coordinates multi-agent debate for high-risk questions."""

    def __init__(
        self,
        router: LLMRouter | MultiModelRouter,
        settings: Settings,
        retriever: HybridRetriever,
        reranker: Reranker | NoOpReranker,
    ) -> None:
        self._router = router
        self._settings = settings
        self._retriever = retriever
        self._reranker = reranker
        self._agents = create_agents(router, settings)
        self._max_rounds = settings.multiagent_max_rounds
        self._skeptic_threshold = settings.multiagent_skeptic_threshold
        self._disagreement_threshold = settings.multiagent_disagreement_threshold

    def should_activate_agents(self, state: OrchestrationState) -> list[AgentRole]:
        """Determine which agents should activate for this state."""
        if not self._settings.multiagent_enabled:
            return []

        plan = state.get("plan")
        evidence = state.get("evidence", [])

        # Always activate Researcher and Verifier
        active_roles = [AgentRole.RESEARCHER, AgentRole.VERIFIER]

        # Check if we should activate Skeptic and Alternative Hypothesis
        should_activate_skeptic = False
        should_activate_alt_hyp = False

        # Rule 1: High stakes / high uncertainty (from plan risk_level)
        if plan and plan.risk_level in ("medium", "high"):
            should_activate_skeptic = True
            should_activate_alt_hyp = True
            logger.info("multi_agent_activate_high_risk", risk_level=plan.risk_level)

        # Rule 2: Conflicting evidence (check evidence scores for divergence)
        if evidence and self._has_conflicting_evidence(evidence):
            should_activate_skeptic = True
            should_activate_alt_hyp = True
            logger.info("multi_agent_activate_conflicting_evidence")

        # Rule 3: High uncertainty (low evidence scores or few sources)
        if evidence and self._has_high_uncertainty(evidence):
            should_activate_skeptic = True
            should_activate_alt_hyp = True
            logger.info("multi_agent_activate_high_uncertainty")

        # Rule 4: Researcher/evidence confidence below the skeptic threshold.
        # Low average evidence score signals the researcher cannot ground its
        # answer confidently -> surface skepticism and an alternative hypothesis.
        if evidence and self._below_skeptic_threshold(evidence):
            should_activate_skeptic = True
            should_activate_alt_hyp = True
            logger.info("multi_agent_activate_low_confidence")

        if should_activate_skeptic:
            active_roles.append(AgentRole.SKEPTIC)
        if should_activate_alt_hyp:
            active_roles.append(AgentRole.ALTERNATIVE_HYPOTHESIS)

        # Judge always activates last
        active_roles.append(AgentRole.JUDGE)

        return active_roles

    def _has_conflicting_evidence(self, evidence: list[Any]) -> bool:
        """Check if evidence shows conflicting signals (score divergence)."""
        if len(evidence) < 3:
            return False
        scores = [e.score for e in evidence]
        # High variance in top evidence scores suggests conflict
        import statistics
        if len(scores) >= 3:
            stdev = statistics.stdev(scores)
            mean_score = statistics.mean(scores)
            # Coefficient of variation > threshold
            cv = stdev / mean_score if mean_score > 0 else 0
            return cv > self._disagreement_threshold
        return False

    def _has_high_uncertainty(self, evidence: list[Any]) -> bool:
        """Check if evidence base has high uncertainty."""
        # Low average score = high uncertainty
        avg_score = sum(e.score for e in evidence) / len(evidence)
        return avg_score < 0.5

    def _below_skeptic_threshold(self, evidence: list[Any]) -> bool:
        """True when the average evidence score is below the skeptic threshold.

        Uses ``multiagent_skeptic_threshold`` (V2 §10.1: "Researcher confidence
        below threshold -> activate Skeptic"). Deliberately strict (``<``) so
        reserves the extra agent budget for genuinely weak grounding.
        """
        avg_score = sum(e.score for e in evidence) / len(evidence)
        return avg_score < self._skeptic_threshold

    async def run_debate(
        self,
        state: OrchestrationState,
        agents: list[AgentInterface] | None = None,
        max_rounds: int | None = None,
    ) -> OrchestrationState:
        """Run a multi-agent debate and return updated state."""
        if agents is None:
            active_roles = self.should_activate_agents(state)
            agents = [self._agents[role] for role in active_roles if role in self._agents]
        else:
            active_roles = [a.role for a in agents]

        max_rounds = max_rounds or self._max_rounds

        logger.info(
            "multi_agent_debate_started",
            request_id=state.get("request_id"),
            agents=[r.value for r in active_roles],
            max_rounds=max_rounds,
        )

        # Initialize debate state in the orchestration state
        agent_messages: list[dict[str, Any]] = []
        round_num = 0

        # Convert agent_messages to list of AgentMessage for process() calls
        def _get_agent_messages() -> list[AgentMessage]:
            return [
                AgentMessage(
                    from_agent=AgentRole(msg["from_agent"]),
                    content=msg["content"],
                    to_agent=AgentRole(msg["to_agent"]) if msg["to_agent"] else None,
                    payload=msg["payload"],
                    timestamp=msg["timestamp"],
                )
                for msg in agent_messages
            ]

        # Store initial debate state
        state_dict = dict(state)
        state_dict["agent_messages"] = agent_messages
        state_dict["agent_round"] = 0
        state_dict["debate_active"] = True
        state_dict["disagreement_detected"] = False

        # Sort agents by role order: Researcher -> Skeptic -> AltHyp -> Verifier -> Judge
        role_order = {
            AgentRole.RESEARCHER: 0,
            AgentRole.SKEPTIC: 1,
            AgentRole.ALTERNATIVE_HYPOTHESIS: 2,
            AgentRole.VERIFIER: 3,
            AgentRole.JUDGE: 4,
        }
        agents.sort(key=lambda a: role_order.get(a.role, 99))

        for round_num in range(1, max_rounds + 1):
            state["agent_round"] = round_num
            round_messages: list[AgentMessage] = []

            logger.info("multi_agent_round_started", round=round_num, request_id=state_dict.get("request_id"))

            for agent in agents:
                # Skip agents not in active roles for this round
                if agent.role not in active_roles and agent.role != AgentRole.JUDGE:
                    continue

                try:
                    agent_output = await agent.process(state_dict, round_messages + _get_agent_messages())  # type: ignore[arg-type]
                    round_messages.extend(agent_output)

                    # Log agent output
                    for msg in agent_output:
                        logger.info(
                            "multi_agent_message",
                            from_agent=msg.from_agent.value,
                            round=round_num,
                            request_id=state_dict.get("request_id"),
                        )
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "multi_agent_agent_failed",
                        agent=agent.role.value,
                        round=round_num,
                        error=str(exc),
                    )
                    # Add error message to keep debate moving
                    round_messages.append(AgentMessage(
                        from_agent=agent.role,
                        content=f"Agent {agent.role.value} encountered an error: {exc}",
                        payload={"error": str(exc)},
                    ))

            # Add round messages to global message list
            for msg in round_messages:
                agent_messages.append({
                    "from_agent": msg.from_agent.value,
                    "content": msg.content,
                    "to_agent": msg.to_agent.value if msg.to_agent else None,
                    "payload": msg.payload,
                    "timestamp": msg.timestamp.isoformat(),
                    "round": round_num,
                })

            # Check for disagreement after each round (before Judge)
            if round_num < max_rounds:
                disagreement = self._detect_disagreement(agent_messages, round_messages)
                if disagreement:
                    state_dict["disagreement_detected"] = True
                    logger.info("multi_agent_disagreement_detected", round=round_num, details=disagreement)

                    # Trigger targeted retrieval based on disagreement source
                    try:
                        queries = self._identify_disagreement_source(state_dict, disagreement)  # type: ignore[arg-type]
                        if queries:
                            logger.info("multi_agent_targeted_retrieval", round=round_num, queries=queries)
                            new_evidence = []
                            for query in queries:
                                results = self._retriever.search(query, top_k=3)
                                if results:
                                    results = self._reranker.rerank(query, results, top_k=3)
                                    new_evidence.extend(results)
                            if new_evidence:
                                # Merge new evidence, deduplicating by chunk_id. Keep
                                # the highest score per chunk so a lower-scoring
                                # re-retrieved copy can never reduce evidence quality
                                # (consistent with the Phase 02 evidence merge).
                                state_dict["evidence"] = self._merge_evidence(
                                    state_dict["evidence"],
                                    new_evidence,
                                )
                                logger.info("multi_agent_evidence_injected", count=len(new_evidence))
                    except Exception as exc:  # noqa: BLE001 - targeted retrieval is non-critical
                        logger.warning("multi_agent_targeted_retrieval_failed", error=str(exc))

            # Check if Judge says to continue
            judge_msg = next((m for m in round_messages if m.from_agent == AgentRole.JUDGE), None)
            if judge_msg and not judge_msg.payload.get("should_continue_debate", False):
                logger.info("multi_agent_judge_stopped", round=round_num)
                break

        state_dict["agent_messages"] = agent_messages
        state_dict["agent_round"] = round_num
        state_dict["debate_active"] = False

        logger.info(
            "multi_agent_debate_finished",
            request_id=state_dict.get("request_id"),
            rounds=round_num,
            total_messages=len(agent_messages),
        )

        return state_dict  # type: ignore[return-value]

    @staticmethod
    def _merge_evidence(existing: list[Any], incoming: list[Any]) -> list[Any]:
        """Deduplicate evidence by chunk_id, keeping the highest score per chunk.

        Mirrors the Phase 02 evidence merge: re-retrieved evidence can add or
        replace a chunk but may never *lower* its score. Results are returned
        sorted by score descending so the top of the list stays the strongest.
        """
        by_chunk: dict[Any, Any] = {}
        for item in list(existing) + list(incoming):
            current = by_chunk.get(item.chunk_id)
            if current is None or item.score > current.score:
                by_chunk[item.chunk_id] = item
        return sorted(by_chunk.values(), key=lambda e: e.score, reverse=True)

    def _detect_disagreement(
        self,
        all_messages: list[dict[str, Any]],
        round_messages: list[AgentMessage],
    ) -> dict[str, Any] | None:
        """Detect material disagreement between agents in this round."""
        skeptic_msgs = [m for m in round_messages if m.from_agent == AgentRole.SKEPTIC]
        alt_hyp_msgs = [m for m in round_messages if m.from_agent == AgentRole.ALTERNATIVE_HYPOTHESIS]
        verifier_msgs = [m for m in round_messages if m.from_agent == AgentRole.VERIFIER]

        if not skeptic_msgs and not alt_hyp_msgs:
            return None

        # Check severity from Skeptic
        skeptic_severity = 0.0
        for msg in skeptic_msgs:
            skeptic_severity = max(skeptic_severity, msg.payload.get("severity", 0.0))

        # Check distinctiveness from Alternative Hypothesis
        alt_distinctiveness = 0.0
        for msg in alt_hyp_msgs:
            alt_distinctiveness = max(alt_distinctiveness, msg.payload.get("distinctiveness", 0.0))

        # Check verifier statuses
        verifier_unsupported = 0
        verifier_contradicted = 0
        for msg in verifier_msgs:
            for v in msg.payload.get("verifications", []):
                if v.get("status") == "UNSUPPORTED":
                    verifier_unsupported += 1
                elif v.get("status") == "CONTRADICTED":
                    verifier_contradicted += 1

        # Disagreement if skeptic severity high, alt distinctiveness high, or verifier finds issues
        if (skeptic_severity >= self._disagreement_threshold or
            alt_distinctiveness >= self._disagreement_threshold or
            verifier_contradicted > 0 or
            verifier_unsupported >= 2):
            return {
                "skeptic_severity": skeptic_severity,
                "alt_distinctiveness": alt_distinctiveness,
                "verifier_unsupported": verifier_unsupported,
                "verifier_contradicted": verifier_contradicted,
                "round": len(all_messages),  # Approximate
            }

        return None

    def _identify_disagreement_source(
        self,
        state: OrchestrationState,
        disagreement: dict[str, Any],
    ) -> list[str]:
        """Identify which claims/sources cause the disagreement for targeted retrieval."""
        # Extract claims from researcher
        researcher_claims = []
        for msg_dict in state.get("agent_messages", []):
            if msg_dict.get("from_agent") == AgentRole.RESEARCHER.value:
                researcher_claims = msg_dict.get("payload", {}).get("key_claims", [])
                break

        # Identify which claims are challenged
        challenged_claims = []
        for msg_dict in state.get("agent_messages", []):
            if msg_dict.get("from_agent") == AgentRole.SKEPTIC.value:
                challenged_claims = msg_dict.get("payload", {}).get("challenges", [])
                break

        # Build targeted retrieval queries
        queries = []
        for claim in researcher_claims:
            for challenge in challenged_claims:
                # Simple heuristic: if challenge relates to claim, create query
                if any(word in challenge.lower() for word in claim.lower().split() if len(word) > 4):
                    queries.append(f"Evidence for or against: {claim}")

        return queries[:3]  # Limit to top 3


def create_agent_coordinator(
    router: LLMRouter | MultiModelRouter,
    settings: Settings,
    retriever: HybridRetriever,
    reranker: Reranker | NoOpReranker,
) -> AgentCoordinator:
    """Factory for creating the agent coordinator."""
    return AgentCoordinator(router, settings, retriever, reranker)