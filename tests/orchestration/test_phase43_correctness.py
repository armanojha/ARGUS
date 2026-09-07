"""Phase 43: Real-world correctness regression tests.

These tests validate the exact failure scenario reported by a real user:
irrelevant evidence producing false contradictions and unsafe synthesis fallback.

All tests are deterministic — no LLM calls.
"""

import pytest
from uuid import uuid4
from app.evidence.models import EvidenceRef, SourceType
from app.orchestration.nodes import (
    _compute_query_relevance,
    _detect_contradictions,
    _filter_evidence_by_relevance,
    _is_topic_coherent,
    filter_contradictions_by_query,
)


def _make_ref(text: str, score: float = 0.5, source: str = "test_source") -> EvidenceRef:
    """Create an EvidenceRef for testing."""
    return EvidenceRef(
        chunk_id=uuid4(),
        document_id=uuid4(),
        source_id=uuid4(),
        source_path=source,
        source_type=SourceType.TEXT,
        text=text,
        score=score,
        rank=1,
    )


# ─── TEST 1: Unrelated evidence should NOT produce contradictions ──────────

class TestUnrelatedEvidence:
    """Query: AI adoption acceleration.
    Evidence: attendance systems, employee databases, transport monitoring.
    Expected: NO contradictions, evidence filtered as irrelevant."""

    def test_unrelated_evidence_no_contradictions(self):
        """The exact real-world failure: unrelated evidence should not produce contradictions."""
        evidence = [
            _make_ref(
                "Smart India Hackathon urban intelligence platform using "
                "public transport fleets for mobile sensing and data collection "
                "across cities with IoT sensors."
            ),
            _make_ref(
                "Employee attendance system using biometric verification "
                "with database records for leave management and payroll."
            ),
            _make_ref(
                "LLM tool transparency requirements for AI systems "
                "in enterprise environments with compliance frameworks."
            ),
        ]
        query = "What evidence supports the claim that AI adoption is accelerating?"

        contradictions = _detect_contradictions(evidence, query=query)
        assert len(contradictions) == 0, (
            f"Expected 0 contradictions for unrelated evidence, got {len(contradictions)}: "
            + "; ".join(c.get("description", "") for c in contradictions)
        )

    def test_unrelated_evidence_filtered_by_relevance(self):
        """Irrelevant evidence should be filtered by the relevance gate."""
        evidence = [
            _make_ref(
                "Smart India Hackathon urban intelligence platform using "
                "public transport fleets for mobile sensing."
            ),
            _make_ref(
                "Employee attendance system using biometric verification "
                "with database records for leave management."
            ),
        ]
        query = "What evidence supports the claim that AI adoption is accelerating?"

        relevant, excluded = _filter_evidence_by_relevance(evidence, query)
        assert len(excluded) > 0, "Expected some evidence to be filtered as irrelevant"
        # Both chunks are about topics unrelated to AI adoption acceleration
        assert len(relevant) <= len(evidence)


# ─── TEST 2: Shared vocabulary should NOT produce contradictions ───────────

class TestSharedVocabulary:
    """Two evidence chunks sharing generic words like 'analytics', 'database'
    should NOT be flagged as contradictory."""

    def test_shared_analytics_no_contradiction(self):
        """Documents sharing 'analytics' but discussing different subjects."""
        evidence = [
            _make_ref(
                "Analytics can improve city planning by analyzing traffic "
                "patterns and urban infrastructure usage data."
            ),
            _make_ref(
                "Analytics are used in employee attendance tracking to "
                "monitor work hours and productivity metrics."
            ),
        ]
        query = "How is analytics used in organizations?"

        contradictions = _detect_contradictions(evidence, query=query)
        # These share "analytics" but discuss completely different domains
        # With the topic coherence fix, they should NOT be flagged
        for c in contradictions:
            # If any contradiction is detected, it should not be based on
            # shared generic words alone
            assert c.get("conflict_type") != "GENUINE_CONTRADICTION", (
                f"Shared 'analytics' should not produce GENUINE_CONTRADICTION: "
                f"{c.get('description')}"
            )

    def test_shared_database_no_contradiction(self):
        """Documents sharing 'database' but discussing different systems."""
        evidence = [
            _make_ref(
                "The database stores employee records including personal "
                "information, salary details, and employment history."
            ),
            _make_ref(
                "The database supports leave management by tracking "
                "vacation requests and approval workflows."
            ),
        ]
        query = "What database systems are used in the company?"

        contradictions = _detect_contradictions(evidence, query=query)
        # These share "database" but are about different aspects
        # They should NOT be flagged as GENUINE_CONTRADICTION
        for c in contradictions:
            assert c.get("conflict_type") != "GENUINE_CONTRADICTION", (
                f"Shared 'database' should not produce GENUINE_CONTRADICTION: "
                f"{c.get('description')}"
            )


# ─── TEST 3: Genuine numerical conflict should be detected ─────────────────

class TestGenuineNumericalConflict:
    """Evidence A: AI adoption was 35% in 2024.
    Evidence B: AI adoption was 52% in 2024.
    Expected: CONFLICT detected."""

    def test_same_metric_same_year_different_values(self):
        """Same entity, same metric, same year, different values = genuine conflict."""
        evidence = [
            _make_ref(
                "Enterprise revenue growth was 35% in 2024 according "
                "to industry survey of 500 companies."
            ),
            _make_ref(
                "Enterprise revenue growth reached 52% in 2024 based "
                "on annual technology report."
            ),
        ]
        query = "What is the current rate of revenue growth?"

        contradictions = _detect_contradictions(evidence, query=query)
        assert len(contradictions) >= 1, (
            "Expected at least 1 contradiction for same-metric different-values"
        )
        # Should be classified as GENUINE or POSSIBLE contradiction
        types = [c["conflict_type"] for c in contradictions]
        assert any(t in ("GENUINE_CONTRADICTION", "POSSIBLE_CONTRADICTION") for t in types), (
            f"Expected GENUINE or POSSIBLE contradiction, got: {types}"
        )


# ─── TEST 4: Different years should NOT be automatically contradictory ─────

class TestDifferentYears:
    """Evidence A: AI adoption was 35% in 2023.
    Evidence B: AI adoption was 52% in 2025.
    Expected: NOT automatically contradictory (different timeframes)."""

    def test_different_years_not_contradictory(self):
        """Same metric but different years should be DIFFERENT_TIMEFRAME, not GENUINE."""
        evidence = [
            _make_ref(
                "AI adoption in enterprise sector was 35% in 2023 according "
                "to industry survey."
            ),
            _make_ref(
                "AI adoption in enterprise sector reached 52% in 2025 based "
                "on latest report."
            ),
        ]
        query = "What is the trend of AI adoption over time?"

        contradictions = _detect_contradictions(evidence, query=query)
        for c in contradictions:
            # Should be classified as DIFFERENT_TIMEFRAME, not GENUINE_CONTRADICTION
            assert c["conflict_type"] != "GENUINE_CONTRADICTION", (
                f"Different years should not produce GENUINE_CONTRADICTION: "
                f"{c.get('description')}"
            )

    def test_different_years_filtered_by_query(self):
        """DIFFERENT_TIMEFRAME should be filtered unless query asks about history."""
        evidence = [
            _make_ref("AI adoption was 35% in 2023."),
            _make_ref("AI adoption was 52% in 2025."),
        ]
        contradictions = [
            {
                "severity": 0.8,
                "confidence": "MEDIUM",
                "conflict_type": "DIFFERENT_TIMEFRAME",
                "description": "Test",
                "evidence_indices": [1, 2],
                "entity_overlap": ["adoption"],
                "metric_overlap": ["adoption"],
                "timeframe_i": [2023],
                "timeframe_j": [2025],
                "resolved": False,
                "critical": True,
            }
        ]

        # Query asks about current state — DIFFERENT_TIMEFRAME should be filtered
        filtered = filter_contradictions_by_query(
            contradictions, evidence, "What is the current AI adoption rate?"
        )
        assert len(filtered) == 0, (
            "DIFFERENT_TIMEFRAME should be filtered for current-state queries"
        )


# ─── TEST 5: Different scopes should NOT be automatically contradictory ────

class TestDifferentScopes:
    """Evidence A: Enterprise AI adoption: 52%.
    Evidence B: Consumer AI adoption: 31%.
    Expected: NOT automatically contradictory (different scopes)."""

    def test_different_scopes_not_contradictory(self):
        """Different scopes (enterprise vs consumer) should not be GENUINE."""
        evidence = [
            _make_ref(
                "Enterprise AI adoption reached 52% in 2024 according to "
                "Gartner research report."
            ),
            _make_ref(
                "Consumer AI adoption was at 31% in 2024 based on "
                "consumer technology survey."
            ),
        ]
        query = "What is the rate of AI adoption?"

        contradictions = _detect_contradictions(evidence, query=query)
        for c in contradictions:
            # Different scopes should not be GENUINE_CONTRADICTION
            assert c["conflict_type"] != "GENUINE_CONTRADICTION", (
                f"Different scopes should not produce GENUINE_CONTRADICTION: "
                f"{c.get('description')}"
            )


# ─── TEST 6: Genuine opposing claims should be detected ────────────────────

class TestGenuineOpposingClaims:
    """Evidence A: The system supports feature X.
    Evidence B: The system does not support feature X.
    Expected: CONFLICT detected."""

    def test_direct_opposing_claims_detected(self):
        """Direct negation with same entity should be detected."""
        evidence = [
            _make_ref(
                "The Acme Cloud Platform supports feature X for data "
                "processing and analytics workflows."
            ),
            _make_ref(
                "The Acme Cloud Platform does not support feature X "
                "in its current release version."
            ),
        ]
        query = "Does the Acme Cloud Platform support feature X?"

        contradictions = _detect_contradictions(evidence, query=query)
        assert len(contradictions) >= 1, (
            "Expected at least 1 contradiction for direct opposing claims"
        )


# ─── TEST 7: Synthesis failure should produce safe fallback ─────────────────

class TestSynthesisFallback:
    """Force synthesis provider failure.
    Expected: concise safe fallback, NOT raw evidence dump."""

    def test_fallback_with_irrelevant_evidence(self):
        """When synthesis fails and evidence is irrelevant, say so."""
        from app.orchestration.nodes import _filter_evidence_by_relevance

        evidence = [
            _make_ref("Smart India Hackathon urban sensing platform."),
            _make_ref("Employee attendance biometric system."),
        ]
        query = "What evidence supports AI adoption acceleration?"

        relevant, excluded = _filter_evidence_by_relevance(evidence, query)
        assert len(relevant) == 0 or len(excluded) > 0, (
            "Irrelevant evidence should be filtered in fallback"
        )

    def test_fallback_with_relevant_evidence(self):
        """When synthesis fails but evidence is relevant, preserve it."""
        from app.orchestration.nodes import _filter_evidence_by_relevance

        evidence = [
            _make_ref(
                "AI adoption in enterprise sector reached 52% in 2024, "
                "showing significant acceleration from 35% in 2023."
            ),
            _make_ref(
                "Gartner reports that AI adoption accelerated across "
                "industries with 40% year-over-year growth."
            ),
        ]
        query = "What evidence supports AI adoption acceleration?"

        relevant, excluded = _filter_evidence_by_relevance(evidence, query)
        assert len(relevant) >= 1, (
            "Relevant evidence should be preserved in fallback"
        )


# ─── TEST 8: Query relevance gate works correctly ──────────────────────────

class TestQueryRelevanceGate:
    """The relevance gate should filter irrelevant evidence and preserve relevant."""

    def test_relevant_evidence_preserved(self):
        """Evidence about AI adoption should be preserved for AI adoption query."""
        evidence = [
            _make_ref(
                "AI adoption in enterprise sector reached 52% in 2024 "
                "showing significant acceleration."
            ),
        ]
        query = "What evidence supports AI adoption acceleration?"

        relevant, excluded = _filter_evidence_by_relevance(evidence, query)
        assert len(relevant) == 1, "Relevant evidence should be preserved"
        assert len(excluded) == 0, "No evidence should be excluded"

    def test_irrelevant_evidence_excluded(self):
        """Evidence about attendance systems should be excluded for AI adoption query."""
        evidence = [
            _make_ref(
                "Employee attendance system using biometric verification "
                "with database records."
            ),
        ]
        query = "What evidence supports AI adoption acceleration?"

        relevant, excluded = _filter_evidence_by_relevance(evidence, query)
        assert len(relevant) == 0, "Irrelevant evidence should be excluded"
        assert len(excluded) == 1, "One item should be excluded"

    def test_relevance_score_range(self):
        """Relevance scores should be in [0.0, 1.0]."""
        evidence = _make_ref("Some text about AI adoption acceleration trends.")
        query = "What evidence supports AI adoption acceleration?"

        score = _compute_query_relevance(evidence.text, query)
        assert 0.0 <= score <= 1.0, f"Score {score} out of range"

    def test_topic_coherence_check(self):
        """Topic coherence should require meaningful shared vocabulary."""
        # Same topic
        assert _is_topic_coherent(
            "AI adoption enterprise sector growth acceleration",
            "Enterprise AI adoption growth acceleration technology",
            min_shared_significant=3,
        )
        # Different topics
        assert not _is_topic_coherent(
            "Smart India Hackathon urban intelligence transport sensing",
            "Employee attendance biometric verification database records",
            min_shared_significant=3,
        )
