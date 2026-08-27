"""Contradiction Detection Engine (Phase 04).

Detects contradictions between claims in the Evidence Graph.
Implements the contradiction types from V2 §9.2.
"""

from __future__ import annotations

from app.graph.models import Claim
from app.logging_config import get_logger
from app.verification.models import ContradictionDetail, ContradictionType

logger = get_logger("argus.verification.contradiction")


class ContradictionDetector:
    """Detects contradictions between claims in the graph."""

    def __init__(self):
        pass

    def detect_contradictions(
        self,
        claims: list[Claim],
        min_severity: float = 0.3,
    ) -> list[ContradictionDetail]:
        """Detect all contradictions among a set of claims.

        Checks for all contradiction types from V2 §9.2:
        - publication_date
        - metric_definition
        - geographic_scope
        - time_period
        - revised_numbers
        - entity_mismatch
        - source_conflict
        - temporal_conflict
        """
        contradictions = []

        for i, claim_a in enumerate(claims):
            for claim_b in claims[i + 1:]:
                # Check if claims are about the same subject/predicate
                if not self._claims_overlap(claim_a, claim_b):
                    continue

                # Check each contradiction type
                for contra_type in ContradictionType:
                    detail = self._check_contradiction_type(claim_a, claim_b, contra_type)
                    if detail and detail.severity >= min_severity:
                        contradictions.append(detail)

        logger.info("contradiction_detection_completed", claims=len(claims), contradictions=len(contradictions))
        return contradictions

    def _claims_overlap(self, claim_a: Claim, claim_b: Claim) -> bool:
        """Check if two claims are about the same subject/predicate."""
        # Same subject entity
        if (
            claim_a.subject_entity_id
            and claim_b.subject_entity_id
            and claim_a.subject_entity_id == claim_b.subject_entity_id
        ):
            return True

        # Same object entity
        if (
            claim_a.object_entity_id
            and claim_b.object_entity_id
            and claim_a.object_entity_id == claim_b.object_entity_id
        ):
            return True

        # Similar predicates (simple string similarity)
        if self._predicates_similar(claim_a.predicate, claim_b.predicate):
            # Check if they share entities or values
            if claim_a.subject_entity_id and claim_a.subject_entity_id == claim_b.subject_entity_id:
                return True
            if claim_a.object_entity_id and claim_a.object_entity_id == claim_b.object_entity_id:
                return True
            if (
                claim_a.object_value
                and claim_b.object_value
                and claim_a.object_value.lower() == claim_b.object_value.lower()
            ):
                return True

        return False

    def _predicates_similar(self, pred_a: str, pred_b: str) -> bool:
        """Simple predicate similarity check."""
        pred_a_lower = pred_a.lower().strip()
        pred_b_lower = pred_b.lower().strip()

        if pred_a_lower == pred_b_lower:
            return True

        # Check for common synonyms/related predicates
        synonym_groups = [
            {"is", "was", "equals", "="},
            {"works at", "employed by", "works for", "employee of"},
            {"located in", "based in", "headquartered in", "in"},
            {"occurred in", "happened in", "took place in"},
            {"said", "stated", "claimed", "reported"},
            {"increased", "rose", "went up", "grew"},
            {"decreased", "fell", "dropped", "declined"},
        ]

        for group in synonym_groups:
            if pred_a_lower in group and pred_b_lower in group:
                return True

        return False

    def _check_contradiction_type(
        self,
        claim_a: Claim,
        claim_b: Claim,
        contra_type: ContradictionType,
    ) -> ContradictionDetail | None:
        """Check for a specific contradiction type between two claims."""
        if contra_type == ContradictionType.PUBLICATION_DATE:
            return self._check_publication_date(claim_a, claim_b)
        elif contra_type == ContradictionType.METRIC_DEFINITION:
            return self._check_metric_definition(claim_a, claim_b)
        elif contra_type == ContradictionType.GEOGRAPHIC_SCOPE:
            return self._check_geographic_scope(claim_a, claim_b)
        elif contra_type == ContradictionType.TIME_PERIOD:
            return self._check_time_period(claim_a, claim_b)
        elif contra_type == ContradictionType.REVISED_NUMBERS:
            return self._check_revised_numbers(claim_a, claim_b)
        elif contra_type == ContradictionType.ENTITY_MISMATCH:
            return self._check_entity_mismatch(claim_a, claim_b)
        elif contra_type == ContradictionType.SOURCE_CONFLICT:
            return self._check_source_conflict(claim_a, claim_b)
        elif contra_type == ContradictionType.TEMPORAL_CONFLICT:
            return self._check_temporal_conflict(claim_a, claim_b)
        return None

    def _check_publication_date(self, claim_a: Claim, claim_b: Claim) -> ContradictionDetail | None:
        """Check if claims have different publication dates for same info."""
        # If same claim text but different publication dates
        if (
            claim_a.published_at
            and claim_b.published_at
            and claim_a.text.strip().lower() == claim_b.text.strip().lower()
            and claim_a.published_at != claim_b.published_at
        ):
            return ContradictionDetail(
                contradiction_type=ContradictionType.PUBLICATION_DATE,
                description=f"Same claim published at different times: {claim_a.published_at} vs {claim_b.published_at}",
                claim_a_id=claim_a.id,
                claim_b_id=claim_b.id,
                evidence_a_ids=claim_a.supporting_chunk_ids,
                evidence_b_ids=claim_b.supporting_chunk_ids,
                severity=0.7,
                resolution_suggestion="Use the most recent publication unless historical context is needed",
            )
        return None

    def _check_metric_definition(self, claim_a: Claim, claim_b: Claim) -> ContradictionDetail | None:
        """Check if claims use different definitions for the same metric."""
        # Look for numeric values with different units or definitions
        import re

        numbers_a = re.findall(r'\d+(?:\.\d+)?\s*[%$€£¥]?\s*(?:percent|percentage|ratio|rate|index|score)?', claim_a.text, re.IGNORECASE)
        numbers_b = re.findall(r'\d+(?:\.\d+)?\s*[%$€£¥]?\s*(?:percent|percentage|ratio|rate|index|score)?', claim_b.text, re.IGNORECASE)

        if numbers_a and numbers_b and claim_a.subject_entity_id == claim_b.subject_entity_id:
            # If same subject but different numeric values
            # Extract just the numbers
            def _extract_nums(text: str) -> list[float]:
                matches = re.findall(r'\d+(?:\.\d+)?', text)
                return [float(m) for m in matches]

            vals_a = _extract_nums(claim_a.text)
            vals_b = _extract_nums(claim_b.text)

            if vals_a and vals_b and vals_a[0] != vals_b[0]:
                return ContradictionDetail(
                    contradiction_type=ContradictionType.METRIC_DEFINITION,
                    description=f"Different metric values for same subject: {vals_a[0]} vs {vals_b[0]}",
                    claim_a_id=claim_a.id,
                    claim_b_id=claim_b.id,
                    evidence_a_ids=claim_a.supporting_chunk_ids,
                    evidence_b_ids=claim_b.supporting_chunk_ids,
                    severity=0.8,
                    resolution_suggestion="Check if metrics use different definitions, time periods, or methodologies",
                )
        return None

    def _check_geographic_scope(self, claim_a: Claim, claim_b: Claim) -> ContradictionDetail | None:
        """Check if claims have different geographic scopes."""
        # Look for location entities or geographic terms
        geo_terms = ["country", "state", "city", "region", "province", "district", "jurisdiction", "national", "federal", "local", "global", "international"]

        text_a = claim_a.text.lower()
        text_b = claim_b.text.lower()

        geo_a = [term for term in geo_terms if term in text_a]
        geo_b = [term for term in geo_terms if term in text_b]

        if (
            geo_a
            and geo_b
            and geo_a != geo_b
            and claim_a.subject_entity_id == claim_b.subject_entity_id
        ):
            return ContradictionDetail(
                contradiction_type=ContradictionType.GEOGRAPHIC_SCOPE,
                description=f"Different geographic scopes: {geo_a} vs {geo_b}",
                claim_a_id=claim_a.id,
                claim_b_id=claim_b.id,
                evidence_a_ids=claim_a.supporting_chunk_ids,
                evidence_b_ids=claim_b.supporting_chunk_ids,
                severity=0.6,
                resolution_suggestion="Specify the geographic scope explicitly; results may not be comparable across jurisdictions",
            )
        return None

    def _check_time_period(self, claim_a: Claim, claim_b: Claim) -> ContradictionDetail | None:
        """Check if claims refer to different time periods."""
        # Check validity time ranges
        if (
            claim_a.valid_from
            and claim_b.valid_from
            # Same subject but different validity periods
            and claim_a.subject_entity_id == claim_b.subject_entity_id
            and claim_a.valid_to
            and claim_b.valid_to
            # Simple check: if one ends before the other starts
            and (claim_a.valid_to < claim_b.valid_from or claim_b.valid_to < claim_a.valid_from)
        ):
            return ContradictionDetail(
                contradiction_type=ContradictionType.TIME_PERIOD,
                description=f"Non-overlapping validity periods: {claim_a.valid_from}-{claim_a.valid_to} vs {claim_b.valid_from}-{claim_b.valid_to}",
                claim_a_id=claim_a.id,
                claim_b_id=claim_b.id,
                evidence_a_ids=claim_a.supporting_chunk_ids,
                evidence_b_ids=claim_b.supporting_chunk_ids,
                severity=0.7,
                resolution_suggestion="Claims refer to different time periods; specify which period is relevant",
            )
        return None

    def _check_revised_numbers(self, claim_a: Claim, claim_b: Claim) -> ContradictionDetail | None:
        """Check if one claim revises/restates numbers from another."""
        import re

        # Extract numbers with context
        numbers_a = re.findall(r'(\d+(?:\.\d+)?)\s*(?:percent|%|dollars?|\$|million|billion|trillion)', claim_a.text, re.IGNORECASE)
        numbers_b = re.findall(r'(\d+(?:\.\d+)?)\s*(?:percent|%|dollars?|\$|million|billion|trillion)', claim_b.text, re.IGNORECASE)

        if numbers_a and numbers_b:
            vals_a = [float(n) for n in numbers_a]
            vals_b = [float(n) for n in numbers_b]

            # If same subject and predicate but different numbers
            if (
                claim_a.subject_entity_id == claim_b.subject_entity_id
                and self._predicates_similar(claim_a.predicate, claim_b.predicate)
                and vals_a != vals_b
                and claim_a.published_at
                and claim_b.published_at
            ):
                # Check if one is more recent (revised)
                if claim_a.published_at > claim_b.published_at:
                    return ContradictionDetail(
                        contradiction_type=ContradictionType.REVISED_NUMBERS,
                        description=f"Later claim revises numbers: {vals_b} -> {vals_a}",
                        claim_a_id=claim_a.id,
                        claim_b_id=claim_b.id,
                        evidence_a_ids=claim_a.supporting_chunk_ids,
                        evidence_b_ids=claim_b.supporting_chunk_ids,
                        severity=0.8,
                        resolution_suggestion=f"Use revised value {vals_a} from {claim_a.published_at.date() if claim_a.published_at else 'later source'}",
                    )
                elif claim_b.published_at > claim_a.published_at:
                    return ContradictionDetail(
                        contradiction_type=ContradictionType.REVISED_NUMBERS,
                        description=f"Later claim revises numbers: {vals_a} -> {vals_b}",
                        claim_a_id=claim_a.id,
                        claim_b_id=claim_b.id,
                        evidence_a_ids=claim_a.supporting_chunk_ids,
                        evidence_b_ids=claim_b.supporting_chunk_ids,
                        severity=0.8,
                        resolution_suggestion=f"Use revised value {vals_b} from {claim_b.published_at.date() if claim_b.published_at else 'later source'}",
                    )
        return None

    def _check_entity_mismatch(self, claim_a: Claim, claim_b: Claim) -> ContradictionDetail | None:
        """Check if claims refer to different entities with similar names."""
        # This would require entity resolution - simplified check
        if (
            claim_a.subject_entity_id
            and claim_b.subject_entity_id
            and claim_a.subject_entity_id != claim_b.subject_entity_id
            # Check if predicates are similar (talking about same attribute)
            and self._predicates_similar(claim_a.predicate, claim_b.predicate)
        ):
            return ContradictionDetail(
                contradiction_type=ContradictionType.ENTITY_MISMATCH,
                description=f"Similar claims about different entities: {claim_a.subject_entity_id} vs {claim_b.subject_entity_id}",
                claim_a_id=claim_a.id,
                claim_b_id=claim_b.id,
                evidence_a_ids=claim_a.supporting_chunk_ids,
                evidence_b_ids=claim_b.supporting_chunk_ids,
                severity=0.5,
                resolution_suggestion="Verify entity identity; may be different entities with similar attributes",
            )
        return None

    def _check_source_conflict(self, claim_a: Claim, claim_b: Claim) -> ContradictionDetail | None:
        """Check for direct source disagreement on same claim."""
        # Same subject, same predicate, different object values
        if (claim_a.subject_entity_id == claim_b.subject_entity_id and
            self._predicates_similar(claim_a.predicate, claim_b.predicate)):

            # Different object values
            obj_a = claim_a.object_value or (str(claim_a.object_entity_id) if claim_a.object_entity_id else "")
            obj_b = claim_b.object_value or (str(claim_b.object_entity_id) if claim_b.object_entity_id else "")

            if obj_a and obj_b and obj_a.lower() != obj_b.lower():
                # Check if they have different supporting sources
                sources_a = set()
                sources_b = set()

                for chunk_id in claim_a.supporting_chunk_ids:
                    # Would need to get source from chunk - simplified
                    sources_a.add(str(chunk_id)[:8])

                for chunk_id in claim_b.supporting_chunk_ids:
                    sources_b.add(str(chunk_id)[:8])

                if sources_a != sources_b:
                    return ContradictionDetail(
                        contradiction_type=ContradictionType.SOURCE_CONFLICT,
                        description=f"Sources disagree: '{obj_a}' vs '{obj_b}' for {claim_a.predicate}",
                        claim_a_id=claim_a.id,
                        claim_b_id=claim_b.id,
                        evidence_a_ids=claim_a.supporting_chunk_ids,
                        evidence_b_ids=claim_b.supporting_chunk_ids,
                        severity=0.9,
                        resolution_suggestion="Evaluate source credibility; check for more recent or authoritative sources",
                    )
        return None

    def _check_temporal_conflict(self, claim_a: Claim, claim_b: Claim) -> ContradictionDetail | None:
        """Check for temporal validity conflicts."""
        # Both claims have validity periods that conflict
        if (
            claim_a.valid_from
            and claim_b.valid_from
            and claim_a.subject_entity_id == claim_b.subject_entity_id
            # Check if validity periods overlap but claims contradict
            and self._predicates_similar(claim_a.predicate, claim_b.predicate)
        ):
            obj_a = claim_a.object_value or ""
            obj_b = claim_b.object_value or ""
            if obj_a and obj_b and obj_a.lower() != obj_b.lower():
                # Check temporal overlap
                a_start = claim_a.valid_from
                a_end = claim_a.valid_to or a_start
                b_start = claim_b.valid_from
                b_end = claim_b.valid_to or b_start

                # Overlap check
                if a_start <= b_end and b_start <= a_end:
                    return ContradictionDetail(
                        contradiction_type=ContradictionType.TEMPORAL_CONFLICT,
                        description=f"Conflicting claims during overlapping period: {obj_a} vs {obj_b}",
                        claim_a_id=claim_a.id,
                        claim_b_id=claim_b.id,
                        evidence_a_ids=claim_a.supporting_chunk_ids,
                        evidence_b_ids=claim_b.supporting_chunk_ids,
                        severity=0.8,
                        resolution_suggestion="Check if one claim supersedes the other or if they refer to different aspects",
                    )
        return None


def get_contradiction_detector() -> ContradictionDetector:
    """Get or create the singleton contradiction detector."""
    return ContradictionDetector()