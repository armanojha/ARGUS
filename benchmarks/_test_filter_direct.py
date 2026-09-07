#!/usr/bin/env python3
"""Test filtering logic directly on C1 evidence."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.orchestration.nodes import _detect_contradictions, filter_contradictions_by_query

# Simulated evidence for C1 (revenue)
class FakeRef:
    def __init__(self, text, score=0.5):
        self.text = text
        self.score = score
        self.chunk_id = "c1"
        self.document_id = "d1"
        self.source_id = "s1"
        self.source_path = "test.md"
        self.source_type = type("ST", (), {"value": "markdown"})()

evidence = [
    FakeRef("Acme Revenue 2023 Legacy Report. In 2023, Acme Corporation reported annual revenue of $3.1 billion USD. Acme employed approximately 10,800 people."),
    FakeRef("Acme Revenue 2025 Current Report. In 2025, Acme Corporation reported annual revenue of $4.7 billion USD. Acme employed approximately 12,400 people."),
    FakeRef("Acme Corporation Company Fact File. Annual revenue (2025): $4.7 billion USD. Number of employees (2025): 12,400."),
]

query = "What was Acme's annual revenue?"

detected = _detect_contradictions(evidence, query=query)
print(f"Detected: {len(detected)}")
for d in detected:
    print(f"  type={d.get('conflict_type')} confidence={d.get('confidence')} metrics={d.get('metric_overlap')} entity={d.get('entity_overlap')}")

filtered = filter_contradictions_by_query(detected, evidence, query)
print(f"\nFiltered: {len(filtered)}")
for f in filtered:
    print(f"  type={f.get('conflict_type')} confidence={f.get('confidence')} metrics={f.get('metric_overlap')}")
