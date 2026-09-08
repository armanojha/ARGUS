"""Verify: does _normalize_number_with_unit handle % correctly?"""
from app.orchestration.nodes import _normalize_number_with_unit

tests = ["35%", "52%", "35 percent", "100%", "$3.1 billion", "2024"]
for t in tests:
    v, u = _normalize_number_with_unit(t)
    print(f"  _normalize_number_with_unit({t!r}) = ({v}, {u!r})")
