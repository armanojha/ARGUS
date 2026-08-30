"""ARGUS evaluation suite (Phase 12.3 benchmark + 12.4 ablation).

This package is deliberately isolated from the core `app/` reasoning
architecture (Phase 12 file layout). It builds a benchmark question set,
runs items through **existing** ARGUS pipelines, computes deterministic
metrics, and compares ablation variants — it never alters core modules and
never selects models (routing stays explicit server-side).
"""