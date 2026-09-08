# ARGUS v0.1.0 — First Public Release

First public release of ARGUS, an iterative RAG research system with evidence
verification, contradiction detection, reasoning traces, Brain UI observability,
and reproducible evaluation benchmarks.

## What it does

- **Hybrid retrieval** — BM25 + FAISS dense search with policy-routed dispatch
  (concurrent, bounded fan-out) and cross-encoder reranking.
- **Evidence verification** — deterministic claim-support checking plus
  LLM-assisted verification when a provider is available.
- **Contradiction detection** — deterministic negation/numeric detector with
  topic-coherence and entity-overlap gates, query-aware filtering, and an
  opt-in LLM semantic verification stage (`conflict_semantic_check_enabled`).
- **Reasoning traces** — every run records per-node runtime traces and
  per-iteration strategy snapshots; a WHY-graph (query → tasks → hypothesis →
  evidence → claims → inference → decision → answer, plus rejected
  counterclaims) is materializable from any finished result.
- **Brain UI** — single-file HTML/D3.js observability layer: pipeline view
  with per-node inspector, knowledge-graph view, strategy evolution, and
  "Why this answer?" reasoning traversal.
- **E2E intelligence benchmark** — 38 fixed cases / 12-doc fixed corpus,
  deterministic, CI-safe: Recall@8 **0.987**, gold coverage **0.904**,
  contradiction recall **1.000**. See `benchmarks/E2E_INTELLIGENCE_REPORT.md`.

## Known limitations (read before evaluating)

- **Abstention gap:** the deterministic absent-info gate scores **0.000** on
  absent queries — they proceed to synthesis instead of abstaining.
- **Contradiction precision** of raw deterministic signals is **0.079**
  (query-aware filtering is off by default).
- **Live LLM evaluation pending:** citation accuracy, grounding %, end-to-end
  latency, and cost/query are unmeasured (rate-limit constrained).
- Memory system is disabled by default; graph extraction needs an LLM provider.
- CI verification of the latest commits was still pending at tag time; see
  GitHub Actions history for current status.

## Reproduce

```bash
pip install -e ".[core,retrieval,graph,multimodal,dev-test]"
ruff check app/
python -m pytest tests/ -q
python benchmarks/e2e_intelligence.py
```
