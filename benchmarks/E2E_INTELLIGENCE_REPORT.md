# E2E Intelligence Benchmark

## Why this exists

Unit-test counts measure code coverage, not intelligence. This benchmark
measures fixed, observable research behaviors on fixed inputs: 38 eval cases
across 10 classes against a fixed 12-document corpus (`corpus_v1`).

## Method (deterministic, CI-safe, no LLM)

- Corpus ingested into an **isolated tmp EvidenceStore** (never touches `data/`).
- Real production retrieval: BM25 + FAISS + `all-MiniLM-L6-v2`, `top_k=8`.
- Real deterministic detectors: `_detect_contradictions`, `_is_evidence_absent`.
- Metrics computed against gold `supporting_docs`, `gold_facts`, `conflict` / `absent` flags.

Run: `python benchmarks/e2e_intelligence.py`
Results: `benchmarks/results/e2e_intelligence.json`

## Results (measured 2026-09-09, deterministic mode)

| Metric | Value | Honest reading |
|--------|-------|----------------|
| Retrieval Recall@8 | **0.987** | Retrieval finds the right docs ~always |
| Retrieval Precision@8 | **0.155** | Expected: supporting sets are 1–2 docs of 8 retrieved from 12 (ceiling ≈ 0.25) |
| Gold-fact coverage | **0.904** | 90% of gold facts appear in retrieved text |
| Contradiction recall | **1.000** | All 3 conflict cases flagged |
| Contradiction precision | **0.079** | Detector fires on ~everything (raw signals, query filtering OFF by default) |
| Abstention-trigger accuracy | **0.000** | **Real gap:** the deterministic absent-info gate never fires on the 4 absent cases — they proceed to synthesis instead of abstaining |
| Avg retrieval latency | **48ms** | Retrieval only, not end-to-end |

Per-class detail is in `benchmarks/results/e2e_intelligence.json`.

## What this does NOT yet measure (live columns, pending)

Citation accuracy, claim-level grounding %, abstention accuracy on full runs,
LLM calls, end-to-end latency, and cost/query require live provider runs and
are intentionally NOT reported here. The harness accepts `--live` for future
rate-limit-tolerant runs. No numbers are reported for runs that did not happen.
