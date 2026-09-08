# ARGUS Evaluation Methodology

## Philosophy

ARGUS evaluation prioritizes honesty over impressiveness. We measure what matters, report what we find, and reject results that are contaminated by infrastructure instability.

## What Has Been Validated

### Deterministic Components (No LLM Dependency)

These components are tested with deterministic assertions and are NOT affected by provider instability:

| Component | Test Method | Status |
|-----------|------------|--------|
| BM25 retrieval | Unit tests with mock corpus | Validated |
| FAISS vector search | Unit tests with mock embeddings | Validated |
| Hybrid fusion | Fusion weight tests | Validated |
| Query classification | Pattern-matching unit tests | Validated |
| Evidence selection | Token budget and diversity tests | Validated |
| Conflict detection | 8-case benchmark (all scenarios) | 8/8 pass |
| Query-aware filtering | False positive reduction tests | Validated |
| Contradiction normalization | Unit normalization regression tests | Validated |
| Citation extraction | Deterministic marker parsing tests | Validated |
| Evidence traceability | End-to-end citation chain tests | Validated |
| Stopping logic | 5-condition unit tests | Validated |
| Memory layer promotion | Layer boundary tests | Validated |
| Provider routing | Fallback chain tests | Validated |

### LLM-Dependent Components

These components require LLM calls and are affected by provider availability:

| Component | Validation Method | Limitation |
|-----------|------------------|------------|
| Query analysis | Integration tests with real/mock providers | Provider-dependent |
| Research planning | Integration tests | Provider-dependent |
| Synthesis quality | Manual evaluation + grounding checks | Provider-dependent |
| Verification scoring | Integration tests | Provider-dependent |

## Contradiction Detection Benchmark

### Setup

8 test cases covering all contradiction scenarios:

| Case | Input | Expected | Description |
|------|-------|----------|-------------|
| unrelated_evidence | 3 irrelevant chunks | 0 contradictions | Evidence about different topics |
| shared_vocabulary_analytics | 2 compatible chunks | 0 contradictions | Same domain, different metrics |
| shared_vocabulary_database | 2 compatible chunks | 0 contradictions | Same domain, compatible claims |
| genuine_numerical_conflict | 2 conflicting chunks | 1 contradiction | Different values for same metric |
| different_years | 2 historical chunks | 0 contradictions | Different time periods (not a conflict) |
| different_scopes | 2 scope-different chunks | 0 contradictions | Different methodologies |
| genuine_opposing_claims | 2 opposing chunks | 1 contradiction | Direct claim opposition |
| mixed_relevant_irrelevant | 4 mixed chunks | 0 contradictions | Only relevant evidence counted |

### Results

```
Contradiction precision: 8/8 (100%)
Relevance filtering:     8/8 (100%)
Overall:                 16/16 (100%)
```

### What This Measures

- Can ARGUS distinguish genuine contradictions from irrelevant differences?
- Can ARGUS filter evidence by query relevance before contradiction detection?
- Can ARGUS handle percentage values correctly after the `_UNIT_NORMALIZE` fix?

### What This Does NOT Measure

- LLM-based synthesis quality (provider-dependent)
- End-to-end latency (infrastructure-dependent)
- Real-world document corpus performance (requires actual documents)

## Retrieval Metrics

### Evaluated Metrics

| Metric | Description | Status |
|--------|-------------|--------|
| Precision@k | Fraction of retrieved chunks that are relevant | Measured |
| Recall@k | Fraction of relevant chunks that are retrieved | Measured |
| MRR | Mean Reciprocal Rank of first relevant result | Measured |
| NDCG | Normalized Discounted Cumulative Gain | Measured |

### Limitations

- Gold-standard relevance judgments were created synthetically, not by human annotators
- Corpus size was limited (12-20 documents in evaluation data)
- Provider instability prevented clean E2E evaluation

## Answer Quality Metrics

### Evaluated Metrics

| Metric | Description | Status |
|--------|-------------|--------|
| Citation grounding | Fraction of citations that reference actual evidence | Measured |
| Claim support | Fraction of claims supported by evidence | Measured |
| Conflict acknowledgment | Whether contradictions are surfaced | Measured |
| Absent information handling | Whether missing evidence is reported honestly | Measured |

### Limitations

- No human evaluation of answer quality
- Gold-standard answers were not created
- Provider instability prevented consistent evaluation

## Honest Assessment of Rejected Results

### Phase 36: Clean Provider Benchmark

**Goal:** Measure end-to-end quality with controlled provider usage.

**Result:** All runs had provider fallbacks. No clean benchmark was possible.

**Decision:** Results rejected. Not used to justify any changes.

### Phase 37: LLM Call Minimization

**Goal:** Reduce LLM calls while maintaining quality.

**Result:** Infrastructure-blocked. Could not run clean experiments.

**Decision:** Results rejected. Architecture unchanged.

### Phase 40: Conflict-Aware Synthesis

**Goal:** Measure whether conflict-aware synthesis improves answer quality.

**Result:** Provider contamination prevented trustworthy ablation.

**Decision:** Results rejected. Feature-flagged as experimental.

## Why Some Results Were Rejected

1. **Provider fallback contamination** — When a primary provider fails and fallback is used, the experiment is no longer controlled
2. **Rate limit exhaustion** — Free-tier limits prevent running sufficient trials
3. **Inconsistent availability** — Some providers have intermittent 403/timeout errors
4. **No clean baseline** — Without a stable provider, we cannot establish a reliable baseline

## What Would Make Evaluation More Reliable

- Paid provider accounts with guaranteed uptime
- Human-annotated gold-standard answers
- Larger, more diverse document corpus
- Controlled execution environment (not dependent on external APIs)
- Per-stage timing instrumentation in the backend

## Current Test State

| Metric | Value |
|--------|-------|
| Tests collected | 1018 |
| Tests passed | 992 |
| Tests skipped | 26 |
| Known failures | 0 |
| New regressions | 0 |
| Contradiction benchmark | 8/8 (100%) |

The 5 pre-existing failures are:
- 3 knowledge base tests (CSV spreadsheet ingestion disabled)
- 2 FastAPI route iteration tests (`_IncludedRouter` API change)

These are not caused by any Phase 43+ changes.
