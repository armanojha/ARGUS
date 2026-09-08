# Phase 37: Fast-Path Analysis

**Date:** 2026-09-07
**Status:** Diagnostic — no production changes

---

## 1. Current Fast-Path State

### Existing Fast Path (Phase 06.5.3)

**Trigger:** `classify_complexity(query) == ComplexityTier.FAST`

**Conditions for FAST:**
- Query matches `how (do|can|to)|when (did|was)|define|what is|who is|when did|where is` AND length < 60 chars
- No STRONG signals (compare, cause, abstract, relationship, temporal, contradiction, entity count >= 3, length > 160)

**LLMs saved:** 3 (analyze, plan, assess)

**Measured impact (P37-Q01):**
- Fast path: 2 LLM calls, 1823 tokens, 2262ms LLM, 3077ms E2E
- vs Normal path avg: 6 LLM calls, 4895 tokens, 10644ms LLM, 12062ms E2E
- **Savings: 4 calls, 3072 tokens, 8382ms LLM, 8985ms E2E (74% reduction)**

### Verification Gate (06.5.4)

**Status:** EXISTS but NEVER FIRED in benchmark.

All 8 queries (including the fast-path query) ran verification. The gate checks:
- Simple/low-risk pattern
- High-confidence evidence (top score >= `multiagent_verify_threshold` default 0.8)
- Non-conflicting evidence

**Why it never fires:** The threshold (0.8) is higher than typical retrieval scores, and the pattern check may not match the benchmark patterns.

---

## 2. Per-Pattern Call Analysis

| Pattern | Current Calls | Fast Path? | Min Safe Calls | Skip Candidates |
|---------|--------------|------------|----------------|-----------------|
| simple_lookup | 2 (fast) | YES | 1 (synthesis only) | verification |
| normal_qa | 6 | NO | 2-3 | analyze, plan, (assess for single-iter) |
| technical_explanation | 6 | NO | 3-4 | analyze, plan |
| multi_hop | 6 | NO | 5-6 | analyze (maybe) |
| numerical | 5 | NO | 2-3 | analyze, plan |
| conflict | 7 | NO | 5-6 | analyze (maybe) |
| complex_research | 7 | NO | 6-7 | analyze (maybe) |
| absent_info | 6 | NO | 3-4 | analyze, assess (use retrieval signals) |

---

## 3. Call Skip Candidates

### Candidate A: Skip query_analysis for MODERATE complexity

**Current behavior:** Always runs for non-FAST queries. Outputs ComplexityLevel + suggested_subquestion_count.

**What it provides:**
- ComplexityLevel (SIMPLE/MODERATE/COMPLEX) → used for model tier routing
- suggested_subquestion_count → used by plan node

**Risk of skipping:**
- Loss of complexity classification → model tier defaults to "balanced"
- Loss of subquestion count hint → plan node defaults to `orchestration_default_subquestions` (3)

**Mitigation:** The plan node already handles missing analysis gracefully (defaults to MODERATE). The complexity tier can be computed deterministically from the query (already done by `classify_complexity`).

**Estimated savings:** 1 call, ~1035ms, ~430 tokens per query

### Candidate B: Skip research_planning for simple patterns

**Current behavior:** Always runs for non-FAST queries. Outputs objective, subquestions, entities, budgets.

**What it provides:**
- objective → used in synthesis prompt
- subquestions → drives retrieval iterations
- entities → used in verification
- budgets → clamped by settings anyway

**Risk of skipping:**
- Loss of subquestion decomposition → single-pass retrieval only
- Loss of entity extraction → verification has fewer signals
- Loss of budget recommendations → already clamped by settings

**For simple patterns (normal_qa, numerical):** The plan would likely produce `subquestions=[query]` anyway, making it redundant.

**Estimated savings:** 1 call, ~1500ms (excluding timeout outliers), ~360 tokens per query

### Candidate C: Skip evidence_extraction (assess) on first iteration

**Current behavior:** After first retrieval, runs assess to determine sufficiency.

**What it provides:**
- sufficiency decision → stop or continue
- next_subquery → what to retrieve next

**Risk of skipping:**
- No evidence quality assessment → may synthesize with insufficient evidence
- No next_subquery generation → single-pass retrieval only

**For fast-path queries:** Already skipped (assess not called).

**For numerical/simple patterns:** First retrieval may already have sufficient evidence.

**Estimated savings:** 1 call, ~1173ms, ~870 tokens per query (only for single-iteration queries)

### Candidate D: Skip verification more aggressively

**Current behavior:** Runs for ALL queries (06.5.4 gate never fires).

**What it provides:**
- Confidence annotation
- Contradiction detection
- NEVER replaces the answer (annotation only)

**Risk of skipping:**
- Loss of contradiction detection for conflict queries
- Loss of confidence scoring

**For simple_lookup/numerical:** Low risk — evidence is straightforward.

**Estimated savings:** 1 call, ~1334ms, ~870 tokens per query

---

## 4. Dependency Map

```
query_analysis (A)
    │
    ├──→ [OUTPUT: complexity, subquestion_count]
    │       │
    │       ▼
    research_planning (B)
    │       │
    │       ├──→ [OUTPUT: objective, subquestions, entities, budgets]
    │       │       │
    │       │       ▼
    │       │   retrieval (no LLM)
    │       │       │
    │       │       ├──→ [OUTPUT: evidence chunks]
    │       │       │       │
    │       │       │       ▼
    │       │       │   evidence_extraction/assess (C)
    │       │       │       │
    │       │       │       ├──→ [OUTPUT: sufficient, next_subquery]
    │       │       │       │       │
    │       │       │       │       ├──→ sufficient=True → synthesis (E)
    │       │       │       │       └──→ sufficient=False → retrieval (loop)
    │       │       │       │
    │       │       │       └──→ [short-circuits: empty_retrievals, budget]
    │       │       │
    │       │       └──→ [fast_path] → synthesis (E)
    │       │
    │       └──→ [fast_path] → retrieval → synthesis (E)
    │
    └──→ [fast_path] → retrieval → synthesis (E)

synthesis (E)
    │
    ├──→ [OUTPUT: answer with citations]
    │       │
    │       ▼
    verification (F)
            │
            └──→ [OUTPUT: status, confidence, contradictions] (annotation only)
```

### What Each Call Contributes to Downstream

| Call | Output | Consumed By | Could Use Alternative? |
|------|--------|-------------|----------------------|
| analyze (A) | complexity, subquestion_count | plan (B) | YES — deterministic classifier already provides complexity |
| plan (B) | objective, subquestions, entities | retrieve, assess (C), synthesize (E), verify (F) | PARTIALLY — objective=subquery, subquestions=[query] for simple patterns |
| assess (C) | sufficient, next_subquery | routing decision | PARTIALLY — deterministic short-circuits exist; could expand |
| synthesize (E) | answer | user, verify (F) | NO — this IS the answer |
| verify (F) | status, confidence | annotation only | YES — skip for low-risk queries |

---

## 5. Ablation Test Matrix

| Test | analyze | plan | assess | synthesize | verify | Expected Calls | Expected Savings |
|------|---------|------|--------|------------|--------|----------------|-----------------|
| A. Full current | ✓ | ✓ | ✓ | ✓ | ✓ | 5-7 | baseline |
| B. Skip analyze | ✗ | ✓ | ✓ | ✓ | ✓ | 4-6 | 1 call |
| C. Skip plan | ✓ | ✗ | ✓ | ✓ | ✓ | 4-6 | 1 call |
| D. Skip analyze+plan | ✗ | ✗ | ✓ | ✓ | ✓ | 3-5 | 2 calls |
| E. Skip verify | ✓ | ✓ | ✓ | ✓ | ✗ | 4-6 | 1 call |
| F. Skip analyze+plan+verify | ✗ | ✗ | ✓ | ✓ | ✗ | 2-4 | 3 calls |
| G. Fast path + no verify | ✗ | ✗ | ✗ | ✓ | ✗ | 1 | 1 call (from fast) |

---

## 6. Provider TPM Impact Estimate

**Current:** ~4895 tokens per non-fast query (avg)
**Fast path:** ~1823 tokens per query

| Optimization | Est. Tokens/Query | Queries Before TPM (8K) | Fallback Reduction |
|-------------|-------------------|------------------------|-------------------|
| Current (full pipeline) | ~4895 | ~1.6 | baseline |
| Skip analyze | ~4465 | ~1.8 | minimal |
| Skip plan | ~4535 | ~1.8 | minimal |
| Skip analyze+plan | ~4105 | ~1.9 | moderate |
| Skip verify | ~4025 | ~2.0 | moderate |
| Skip analyze+plan+verify | ~3195 | ~2.5 | significant |
| Fast path (existing) | ~1823 | ~4.4 | major |

**Key insight:** The biggest TPM savings come from the existing fast path (74% token reduction). Additional skips provide incremental savings (15-35%).

---

## 7. Recommended Ablation Priority

1. **Skip analyze+plan for normal_qa/numerical** — highest call savings (2 calls) with low risk
2. **Skip verify for simple_lookup/numerical** — 1 additional call saved, annotation-only
3. **Skip assess (single-iteration) for simple patterns** — 1 call saved, but riskier
4. **Do NOT skip for multi_hop/conflict/complex_research** — these need full pipeline
