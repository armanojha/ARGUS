# Phase 37: Complete LLM Call Graph

**Date:** 2026-09-07
**Status:** Diagnostic — no production changes

---

## 1. Complete LLM Call Inventory

### Production Pipeline (Defaults)

| # | Call | call_type | Provider/Model | Avg Latency | Input Tokens | Output Tokens | Required? | Skip Candidate? |
|---|------|-----------|----------------|-------------|-------------|--------------|-----------|----------------|
| 1 | Query Analysis | `query_analysis` | groq/gpt-oss-20b | ~900ms | 150-300 | 80-150 | YES | **YES** — already skipped for FAST tier |
| 2 | Research Planning | `research_planning` | gemini/flash-lite | ~1,700ms | 250-500 | 300-500 | YES | **YES** — already skipped for FAST tier |
| 3 | Evidence Assessment | `evidence_extraction` | groq/gpt-oss-120b | ~1,400ms | 2,500-5,000 | 200-400 | YES | **YES** — already skipped for FAST tier; also has deterministic short-circuits |
| 4 | Synthesis | `synthesis` | nvidia_nim/nemotron-3-ultra | ~1,700ms | 2,500-5,000 | 500-1,500 | YES | NO — final answer production |
| 5 | Verification | `verification` | gemini/flash-lite | ~2,000ms | 3,000-8,000 | 500-1,000 | CONDITIONAL | **YES** — gated by 06.5.4 logic; skipped for simple/low-risk |

**Disabled by default (not in production path):**
- Calls #4a/#4b: Two-pass verified synthesis (`verified_synthesis_enabled=False`)
- Calls #6-#10: Multi-agent debate (`multiagent_enabled=False`)
- Adaptive research pre-check (`adaptive_research_enabled=False`)
- Memory enhancement (`memory_enabled=False`)

### Feature-Flagged Calls (Currently OFF)

| # | Call | Feature Flag | When ON |
|---|------|-------------|---------|
| 4a | Claim Generation (Pass 1) | `verified_synthesis_enabled=True` | Replaces Call #4 |
| 4b | Verified Rendering (Pass 2) | `verified_synthesis_enabled=True` | Conditional on early exit |
| 6 | Researcher Agent | `multiagent_enabled=True` | Up to 3 rounds × 1 call |
| 7 | Skeptic Agent | `multiagent_enabled=True` + conditions | Up to 3 rounds × 1 call |
| 8 | Alt-Hypothesis Agent | `multiagent_enabled=True` + conditions | Up to 3 rounds × 1 call |
| 9 | Verifier Agent | `multiagent_enabled=True` | Up to 3 rounds × 3 calls |
| 10 | Judge Agent | `multiagent_enabled=True` | Up to 3 rounds × 1 call |

---

## 2. Execution Order

### Normal Path (BALANCED/STRONG tier, non-simple query)

```
START
  │
  ▼
┌─────────────────────────┐
│ 1. ANALYZE              │  call_type=query_analysis
│    router.complete()    │  provider: groq/gpt-oss-20b
│    ~900ms               │  structured output: QueryAnalysis
└─────────┬───────────────┘
          │
          ▼
┌─────────────────────────┐
│ 2. PLAN                 │  call_type=research_planning
│    router.complete()    │  provider: gemini/flash-lite
│    ~1,700ms             │  structured output: ResearchPlan
└─────────┬───────────────┘
          │
          ▼
┌─────────────────────────┐
│    RETRIEVE (no LLM)    │  BM25 + FAISS + fusion + rerank
│    ~2,000ms             │  EvidenceSelector ~1.5ms
└─────────┬───────────────┘
          │
          ▼
┌─────────────────────────┐
│ 3. ASSESS (iteration 1) │  call_type=evidence_extraction
│    router.complete()    │  provider: groq/gpt-oss-120b
│    ~1,400ms             │  structured output: EvidenceAssessment
│    [loop if !sufficient]│  may propose next_subquery
└─────────┬───────────────┘
          │ sufficient?────yes───┐
          │ no                    │
          ▼                       │
    RETRIEVE (loop)               │
          │                       │
          ▼                       │
    ASSESS (iteration 2-3)        │
          │ sufficient?──yes──────┤
          │ no → loop/retry       │
          ▼                       ▼
┌─────────────────────────┐
│ 4. SYNTHESIZE           │  call_type=synthesis
│    router.complete()    │  provider: nvidia_nim/nemotron-3-ultra
│    ~1,700ms             │  free-form text output
└─────────┬───────────────┘
          │
          ▼
┌─────────────────────────┐
│    BUILD RESULT          │  deterministic, no LLM
└─────────┬───────────────┘
          │
          ▼
┌─────────────────────────┐
│ 5. VERIFY (conditional) │  call_type=verification
│    router.complete()    │  provider: gemini/flash-lite
│    ~2,000ms             │  structured output: VerifierOutput
│    [gated by 06.5.4]    │  may be skipped
└─────────┬───────────────┘
          │
          ▼
        END
```

### Fast Path (FAST tier, simple query)

```
START
  │
  ▼
┌─────────────────────────┐
│    RETRIEVE (no LLM)    │  single-pass retrieval
│    ~2,000ms             │
└─────────┬───────────────┘
          │
          ▼
┌─────────────────────────┐
│ 4. SYNTHESIZE           │  call_type=synthesis
│    ~1,700ms             │
└─────────┬───────────────┘
          │
          ▼
┌─────────────────────────┐
│ 5. VERIFY (conditional) │  may be skipped by 06.5.4
└─────────┬───────────────┘
          │
          ▼
        END
```

**LLMs saved by fast path:** 3 calls (analyze, plan, assess)

---

## 3. Call Dependencies

```
query_analysis (Call 1)
    │
    ├──→ plan (Call 2)
    │       │
    │       ├──→ retrieval (no LLM)
    │       │       │
    │       │       ├──→ assess (Call 3)
    │       │       │       │
    │       │       │       ├──→ sufficient? → synthesize (Call 4)
    │       │       │       └──→ not sufficient → retrieval (loop)
    │       │       │
    │       │       └──→ [fast_path] → synthesize (Call 4)
    │       │
    │       └──→ [fast_path] skips this entire subtree
    │
    └──→ [fast_path] skips analyze + plan entirely

synthesize (Call 4)
    │
    └──→ verification (Call 5, conditional)
```

### What Each Call Contributes

| Call | Output | Downstream Impact |
|------|--------|-------------------|
| analyze | ComplexityLevel, suggested_subquestion_count | Determines plan decomposition |
| plan | objective, subquestions, entities, budgets | Drives entire retrieval loop |
| assess | sufficient, next_subquery, stop_reason | Controls loop termination and next retrieval |
| synthesize | answer (cited text) | Final output to user |
| verify | status, confidence, contradictions | Annotation only — never replaces answer |

---

## 4. Existing Fast-Path Logic

### 4.1 Complexity-Based Fast Path (Phase 06.5.3)

**Trigger:** `classify_complexity(query) == ComplexityTier.FAST`

**Conditions for FAST:**
- Query matches `how (do|can|to)|when (did|was)|define|what is|who is|when did|where is` AND length < 60 chars
- No STRONG signals (compare, cause, abstract, relationship, temporal, contradiction, entity count >= 3, length > 160)

**Effect:**
- `fast_path=True` set in initial state
- Pre-computed `QueryAnalysis(ComplexityLevel.SIMPLE)` injected
- Pre-computed `_fast_path_plan(query)` injected with `subquestions=[query]`
- `_route_entry()` returns `"retrieve"` directly (skips analyze + plan)
- `_route_after_retrieve()` returns `"synthesize"` directly (skips assess)
- **Saves 3 LLM calls** (analyze, plan, assess)

### 4.2 Deterministic Short-Circuits in Assess Node

| Condition | Effect |
|-----------|--------|
| `consecutive_empty_retrievals >= 2` | Stops immediately (NO_NEW_EVIDENCE) |
| `iteration >= max_iterations` | Stops (BUDGET_EXHAUSTED) |
| `tokens_used >= token_budget` | Stops (BUDGET_EXHAUSTED) |

These are zero-LLM checks that prevent wasting an assess call when bounds are already exceeded.

### 4.3 Adaptive Research Policy Pre-Check (Phase 24.1, OFF by default)

When `adaptive_research_enabled=True`:
- Runs deterministic sufficiency model BEFORE the LLM assess call
- If `decision.action == "synthesize"` → skips the LLM assess call entirely
- Triggers: strong evidence (count >= 5, diversity >= 3, avg top-3 score >= 0.7, coverage >= 1.0)

### 4.4 Verification Skip Gates (06.5.4)

Verification is skipped when ANY of:
- `verification_enabled=False` (currently True)
- `check_call_ceiling()` — at call budget limit
- `should_skip_verification(plan, evidence, settings)` — simple/low-risk, high-confidence, non-conflicting
- No answer produced

### 4.5 Two-Pass Early Exit (Phase 29, OFF by default)

When `verified_synthesis_enabled=True`:
- `_should_early_exit(claim_set, pattern)` returns True for simple patterns with no rejected claims
- Skips Pass 2 LLM call — uses deterministic renderer

---

## 5. Loop Structures

### Main Retrieval-Assessment Loop

- **Nodes:** retrieve → assess → stop_check → retrieve (repeat)
- **Max iterations:** `orchestration_max_iterations` (default 3)
- **LLM calls per iteration:** 1 assess call
- **Total assess calls:** 1 to 3 per query
- **Break conditions:** sufficient=True, consecutive_empty >= 2, budget exhausted

### Multi-Agent Debate Loop (OFF by default)

- **Rounds:** up to `multiagent_max_rounds` (default 3)
- **Calls per round:** 2-7 (Researcher + optional Skeptic + optional AltHyp + Verifier(1-3) + Judge)
- **Total debate calls:** 2-21

---

## 6. Token Budget Per Query

| Scenario | LLM Calls | Est. Total Tokens |
|----------|-----------|-------------------|
| **Fast path** (simple) | 1 (synthesis) | 3K-6.5K |
| **Standard, 1 iteration** | 4 (analyze + plan + assess + synthesize) | 5.5K-11K |
| **Standard, 3 iterations** | 6 (analyze + plan + 3× assess + synthesize) | 9.5K-19K |
| **With verification** | +1 | +3.5K-9K |
| **Worst case** | ~7 | ~28K |

### Call Ceiling

- **Config:** `multimodel_call_ceiling` = 16
- **Enforced by:** `check_call_ceiling()` before every provider call
- **Effect:** Raises `CallCeilingExceededError` when exceeded

---

## 7. Key Configuration

| Setting | Default | Description |
|---------|---------|-------------|
| `orchestration_max_iterations` | 3 | Max retrieval-assessment loops |
| `orchestration_default_subquestions` | 3 | Fallback subquestion count |
| `orchestration_token_budget` | 6000 | Token budget per query |
| `orchestration_llm_timeout` | 30.0s | Per-call timeout |
| `orchestration_timeout` | 120s | Overall run timeout |
| `verification_enabled` | True | Post-synthesis verification |
| `verified_synthesis_enabled` | False | Two-pass verified synthesis |
| `multiagent_enabled` | False | Multi-agent debate |
| `adaptive_research_enabled` | False | Deterministic pre-check |
| `memory_enabled` | False | Persistent memory |
| `multimodel_call_ceiling` | 16 | Max LLM calls per run |
| `evidence_selection_max_chunks` | 8 | Chunks sent to LLM |
| `evidence_selection_max_tokens` | 6000 | Token budget for evidence |

---

## 8. Provider Routing Chains

| call_type | Primary | Fallbacks |
|-----------|---------|-----------|
| `query_analysis` | groq/gpt-oss-20b | gemini/flash-lite → cerebras/gpt-oss-120b → zai/glm-4.5-flash → zen/nemotron-3.5-lightning-free |
| `research_planning` | gemini/flash-lite | groq/gpt-oss-120b → cerebras/gpt-oss-120b → zai/glm-4.5-flash → zen/big-pickle |
| `evidence_extraction` | groq/gpt-oss-120b | gemini/flash-lite → cerebras/gpt-oss-120b → zai/glm-4.5-flash → zen/mimo-v2.5-free |
| `synthesis` | nvidia_nim/nemotron-3-ultra | groq/gpt-oss-120b → gemini/flash-lite → cerebras/gpt-oss-120b → zai/glm-4.5-flash → zen/nemotron-3-ultra-free |
| `verification` | gemini/flash-lite | groq/gpt-oss-20b → cerebras/gpt-oss-120b → zai/glm-4.5-flash → zen/mimo-v2.5-free |

---

## 9. Fallback Behavior Summary

| Call | On LLM Failure | On Empty Response | On Malformed Output |
|------|----------------|-------------------|---------------------|
| analyze | Returns `QueryAnalysis(MODERATE)` + warning | Returns fallback | Returns fallback |
| plan | Returns `ResearchPlan(query, [query])` + warning | Returns fallback | Returns fallback |
| assess | Returns `sufficient=True, stop_reason=ASSESSMENT_ERROR` | Returns fallback | Returns fallback |
| synthesize | Sets `answer=""`, degrades to evidence bullets | Degrades to evidence bullets | N/A (free-form) |
| verify | Returns `status="error"` annotation | Returns error annotation | Returns error annotation |
