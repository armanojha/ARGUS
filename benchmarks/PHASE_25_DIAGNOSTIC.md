# Phase 25 Diagnostic: Architecture Audit & Target Selection

## Purpose

Systematically audit every completed component of ARGUS to identify the single highest-value remaining architectural weakness. The selected target must:
1. Address a genuine production gap (not a theoretical concern)
2. Be implementable without rewriting existing modules
3. Produce measurable improvement in answer quality
4. Not reopen any completed phase

---

## 1. Completed Components Audit

| Phase | Component | Status | Maturity |
|-------|-----------|--------|----------|
| 1-5 | Ingestion pipeline | Complete | High |
| 6-10 | Retrieval (BM25+FAISS, hybrid fusion) | Complete | High |
| 11 | Multi-query expansion | Complete | High |
| 12 | Benchmark suite (9 metrics, 110 questions) | Complete | High |
| 13 | Evidence selector (minimal coverage) | Complete | High |
| 14 | Reranker | Complete | High |
| 15 | Evidence graph | Complete | Medium |
| 16 | Cache | Complete | High |
| 17 | Router (query classification, complexity tiers) | Complete | High |
| 18 | Planner (evidence needs, subquestions) | Complete | High |
| 19 | Ablation & architecture audit | Complete | High |
| 20 | Entity linking | Complete (disabled) | N/A |
| 21 | Embedding optimization | Complete (all-MiniLM-L6-v2) | High |
| 22 | Deterministic verification | Complete (library) | Medium |
| 23 | Memory audit | Complete (no changes justified) | N/A |
| 24 | Adaptive research orchestration | Complete (disabled by default) | High |

---

## 2. Candidate Weakness Areas (Ranked)

### Rank 1: Synthesis Pipeline — No Post-Synthesis Quality Check
**Evidence:**
- `make_synthesize_node` (nodes.py:496-554) is a thin wrapper around a single LLM call
- No claim-evidence alignment verification exists
- No hallucination detection exists
- Synthesis prompt (`prompts.py:107-127`) has zero contradiction awareness
- `_build_result` (graph.py:373-428) silently fabricates citations when LLM produces none (line 380-384)
- Evidence quality signals (strong/weak/conflicted) are not passed to synthesis
- `tier="strong"` is hardcoded for all synthesis calls regardless of evidence quality

**Impact:** The final answer — the only thing the user sees — has no quality gate. A synthesized answer can:
- Make claims unsupported by any cited evidence
- Ignore contradictions in the evidence
- Appear well-cited when citations are fabricated

### Rank 2: Verification is Off by Default and Post-Hoc Only
**Evidence:**
- `_run_selective_verification` (graph.py:431-530) runs AFTER synthesis
- Gated by `settings.verification_enabled` (default: False)
- When triggered, annotates only — never blocks or revises the answer
- Deterministic verification module (`deterministic.py`, 528 lines) is dead code in production
- Contradiction detector (`contradiction.py`) is never called from the main loop

**Impact:** The verification subsystem exists but provides zero production value. However, this is a design choice (optimization-first, verification-optional) that was explicitly made in Phase 07b. Reopening it conflicts with Phase 25 constraints.

### Rank 3: Benchmark Metrics Don't Measure Answer Quality
**Evidence:**
- `answer_faithfulness` (metrics.py:145-153) is token F1 — purely lexical
- `claim_support_rate` uses 50% lexical overlap as a proxy
- No semantic similarity, entailment checking, or LLM-as-judge
- No hallucination metric exists
- The benchmark `judge` parameter (metrics.py:148) is supported but not wired

**Impact:** We cannot measure whether ARGUS produces correct answers. But this is a measurement gap, not an architecture gap — fixing the architecture (Rank 1) would also give us something worth measuring.

### Rank 4: Evidence Graph Underutilized
**Evidence:**
- Graph is built during ingestion but rarely queried during retrieval
- `EdgeType` supports SUPPORTS/CONTRADICTS/RELATED_BY but these edges are not used in synthesis
- No graph-guided reasoning path exists

**Impact:** Moderate. The graph adds retrieval diversity but does not improve answer quality directly.

### Rank 5: Memory System Dormant
**Evidence:**
- Memory is disabled by default (`memory_enabled=False`)
- Phase 23 audit concluded no production changes justified
- User explicitly said "Do NOT add a giant memory system prematurely"

**Impact:** Low for current production. Cross-session learning is a future concern.

### Rank 6: Multimodal/Document Understanding
**Evidence:**
- OCR worker uses separate `.venv-ocr` (Python 3.11)
- Pipeline: OCR → structure → chunks → embeddings → retrieval
- No evaluation of OCR quality impact on downstream answer quality

**Impact:** Low. OCR works; downstream pipeline handles chunks uniformly.

---

## 3. Selected Target: Synthesis Quality Gate (Rank 1)

### Rationale

The synthesis pipeline is the **single point of failure** for answer quality. Every upstream improvement (better retrieval, better planning, better evidence selection) is wasted if the synthesis step produces an unsupported or contradictory answer.

The specific gaps are:
1. **No claim-evidence alignment check** — the LLM can assert facts not present in cited evidence
2. **No contradiction awareness** — the synthesis prompt never mentions evidence may conflict
3. **Silent citation fabrication** — fallback auto-attaches top-3 evidence as citations
4. **No evidence quality signals** — strong and weak evidence presented identically

### What Phase 25 Will Do

Add a deterministic post-synthesis grounding check that:
1. Verifies every answer claim is traceable to a cited evidence passage
2. Detects unsupported claims (claims with no lexical/semantic anchor in cited evidence)
3. Injects contradiction awareness into the synthesis prompt when `contradiction_signals` is non-empty
4. Replaces the silent citation fabrication with a degraded-answer warning

### What Phase 25 Will NOT Do

- Rewrite the synthesis LLM call
- Add an LLM-based hallucination detector (too expensive for production)
- Replace the verification engine
- Change the benchmark metrics
- Modify the evidence graph
- Touch the adaptive research policy
- Add new dependencies

---

## 4. Implementation Plan

### Step 1: Add contradiction context to synthesis prompt
**File:** `app/orchestration/prompts.py`
- Modify `build_synthesis_messages` to accept optional `contradiction_signals` parameter
- When contradictions exist, add a "CONTRADICTION ALERT" section to the synthesis prompt
- Instruct the LLM to acknowledge conflicts and present both sides

### Step 2: Add evidence quality annotations to synthesis
**File:** `app/orchestration/prompts.py`
- Modify `_format_evidence_block` to include retrieval score (e.g., `[1] (score: 0.87, source: ...)`)
- This gives the LLM a signal about which evidence is stronger

### Step 3: Add deterministic claim grounding check
**File:** `app/orchestration/nodes.py`
- After synthesis, run a lightweight check: split answer into sentences, verify each sentence has at least one bracket citation, verify each citation resolves to a real evidence chunk
- Log warnings for unsupported sentences (do not block — just annotate)

### Step 4: Fix citation fabrication fallback
**File:** `app/orchestration/graph.py`
- When no citations found, instead of silently attaching top-3 evidence, add a `citation_fallback` warning
- The answer still gets the fallback citations (to preserve provenance), but the user/metadata sees the warning

### Step 5: Wire contradiction signals through to synthesis
**File:** `app/orchestration/nodes.py`
- Pass `state["contradiction_signals"]` from the assess node through to the synthesis node
- The synthesis node passes them to `build_synthesis_messages`

### Step 6: Tests
**Files:** `tests/orchestration/test_synthesis_quality.py` (new)
- Test contradiction-aware synthesis prompt generation
- Test evidence quality annotations in prompt
- Test claim grounding check (sentences with/without citations)
- Test citation fabrication warning
- Test degraded answer path

### Step 7: Benchmark evaluation
**Files:** `benchmarks/benchmark_synthesis_quality.py` (new)
- Run baseline (current synthesis) vs improved (quality-gated synthesis) on the 38-query eval plan
- Measure: claim_support_rate, citation_correctness, answer_faithfulness
- Report in `benchmarks/PHASE_25_REPORT.md`

---

## 5. Risk Assessment

| Risk | Mitigation |
|------|-----------|
| LLM call cost increase | No new LLM calls — all changes are deterministic pre/post processing |
| Regression in existing tests | Run full test suite (838+ tests) after each step |
| Synthesis prompt bloat | Contradiction section only added when contradictions exist |
| Overly aggressive grounding check | Use lexical overlap (50% threshold) consistent with existing `claim_support_rate` |
| Breaking the citation fallback | Keep fallback citations but add warning annotation |

---

## 6. Expected Outcomes

- Synthesis prompt includes contradiction awareness when evidence conflicts
- Evidence quality (retrieval score) visible to synthesis LLM
- Every answer sentence checked for citation grounding
- Citation fabrication produces a visible warning
- 6+ new unit tests, 2+ integration tests
- Benchmark comparison showing measurable improvement in claim_support_rate
- Total test count: 844+ with 0 failures
