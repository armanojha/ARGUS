# Phase 26 Diagnostic: Answer Quality & Citation Evaluation Audit

## Purpose

Inspect the actual ARGUS implementation to determine what answer-quality evaluation capabilities exist, what is missing, and what the smallest architecture change would be to evaluate the missing dimensions.

---

## 1. What Exactly Constitutes a "Claim" in the Current Answer?

**Current state:** There is no claim decomposition. The entire answer text is treated as a single claim in the verification engine (`graph.py:468` passes `claim_text=answer` as one string). The `check_claim_grounding()` function (Phase 25) splits the answer into sentences via regex `(?<=[.!?])\s+`, but this is purely structural — it only checks whether a sentence has a bracket citation, not whether the citation supports the claim.

**Problem:** A sentence like "The company generated $20M revenue and employed 500 people. [1]" is treated as one unit. If the cited chunk supports revenue but not employee count, the current system counts it as fully grounded.

**What's needed:** A claim is a single factual assertion that can be independently verified against evidence. A compound sentence may contain multiple claims.

---

## 2. How Are Citations Represented?

**Implementation** (`nodes.py:564-577`):
- Bracket markers `[N]` / `【N】` in the answer text
- Full-width characters normalized to ASCII
- Parsed by `extract_cited_indices()` → 1-based integer indices
- Invalid/out-of-range markers silently dropped
- Deduplicated, first-seen order

**In the final result** (`graph.py:386-408`):
- Each index maps to an `OrchestrationCitation` with:
  - `ref_id` (1-based marker)
  - `chunk_id` (UUID)
  - `document_id` (UUID)
  - `source_id` (UUID)
  - `source_path` (str)
  - `source_type` (str)
  - `text` (str — the chunk text)
  - `page_start`, `page_end` (int | None)
  - `section_path` (str | None)
  - `score` (float — retrieval score)
  - `metadata` (dict)

**Citation IDs are stable within a single run.** They are 1-based indices into the evidence list, which is ordered by retrieval rank. The same chunk may appear at different indices across runs if retrieval order changes.

---

## 3. How Are Citations Resolved to Evidence?

**Resolution path:**
1. `extract_cited_indices(answer, len(evidence))` → list of int indices
2. For each index `idx`: `evidence[idx - 1]` → `EvidenceRef` object
3. `EvidenceRef` contains `chunk_id`, `document_id`, `source_path`, `text`, `score`
4. Mapped to `OrchestrationCitation` with full provenance

**Can a citation be checked against the cited chunk?** Yes. The `OrchestrationCitation.text` field contains the exact chunk text that the citation references. This is sufficient for lexical/structural comparison.

---

## 4. Can We Determine Whether the Cited Chunk Actually Supports the Claim?

**Current capability:** Partially. The existing `claim_support_rate` metric in `benchmarks/metrics.py:97-124` uses a lexical overlap proxy: a sentence is "supported" if it has a valid bracket citation whose chunk is gold, OR shares ≥50% lexical tokens with a gold chunk.

**Limitation:** This checks whether the sentence *overlaps textually* with the evidence, not whether the evidence *logically supports* the claim. A sentence about "revenue" may share keywords with a chunk about "revenue" but the chunk could cite a different number.

**What's needed:** For each claim-sentence, we need to:
1. Extract the factual assertion (especially numbers, names, dates)
2. Check if the cited chunk contains matching facts
3. Determine if the support is full, partial, or absent

---

## 5. Can We Detect Partially Supported Claims?

**Current capability:** No. The `check_claim_grounding()` function (Phase 25) only returns binary: cited or not cited. The `claim_support_rate` metric uses ≥50% lexical overlap as a threshold but doesn't distinguish partial from full support.

**What's needed:** A claim like "The company generated $20M revenue and employed 500 people" should be decomposed into two sub-claims, each evaluated independently against the cited evidence.

---

## 6. Can We Detect Unsupported Numerical Claims?

**Current capability:** Partially. The `TextContradictionDetector` in `deterministic.py:174` can extract numerical claims from evidence chunks via regex patterns (revenue, employees, utilization, output, founded date). But it operates on evidence chunks, not on answer claims.

**What's needed:** Apply the same numerical extraction to answer sentences, then compare extracted numbers against numbers in the cited evidence chunks.

---

## 7. Can We Detect Contradictions Between Cited Evidence?

**Current capability:** The `TextContradictionDetector` can detect cross-document contradictions. However:
- It is not called from the main orchestration loop
- It operates on evidence chunks, not on the answer
- The `contradiction_signals` state field is always empty in production

**What's needed:** When the answer cites multiple chunks that contradict each other, the evaluator should detect this.

---

## 8. Can We Determine Whether the Answer Omitted Required Information?

**Current capability:** No. The evaluator has no concept of "completeness." The `gold_facts` in `eval_plan_v1.json` provide expected factual strings, but no metric checks whether the answer includes all of them.

**What's needed:** Compare the answer against `gold_facts` to determine coverage.

---

## 9. What Evaluation Capabilities Already Exist?

| Capability | Location | What It Does | Limitation |
|------------|----------|-------------|------------|
| `claim_support_rate` | `metrics.py:97` | Lexical overlap between sentences and gold chunks | Not claim-level; lexical only |
| `answer_faithfulness` | `metrics.py:145` | Token F1 vs gold answer | Requires gold answer text; lexical only |
| `citation_correctness` | `metrics.py:71` | Fraction of citations pointing to gold chunks | Requires gold chunk IDs; structural only |
| `evidence_precision` | `metrics.py:64` | Fraction of cited chunks that are gold | Requires gold chunk IDs |
| `contradiction_recall` | `metrics.py:127` | Whether expected contradiction is detected | Binary; requires gold expectation |
| `temporal_accuracy` | `metrics.py:138` | Whether gold year appears in answer | Requires gold years |
| `check_claim_grounding` | `nodes.py:584` | Whether each sentence has a citation | Structural only; no semantic check |
| `TextContradictionDetector` | `deterministic.py:174` | Cross-document numerical contradictions | Not wired into answer evaluation |
| `EvidenceNeedCoverageVerifier` | `deterministic.py:50` | Evidence coverage per plan need | Operates on plan needs, not answer claims |
| `EvidenceConfidenceScorer` | `deterministic.py:472` | Aggregate confidence from coverage/agreement | Operates on plan needs, not answer claims |

---

## 10. What Is Missing?

### Critical Gaps

1. **No claim decomposition**: The answer is not split into individual factual assertions. A compound sentence with two facts is treated as one unit.

2. **No claim→evidence semantic alignment**: We can check if a citation *exists* but not if the cited chunk *supports the specific claim*. A citation to a relevant-looking chunk may not contain the specific fact asserted.

3. **No partial support detection**: No distinction between "evidence fully supports," "evidence partially supports," and "evidence is topically related but does not support."

4. **No unsupported claim rate**: We cannot measure what fraction of answer claims lack adequate evidence support.

5. **No citation-to-claim alignment**: We cannot verify that `[1]` after "revenue was $20M" actually cites a chunk mentioning $20M revenue.

6. **No numerical consistency check**: We cannot verify that numbers in the answer match numbers in the cited evidence.

7. **No completeness evaluation**: We cannot measure whether the answer includes all expected facts from `gold_facts`.

8. **No contradiction acknowledgement evaluation**: We cannot measure whether the answer acknowledges contradictions when they exist in the evidence.

---

## 11. What Is the Smallest Architecture Change Needed?

### Recommended: `app/evaluation/answer_quality.py`

A deterministic answer quality evaluator that operates on the `OrchestrationResult` and produces structured evaluation results. This module should:

1. **Decompose the answer into claims** using sentence splitting + compound-sentence decomposition (split on "and", "but", semicolons for factual claims)

2. **Map each claim to its citations** by finding bracket markers in or near the claim sentence

3. **Resolve citations to evidence chunks** using the `OrchestrationCitation` list

4. **Evaluate claim support** using:
   - Structural: does the claim have a citation?
   - Numerical: does the claim contain numbers? Do the cited chunks contain matching numbers?
   - Lexical: does the claim share key terms with the cited chunk?
   - Temporal: does the claim contain dates? Do the cited chunks contain matching dates?

5. **Compute metrics**:
   - Claim support rate (fraction of claims with at least partial evidence support)
   - Unsupported claim rate
   - Citation precision (fraction of citations that are topically relevant)
   - Numerical consistency rate
   - Citation presence rate

6. **Optionally compare against gold facts** when available

This module should:
- Be deterministic (no LLM calls in production)
- Operate on `OrchestrationResult` (no changes to the orchestration pipeline)
- Be usable both as a benchmark tool and as a production quality signal
- Be disabled by default in production (opt-in via settings)

### What This Does NOT Require

- No changes to the orchestration pipeline
- No changes to the synthesis prompt
- No changes to the retrieval system
- No new LLM calls in production
- No changes to existing benchmark infrastructure
- No changes to the verification engine

---

## 12. Dataset Assessment

### 38-Query Eval Plan (`eval_plan_v1.json`)
- **Has gold_facts**: Yes, per query (e.g., `["New York City"]`)
- **Has gold_answer**: No (only in `questions_v1.json`)
- **Has supporting_docs**: Yes (e.g., `["doc-a"]`)
- **Has query classes**: 10 categories including conflict, numerical, multi-hop
- **Suitability for answer evaluation**: Good for fact-coverage evaluation. `gold_facts` can be checked against answer text.

### 110-Question Dataset (`questions_v1.json`)
- **Has gold_answer**: Yes, full text answers
- **Has gold_evidence**: Yes, source passages
- **Has gold_years**: Yes, for temporal questions
- **Has expect_contradiction**: Yes, for contradiction questions
- **Suitability for answer evaluation**: Excellent. Full gold answers enable token F1 and semantic comparison.

### 34-Query Stress Test (`stress_test_plan_v1.json`)
- **Has gold_facts**: Yes
- **Has gold_answer**: No
- **Has query classes**: 8 categories
- **Suitability**: Good for fact-coverage evaluation.

**Recommendation:** Use `questions_v1.json` (110 items) as the primary answer-quality benchmark since it has full gold answers. Use `eval_plan_v1.json` (38 items) for the core eval with fact-coverage metrics. Use stress test as supplementary.
