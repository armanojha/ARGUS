# PHASE 06.6 — Real-World Performance & Quality Baseline Report

**Phase:** 06.6
**Status:** EVALUATION-ONLY (no production code changed)
**HEAD tested:** `01298541c756625c30269b45ebd70a5764e552ae`
**Full-suite baseline:** 436 passed / 20 skipped
**Date:** 2026-09-03
**Runtime:** Windows PowerShell 5.1, Python 3.14 (cp1252 stdout; eval scripts forced UTF-8)

> Scope, per phase spec: "So far this is a *genuinely advanced practical Agentic RAG
> system*, but … we need real-world performance and quality evaluation." This phase
> **measures, does not improve**. All findings are recorded; no optimization code was
> written. Phase 07 is **not** started.

---

## 1. Executive Summary

A controlled 10-query / 10-document evaluation was run in **two runtime modes:**
1. **Default** (single-provider `LLMRouter` → groq only).
2. **MultiModelRouter** (groq primary → gemini → zen fallback chain).

**The decisive, honest finding:** the shared live **groq free-tier account's DAILY token
budget (200 000 TPD) was exhausted** mid-evaluation. The **Default runtime degraded to
raw-evidence recall for 9 of 10 queries** (`degrade=True`) because it **has no
cross-provider fallback** — synthesis failed and ARGUS safely returned top retrieved
evidence ("Synthesis was unavailable; returning the top retrieved evidence instead").
**No hallucination and no crash occurred.**

The **MultiModelRouter, given the same exhausted-groq condition, fell back to gemini and
produced genuine, cited, synthesized answers for ALL 10 queries (`degrade=False`)** —
including correct handling of a cross-document information conflict and an honest
"no information found" for an absent-topic query.

This single contrast is the phase's core result: **ARGUS's orchestration is genuinely
advanced, but its Default (single-provider) deployment is fragile under quota pressure,
while its MultiModelRouter deployment converts that provider pressure into graceful,
correct cross-provider failover.**

### Headline numbers (avg / median / p95)

| Metric | Default | MultiModelRouter |
|---|---|---|
| Synthesized answers (no degrade) | **1 / 10** | **10 / 10** |
| Latency avg | 15.13 s | 15.45 s |
| Latency median | 16.38 s | 16.04 s |
| Latency p95 | 17.21 s | 21.81 s |
| LLM calls / query (median) | 4 | 7 |
| Retrieval recall (cite-based) | 0.89 | **0.95** |
| Retrieval precision (cite-based) | 0.56 | **0.73** |
| Answer quality (avg, rubric /20) | n/a degraded | **19.3 / 20** |

---

## 2. Methodology

### 2.1 Controlled corpus
Ten real text documents (chunked, `SentenceChunker`, BM25 + FAISS vector hybrid retriever,
`NoOpReranker`) constructed to exercise distinct retrieval challenges:
`refund_policy`, `refund_exceptions`, `shipping_policy`, `warranty_policy`,
`member_benefits`, `privacy_doc`, `retention_doc`, `security_standard`, `serving_slo`,
`support_hours`.

Corpus is engineered with:
- **Simple implied facts** (refund window).
- **Exceptions / special cases** (custom items non-refundable).
- **Two-document information conflict** (`privacy_doc`: 12-month retention vs
  `retention_doc`: 24-month).
- **Absent / out-of-domain topic** (office pet policy — nowhere in corpus) to test
  hallucination resistance.
- **Cross-document synthesis** (membership + shipping + warranty).

### 2.2 Query set (categories A–J)
| ID | Category | Query type |
|---|---|---|
| A_simple | simple factual | single-doc lookup |
| B_multihop | multi-hop | exception + eligibility |
| C_comparison | comparison | standard vs premium |
| D_crossdoc | cross-document | warranty across related docs |
| E_summary | summarization | shipping policy overview |
| F_extract | extraction | retention periods across docs |
| G_ambiguous | ambiguous/conflict | conflicting retention data |
| H_weak | weak/absent evidence | out-of-corpus topic |
| I_synthesis | synthetic multi-fact | premium + shipping + warranty |
| J_research | research/multi-question | security+privacy+retention+SLO |

### 2.3 Evaluation harness
`eval_corpus.py` / `run_eval.py` (throwaway, not committed) used the **real engine**
(`ORCHESTRATOR_IMPL=engine`): `run_query`, `HybridRetriever`, `NoOpReranker`, real
`configs/model_policy.yaml`, a temp evidence store, and real providers (`groq`, `gemini`,
`zen`). Eval runner injected a 12 s cooldown between queries.

Data captured to `E:\ARGUS\ARGUS\data\_phase066_mm.json` (MultiModelRouter) and
`E:\ARGUS\ARGUS\data\_phase066_default.json` (Default).

---

## 3. Answer Quality

Scored on a 4-axis rubric (accuracy / groundedness / synthesis / conflict-and-absence
handling), each 0–5, total /20. Ground-truth answers were authored alongside the corpus.

### 3.1 MultiModelRouter (verified real answers)

| Query | Grade | /20 | Highlights |
|---|---|---|---|
| A_simple | A | 19 | "full refund … within 30 days of purchase" correct; honest about non-standard window |
| B_multihop | A | 20 | "No, custom-ordered items cannot be returned for a refund [1]" — correct |
| C_comparison | A- | 18 | Tiered standard-vs-premium; caveats missing per-tier window detail |
| D_crossdoc | A | 20 | 1-year warranty on manufacturing defects + serial/proof claim steps |
| E_summary | A | 20 | Standard 3–5d / Express 1–2d / >$50 free / <$50 $6 — all correct |
| F_extract | A | 20 | Extracted 12-month vs 24-month retention conflict |
| G_ambiguous | A | 20 | Surfaced 12-vs-24 conflict, did not hallucinate a single answer |
| H_weak | A- | 19 | "there is no information regarding … pets in the office" — honest absence |
| I_synthesis | A | 19 | Combined premium benefits + shipping + warranty |
| J_research | A- | 18 | Comprehensive, inconsistent items explicitly noted |

**Average ≈ 19.3 / 20.** No fabricated facts observed; every claim carried a `【n】`
citation mapping to a real source chunk.

### 3.2 Default (confounded)
Default synthesis was rate-limited (see §6) for 9/10; the only clean measurement
(A_simple, run-1 isolated) produced a correct 30-day answer, confirming Default CAN
synthesize at low volume but is not quota-safe. Source-only (degraded) answers carried
correct raw evidence with no hallucination.

---

## 4. Retrieval Quality

Retrieval recall/precision measured by (a) whether each query's **ground-truth documents**
were present among the **final cited sources**, and (b) how many cited sources were
relevant.

| Mode | Avg recall | Avg precision |
|---|---|---|
| MultiModelRouter | **0.95** | **0.73** |
| Default | 0.89 | 0.56 |

Notes:
- **MultiModelRouter** recalled the ground-truth doc(s) in every query except
  B_multihop (0.50, cited only `refund_exceptions`; `refund_policy` retrieved but not
  cited). Precision is dragged down by incidental co-citation of `support_hours`,
  `retention_doc`, etc.
- **Default** also retrieved ground-truth docs well (recall 0.89) but cited more noise
  (precision 0.56) — partly because degraded source-only answers surfaced broader raw
  evidence without synthesis-level pruning.
- Decomposition visibly helped: I_synthesis and J_research (MultiModelRouter) each
  issued a planning chain and cited the widest relevant set (4 relevant sources each).

---

## 5. Adaptive Behavior (Phase 06 dynamic orchestration)

Observed `question_pattern` classification and dynamic policy selection:
- **B–J** took the **full path** (query_analysis → research_planning → evidence_extraction
  → synthesis) = 4+ calls.
- **A_simple** took the **fast path** (1 call, direct synthesis) = correct adaptive
  gating.
- **MultiModelRouter** engaged **multiple retrieval iterations** on complex queries:
  C (2 iters), I (2 iters), J (3 iters), H (2 iters), stopping via
  `NO_UNRESOLVED_CONTRADICTION` or `BUDGET_EXHAUSTED` — i.e., the engine actively
  decomposed sub-questions and re-retrieved until satisfied or the call ceiling was hit.
- **Conflict and absence** were handled at the synthesis layer (G: conflict surfaced;
  H: honest no-evidence answer), demonstrating Phase 06's anti-hallucination behaviour.

--- 

## 6. Performance & Efficiency

### 6.1 Latency (s)

| Mode | avg | median | p95 | min | max |
|---|---|---|---|---|---|
| Default | 15.13 | 16.38 | 17.21 | 4.68 | 17.27 |
| MultiModelRouter | 15.45 | 16.04 | 21.81 | 2.18 | 23.89 |

- **A_simple** fast path: 2.18 s (MM) / 4.68 s (Default).
- The ~15 s average is **dominated by provider timeout/429 retry-with-backoff waits**
  (each failed groq call burned 3.5–4.5 s of backoff) — not by the engine itself.
- Latency is **provider-bound**, not engine-bound. With healthy providers, per-call
  model latency was 0.7–2.5 s (gemini) and 1.2–4 s (groq).

### 6.2 LLM calls & tokens

| Mode | calls/query (median) | avg calls | avg prompt tok/call | avg comp tok/call |
|---|---|---|---|---|
| Default | 4 | 3.70 | ~0 (all calls failed) | ~0 |
| MultiModelRouter | **7** | 6.50 | 1 883 | 533 |

- Default's 4-call orchestration, while cheaper, produced **no synthesis** under quota
  exhaustion (calls failed before tokens were returned).
- MultiModelRouter's higher call count (6–9) reflects **decomposition + iterative
  retrieval + per-call failure→fallback retries**. Each successful call cost
  ~1.9 k prompt / ~0.5 k completion tokens — significant, because every synthesis step
  re-sends the evidence block.

### 6.3 Efficiency verdict
The 4-call orchestration with large per-step prompts is **token-hungry against a
free-tier 8000 TPM / 200 k TPD budget**. This is the root cause of the Default degrade
(see §7). Explains why the daily budget vanished after a few complex queries.

---

## 7. Reliability & Robustness

### 7.1 Deterministic failure injection (no live dependency)
| Case | Injection | Result |
|---|---|---|
| R1 provider unavailable | `ProviderUnavailableError` | **No crash**; graceful fallback answer (3 cites, length 369) |
| R2 provider timeout | `TimeoutOrNetworkError` | **No crash**; graceful fallback answer |
| R3 malformed/empty synthesis | empty response | **No crash**; graceful fallback answer |
| R5 conflicting evidence | privacy_doc(12) + retention_doc(24) retrieval | Both source docs retrieved → conflict material surfaced correctly |

### 7.2 Live sustained-load degradation (observed, real)
Under the exhausted groq daily budget, the **Default** runtime:
- Returned `degrade=True` for **9/10** queries.
- Every such query ended with the safe fallback message and raw retrieved evidence.
- **No hallucination, no crash, no empty response** — degradation is fail-safe.

The **MultiModelRouter**, same condition, **absorbed the groq 429 via per-call fallback to
gemini** and returned `degrade=False` on all 10.

### 7.3 Reliability summary
- **Safety is strong:** even total provider failure fails safe into evidence-based recall.
- **Default availability is weak under quota:** single-provider, no failover ⇒ degrades.
- **MultiModelRouter availability is strong:** cross-provider fallback converted provider
  failure into correct answered queries.

---

## 8. Default vs MultiModelRouter

| Dimension | Default (groq-only) | MultiModelRouter |
|---|---|---|
| Synthesized answers (10-query stress) | 1 / 10 | **10 / 10** |
| Provider failover | None | groq→gemini→zen ✔ |
| Retrieval recall | 0.89 | 0.95 |
| Retrieval precision | 0.56 | 0.73 |
| Latency avg / med / p95 | 15.1 / 16.4 / 17.2 | 15.5 / 16.0 / 21.8 |
| Calls / query (med) | 4 | 7 |
| Cost per query | lower | higher (more calls + fallbacks) |
| Robustness to provider failure | Degrades (safe) | Fails over (correct) |

**Interpretation:** MultiModelRouter is the production-worthy runtime. Default is
adequate at low volume but not quota-safe for sustained real-world usage.

---

## 9. Documented Findings (severity-graded)

> No changes made. These are recorded for Phase 07 / HARDEN planning.

### BLOCKER
- **B-1. Default runtime has no cross-provider fallback.**
  When the sole provider (groq) rate-limits (429 / daily TPD cap), synthesis degrades to
  raw-evidence recall for the entire run. Continuous (non-interactive) users see missing
  synthesis. *Fix direction:* route Default through `MultiModelRouter` (already built).

### HIGH
- **H-1. Orchestration token burn vs free-tier quota.**
  4-call orchestration + large per-step evidence prompts consumed the 200 k TPD budget
  after a handful of complex queries. Leads directly to B-1 in default mode and raises
  cost in MM mode (gemini absorbs most traffic under pressure).
- **H-2. Latency dominated by provider backoff.**
  ~15 s average is mostly 429/connect-timeout backoff (3.5–4.5 s per failed call), not
  engine work. Under degraded providers, p95 grows (MM p95 21.8 s).

### MEDIUM
- **M-1. Chung-size evidence blocks re-sent per call.**
  Every synthesis/extraction/planning step re-includes the full evidence block
  (~1.9 k prompt tokens/call). Token efficiency suffers; pruning/summarising evidence
  context would cut cost.
- **M-2. Noisy co-citation lowers precision (0.56–0.73).**
  `support_hours`, `retention_doc`, etc. are frequently co-cited, inflating citation
  sources beyond the strictly relevant set.

### LOW
- **L-1. zen provider is high-latency and low-compliant.**
  Returns verbose refusals (e.g., does not follow "reply exactly: ok"), ~11 s/call.
  Correctly demoted to last-resort fallback by the Phase 06.5 policy fix.

### Empirical positives worth preserving
- No hallucination across all modes; honest absence (H) and conflict surfacing (G).
- Correct sub-question decomposition and iterative retrieval on complex queries.
- Fail-safe degradation in every failure injection (R1–R3).

---

## 10. Final Verdict

Separated scores (0–10) to keep honest:

| Dimension | Score | Basis |
|---|---|---|
| Answer quality / accuracy | **9.0** | 19.3/20 rubric; all-correct synthesis (MM) |
| Retrieval (recall/precision) | **8.0** | recall 0.95, precision 0.73 (cite-based) |
| Adaptive orchestration | **8.5** | decomposition, iterative retrieval, pattern gating work |
| Performance (latency/calls/tokens) | **6.0** | provider-bound latency, token-hungry orchestration |
| Reliability / availability | **7.5** | fail-safe + failover, but Default degrades under quota |
| Practical real-world readiness | **6.5** | great capability, but Default deployment not quota-safe |

**Composite ≈ 7.6 / 10.**

### "Is ARGUS currently a genuinely advanced practical Agentic RAG system?"

**Honest answer: The engine and orchestration are — yes. The Default deployment, as
frequently run — not yet, under sustained real-world load.**

- The **MultiModelRouter** deployment behaved like a genuinely advanced practical Agentic
  RAG system: it decomposed complex queries, iteratively retrieved, correctly handled
  ambiguity, conflict, and absent evidence, cited every claim, never hallucinated, and
  converted a provider failure into a correct cross-provider answer — 10/10 synthesized
  answers under the very condition that broke the Default mode.
- The **Default** mode delivered correct, safe answers but **could not synthesize** once
  its sole provider's quota was exhausted. That is a practical, deployment-level gap, not
  an intelligence gap.

**Therefore: ARGUS is a genuinely advanced Agentic RAG system problematically deployed by
default.** The intelligence, orchestration, retrieval, and safety are all real and
high-quality. The practical-availability weakness is concentrated in the single-provider
Default configuration — which, per this evidence, should be superseded by the
MultiModelRouter at runtime. Phase 07 should therefore focus not on raw capability
(which is demonstrably strong) but on **quota-safe default deployment, token efficiency,
and latency**.

---

## 11. Artifacts
- `data/_phase066_mm.json` — MultiModelRouter results (10 queries, decisions, answers).
- `data/_phase066_default.json` — Default results (10 queries, degrade flags).
- `data/_phase066_results_default.json` — Default run-1 (isolated A_simple baseline).
- Raw per-call routing/telemetry embedded in the JSON files (latency, provider, model,
  fallback, tokens, error codes).

## 12. Next steps (NOT started — parked for Phase 07)
1. Recommend Default runtime → `MultiModelRouter` (single-line config) to close B-1.
2. Introduce evidence-context pruning / summarization to cut per-call tokens (H-1/M-1).
3. Tighten citation pruning to raise precision (M-2).
4. Re-evaluate against a healthy provider budget to confirm the uplift and fresh numbers.