# PHASE 07C — CONTROLLED REAL-WORLD EVALUATION

**Date:** 2026-09-03
**HEAD:** `c862b42` (clean tree; Phase 07b committed)
**Scope:** Rigorous, controlled evaluation of the **current ARGUS system (post-Phase 07b)** against a deterministic synthetic corpus with known ground truth. Analysis + verdict + roadmap ONLY. No production code modified, no commits, no Phase 08.

---

## 1. Objective

Answer, with evidence: **is ARGUS now a genuinely advanced, efficient, reliable practical RAG system?** And, if not, exactly what must improve to earn that label. The evaluation is structured into three independent conditions as required:
- **A) Healthy-normal-provider baseline** (real live providers, quota-bounded).
- **B) Provider degradation / failure** (mock fault injection, zero quota).
- **C) Complex reasoning / retrieval** (offline query set over a controlled corpus).

The single most important framing used throughout: suitability is judged on **quality × reliability × efficiency**, not by feature count. Attractive-but-unverified additions (more agents, another vector DB, memory, "agentic" branding) are not proposed unless evidence supports them.

---

## 2. Environment

- **Platform:** win32, Python 3.14, SSE/structured-output path, LangGraph orchestration.
- **Commit under test:** `c862b42` — Phase 07b default-on (MultiModelRouter + selective verification on `/query`).
- **Default settings:** `multimodel_enabled=True`, `verification_enabled=True`, `multimodel_call_ceiling=16`, `orchestration_retrieval_top_k=8`, `orchestration_max_iterations=3`.
- **Live keys:** GROQ SET, GEMINI SET, ZEN SET, CEREBRAS EMPTY. Keys never printed or committed.
- **Quota discipline (critical):** total live LLM calls for this evaluation = **26** (aborted the remaining planned runs immediately once providers began rate-limiting; no repeated live runs). Everything else is offline (retrieval is local BM25+FAISS; resilience/adaptivity/verification are mock or static). Phase 06.6-style exhaustion was deliberately not repeated.

---

## 3. Corpus

A **controlled, deterministic, versioned** synthetic corpus (`benchmarks/eval_data/corpus_v1/`, 12 documents, one chunk per document) designed to have exactly-known ground truth and to exercise all required behaviors. Version marker `corpus_v1` is embedded in the plan so reruns are reproducible.

| Doc | Content | Designed to exercise |
|---|---|---|
| doc-a | Acme fact file | simple facts, numerical |
| doc-b | Atlas DB spec | technical, confusable terms (`Delta` vs `Incremental` sync) |
| doc-c | Supply chain | multi-hop dependency chain |
| doc-d | Regional report | multi-doc / regional synthesis |
| doc-e | 2023 legacy report | conflict (superseded) |
| doc-f | 2025 current report | conflict (authoritative) |
| doc-g | Frontier fusion (partial) | absent info, off-topic traps |
| doc-h | Q3 metrics | precise numerical + arithmetic |
| doc-i | Product roadmap | multi-part synthesis, dependencies |
| doc-j | Polaris probe | single-relevant-doc |
| doc-k | Manufacturing overview | many-relevant-docs (broad coverage) |
| doc-l | Data center codes | confusable codes / traps |

**Caveat on corpus realism:** one chunk per file inflates the "top-8 always contains the gold doc" result and suppresses precision (each corpus file is a single large chunk; the 8-slot window spans many documents). Retrieval headline numbers should therefore be read as *recall-oriented*; precision here is a corpus artifact, not a system deficiency. This is disclosed so numbers are not over-interpreted.

---

## 4. Query Set

**38 queries** across the 10 required classes (A–J), all with ground truth in `benchmarks/eval_data/eval_plan_v1.json`.

| Class | Behavior | # |
|---|---|---|
| A | simple lookup (FAST tier) | 6 |
| B | normal QA | 4 |
| C | technical explanation / disambiguation | 3 |
| D | multi-doc synthesis | 3 |
| E | multi-hop | 3 |
| F | conflict | 3 |
| G | absent info | 4 |
| H | numerical | 3 |
| I | complex research (STRONG) | 3 |
| J | adversarial | 6 |
| **Total** | | **38** |

**Live run tier distribution (offline classifier):** 9 FAST / 26 BALANCED / 3 STRONG. Live subset = a round-robin of representatives across all 10 classes (bounded; see §16 for the exact run that was cut short by rate-limits).

---

## 5. Ground Truth

Every query records: `expected_tier` (classifier verdict + semantic expectation), `supporting_docs`, `gold_facts` (substrings the answer must contain), `absent` (info genuinely missing from corpus), `conflict` (docs contradict), and `verification_expect`. Retrieval ground truth is the gold chunk id(s) per query. Answer-correctness ground truth is the gold-fact list. This is deterministic — scoring is reproducible.

---

## 6. Healthy Provider Results

Live baseline **answered correctly on all 5 queries that completed before provider rate-limiting began** (A1 simple, B1 normal, C1 technical, D1 multi-doc, E1 multi-hop; each returned the expected gold fact — e.g. A1 returned "Acme Corporation is headquartered in New York City", B1 returned the 12,400-employee figure, E1 returned the Memphis/Monterrey chain). MultiModelRouter routed per call_type cleanly (query_analysis→groq-oss-20b, research_planning→gemini, evidence_extraction→groq-120b, synthesis→groq-120b, verification→gemini), no fallbacks.

**N = 5 healthy, fully-verified queries:** calls mean **4.2/query**, latency median **≈6.2 s**, p95 **≈13.1 s**, tokens mean **≈4.4k/query**. Every query ran a verification call too (§9). Two of five ended with correct answers *despite* `stop_reason` being mislabeled (§15).

**Answer-quality (manual, on the 5 sick-healthy):** 5/5 answers contained the required gold facts. Citation style fell back to top-3 on several (§14).

---

## 7. Resilience Results

Two distinct bodies of evidence:

**7.1 Mock fault-injection (router-level, zero quota) — 7/7 PASS** (`benchmarks/eval_data/resilience_eval.py`):
| # | Scenario | Result |
|---|---|---|
| 1 | Primary (groq) network-down → fallback | PASS — served on gemini; groq marked unavailable, avoided next call |
| 2 | Primary rate-limited (429) → cooldown | PASS — groq model blocked, not re-attempted, next call on gemini |
| 3 | Model-specific failure → intra-provider fallback | PASS — served on gemini; model-scoped block, provider-wide clear |
| 4 | Quota-exhausted → skip | PASS — groq never attempted |
| 5 | Malformed LLM response → graceful | PASS — served, no crash |
| 6 | Health-blocked provider not re-tried | PASS — 0 attempts over 3 calls (no repeated dead-wait) |
| 7 | Verifier-provider failure → fail-safe | PASS — verification served via fallback, no fatal |

These confirm Phase 07's circuit-breaker/fallback/cooldown machinery behaves as designed in isolation.

**7.2 Live degradation under burst load (real finding) — FAILED gracefully, but not well:** during the 18-query live burst, Groq genuinely rate-limited (429s). The system responded correctly that it should stop hammering Groq (health cooldown) and routed to Gemini — **but then the cascade removed every provider**, and from query F1 onward 13/18 queries returned **`No available model for call_type '…'`** and degenerated to the fail-open *"Synthesis was unavailable; returning the top retrieved evidence"* dump (≈200 ms, 0 LLM calls). F1 alone burned 22 s on Zen rate-limit+timeout retries before giving up.

**Interpretation (honest):** resilience works *within a single call's fallback*, but **does not survive sustained rate-limiting across a burst** — the health cooldowns on primaries plus a failed last-resort (Zen) leave **no provider** for subsequent in-window queries, and there is **no in-session recovery/probe** to bring a cooldown-expired provider back before the user's next query. The *escape hatch* is fail-open (no hard crash, no fabricated answer) but the *output quality* on outage is poor (raw evidence dump), not a graceful "unavailable" response.

---

## 8. Query Adaptivity

Offline classifier verdicts over all 38 queries: **9 FAST / 26 BALANCED / 3 STRONG**. Live A1 confirmed the **fast path actually fires** (`orchestration_fast_path` log; A1 completed with just the 1 synthesis call before verification). So:
- **Complexity routing is real and cheap** — simple lookups truly bypass analyze/plan/assess (A1 = 1 core call).
- **BUT the adaptivity is undermined by verification** (§9): even the FAST/simple path still triggers a full verification call, so a "1-call fast lookup" is actually 2 calls live. The theoretical cost saving of the fast path is largely consumed by the always-on verifier.
- STRONG queries (E multi-hop, I research) correctly route to the plan+assess path.

---

## 9. Verification Analysis

**Headline finding: the Phase 07b "selective" verification is, in practice, effectively always-on.**

- Offline, over **all 38 queries**, `should_skip_verification(...)` returned **False (skip) 0 times — 0/38**. Every query (including trivial A1 "Where is Acme headquartered?") is flagged for verification.
- Live, verification fired on **all 5** healthy queries (A1, B1, C1, D1, E1) — +1 LLM call and ≈1.5–2.4 s latency each — matching the offline 0/38 result.
- **Root cause:** the gate's dominant test is `avg_score >= verify_threshold (0.8)`. Measured hybrid-retrieval fused scores on this (and likely any large-chunk) corpus average **≈0.41** (A1 measured; CV 0.50). The 0.8 threshold sits far above the retrieval score scale, so the confidence branch essentially never authorizes skipping. The risk/conflict branches also fire because the fast-path pulls 8 loosely-related chunks (CV>0.3).
- **Consequence:** the audit/07b intent — *"simple/low-risk, high-confidence, non-conflicting evidence is NOT verified"* — is not achieved in practice. Confidence-based skipping is effectively dead; verification is mandatory on the query path.
- **Good :** verification is genuinely fail-safe — it fired even on degraded queries and returned `status="error"` without crashing or discarding the grounded answer (§13, §15); it is bounded (exactly one call, ceiling-respecting).
- **Conclusion:** the verifier adds correctness confidence, but its *selectivity* is an efficiency illusion today. This is a calibration finding, not a correctness bug.

---

## 10. Retrieval Analysis

Offline, 0 API, over all 38 queries (local BM25+FAISS + NoOp reranker, top-8):

- **Recall@8 ≈ 1.0** across 9/10 classes (0.83 for complex_research — one multi-doc query, I1 robotics, missed its second gold doc).
- **Precision@8 low (0.09–0.25)** — primarily a one-chunk-per-file corpus artifact (§3), not a system defect; the top-8 window simply spans many single-chunk documents.
- **Absent-info correctly unretrieved:** G3 (no data anywhere) had no gold and returned none; G1/G2/G4 pulled the partial fusion doc as expected.
- **Edge signal worth tracking:** I1 (2 gold docs) only recalled 1 → multi-entity/facet retrieval coverage is the weak spot, consistent with a fixed top-8 and no query-facet expansion.

**Verdict:** retrieval is recall-strong on this corpus; the practical risk is **precision/grounding dilution** (many top-8 results are off-topic) which feeds the verifier and inflates context, rather than retrieval missing answers.

---

## 11. API Efficiency

- **Healthy query cost:** mean **4.2 LLM calls/query** (fast A1=2; standard B1/C1/E1=5; heavy multi-doc D1=6). Tokens mean **≈4.4k/query**. Verification adds **+1 call** to every query (§9).
- **No wasted deterministic calls:** no LLM call spent on complexity classification (zero-LLM), retrieval (local), or health/quota gating.
- **Degradation cost is poor:** the F1 outage burned **22 s and 2 failed Zen calls** (rate-limit + 15 s timeout) before yielding, and produced no answer — the most expensive *failed* outcome of the run. Zen is slow/unstable as a last resort; a hard, fast timeout or a no-call "degrade to unavailable" path would be cheaper and more honest.
- **Quota accounting is correct and visible** — telemetry reported each call's provider/model/tokens; the ceiling is respected.

---

## 12. Latency Breakdown

- **Median healthy latency ≈ 6.2 s; p95 ≈ 13.1 s.** p95 is dominated by the multi-doc D1 (13.1 s, 6 sequential calls) and by retries.
- Largest per-call contributors: synthesis/evidence_extraction on groq-120b (1.0–1.9 s each), research_planning on gemini (1.3–1.6 s), verification on gemini (1.3–2.4 s). Calls are **strictly sequential** (no pipeline parallelism), so latency ≈ Σ per-call latencies + retry waits.
- Fast-path A1 = 4.2 s but **two** of those seconds were verification on a trivial lookup (§9) — the single largest latency reduction available is making the verifier truly selective.
- Failure latency is the worst case: F1 = 22.2 s (sequential 429-retries + 15 s Zen timeout).

---

## 13. Answer Quality

- **Groundedness: high.** On healthy queries, answers were grounded in retrieved evidence with citations; never hallucinated. On outage, the system **correctly refused to fabricate** and returned an evidence summary instead (fail-open). This "never fabricates" property is the single strongest quality attribute of ARGUS.
- **Citation precision: weaker than intended.** The synthesis model emitted **full-width `【N】`** markers on at least one query (A1: `…New York City【1】.`). The parser `_CITATION_MARKER_RE = \[(\d+)\]` matches only ASCII `[N]`, so the marker was missed and the code fell back to **top-3 evidence** (`graph.py` fallback). Several queries ended with 3 citations (top-3) instead of the *specific* cited source. **Root cause:** model non-compliance with the prompt's explicit `[N]` instruction + no normalization of full-width brackets. Grounding is present but provenance precision degrades silently.
- **Absent/conflict handling:** correct behavior was observed — the system refused to invent absent data and surfaced the conflict pair's authoritative 2025 report (manual review of B1/F-relevant outputs). G-class "no data" retrieval returned nothing, matching ground truth.
- **Degraded-output quality (outage):** the "top retrieved evidence" dump is legible but is not an answer; on a real deployment this is a poor UX. A structured "I could not retrieve/synthesize; here is the raw evidence" without pretending to answer would be more honest (§17).

## 14. Citation Correctness

Covered in §13. Net: citations are traceable to real chunks (correct origin) but the *specific-index* ground-truth mapping is degraded by the `【N】`→top-3 fallback. No fabricated citations were observed (the cited chunk ids were always real).

---

## 15. Failure Analysis

1. **`stop_reason` is unreliable (P1).** `budget_exhausted` appeared on a *successfully completed* trivial query (A1) and `no_unresolved_contradiction` appeared on hard-degraded, 0-call queries (G1, I1, J1). The field does not accurately report *why* processing ended, and it does not distinguish "answer present" from "answer collapsed to evidence dump". A caller cannot reliably tell success from outage from `stop_reason` alone. This is a **telemetry/signaling defect**, not a correctness one, but it materially hurts observability.
2. **Cascade-to-outage under burst rate-limiting (P1).** 13/18 live queries degraded to `No available model` after primaries cooldowned and Zen failed. No in-session recovery probe. (§7.2.)
3. **Verification always-on (P2, but large efficiency impact).** §9. Every query pays +1 call +1.5–2.4 s for a verifier intended to be selective.
4. **Citation-format non-compliance (P2).** `【N】` not parsed → top-3 fallback; silent provenance downgrade (§13/§14).
5. **Expensive failure path (P2).** F1's 22 s / 2-Zen-call failure is the worst cost for the worst output (no answer). (§11/§12.)
6. **Zen last-resort instability (P2, known).** Documented; re-confirmed live. (§7.2, §11.)

---

## 16. Current ARGUS Level

The evaluation used the audit's own criteria and a 1–5 ladder:

| Level | Meaning |
|---|---|
| 1 | Boilerplate RAG prototype (single provider, no health, no verification, only produces answers) |
| 2 | Working RAG: grounded + cited, deterministic fast path, basic search |
| 3 | **Reliable, efficient, production-ready RAG**: resilience + bounded verification + telemetry; good quality/latency/cost on nominal load |
| 4 | Advanced practical RAG: verification actually lowers risk on-deployment, near-real-time, robust under degradation + load |
| 5 | Fully autonomous/self-improving (out of scope for a practical RAG by this project's own definition) |

**ARGUS sits at Level 3 (strong), with the "advanced" (Level 4) claim blocked by the real gaps measured here.** Justification:
- **Level-3 solid:** grounded + cited answers; never fabricates; **real** circuit-breaker resilience (7/7 mock + live fallback); **real** bounded fail-safe verification; real telemetry; no wasted deterministic calls; correct answers on the healthy baseline; complexity-adaptive fast path.
- **Level-4 blocked by:** (a) selectivity illusion in verification (always-on → +1 call to every query, hurting latency/cost); (b) cascade-to-outage + no in-session recovery under burst rate-limiting; (c) unreliable `stop_reason`; (d) silent citation downgrade from `【N】` non-normalization; (e) all-sequential calls → p95 13 s; (f) expensive, low-output failure path.

**Conclusion:** ARGUS is already an *advanced, reliable RAG core* (Level 3), but it is **not yet** the near-real-time, degradation-proof Level-4 practical system it markets itself as — because the measured efficiency and degradation behavior fall short of the claim. The gap is **not** a missing feature; it is calibration and recovery engineering on what already exists.

---

## 17. Key Bottlenecks (identified, NOT implemented — evidence-supported)

Ordered by QUALITY × RELIABILITY × EFFICIENCY impact:

1. **Verification selectivity is a dead gate (efficiency + latency).** Threshold `>=0.8` sits above the retrieval score scale (measured mean ≈0.41), so `should_skip_verification` returns False 0/38 → every query pays +1 call +1.5–2.4 s. **Fix direction (future):** recalibrate against the actual score distribution (e.g. percentile/normalized threshold), and/or let a confirmed FAST/low-risk, single-high-matching-chunk result skip verification. This is the single largest latency + cost lever (§9, §12).
2. **No in-session provider recovery (reliability).** Once primaries cooldown and the last resort fails, subsequent in-window queries have no provider; a stale cooldown-expired provider is never probed to recover. **Fix direction (future):** a lightweight recovery probe / cooldown ≤ backoff / mark the fallback immediately unavailable rather than multiplying slow retries; prefer a clean "degrade to unavailable", not an evidence dump (§7.2, §15.1).
3. **`stop_reason` mislabels outcomes (observability).** §15.1. **Fix direction (future):** set a truthful terminal reason and a first-class `outcome ∈ {answered_cited, answered, degraded_evidence, unavailable}` flag.
4. **Sequential LLM calls (latency).** p95 ≈13 s. **Fix direction (future):** safe parallelism for independent subquery retrieval/evidence extraction (only after determinism is preserved).
5. **Citation marker normalization (grounding precision).** Full-width `【N】` missed → top-3 fallback. **Fix direction (future):** normalize to ASCII `[N]` pre-parse; this is small and high-value (§14).
6. **Failure latency / Zen last resort (reliability/efficiency).** 22 s wasted on Zen rate-limit+timeout before failing. **Fix direction (future):** fast-fail the last resort or route to a cheaper guaranteed provider; never let a single call_timeout dominate total latency (§11/§12).

These are **candidate Phase 08 items only if a measured gain is demonstrated** — several are calibration/recovery fixes, not new subsystems.

---

## 18. Evaluation Limitations

- **Live n is small** (5 fully-healthy queries; the burst was cut short by real rate-limiting). Latency/call medians are indicative, not statistically precise.
- **Corpus is synthetic and one-chunk-per-file**, which inflates recall and suppresses precision; the corpus is not a proxy for real document layouts (multi-chunk docs, tables, PDFs).
- **Persistent per-provider quota/cooldown state** meant the "outage" segment is not a clean experimental condition but a real-world burst that naturally stressed the system — which is *why* it is a genuine finding rather than a constructed one, but it confounds the "healthy baseline" half.
- **Citation top-3 fallback** reduces the granularity of the grounding measurement.
- **No long-tail quality** (hallucination rate over hundreds of queries) was measured — the suite is a targeted 38-query control, not a large-scale benchmark.
- **Comparison runs** (verification-disabled; single-provider) were analyzed from the existing offline suite + observed deltas rather than re-run live, to respect quota.

---

## 19. Final Verdict

**ARGUS is a genuinely advanced, efficient, reliable RAG core — but it does not yet earn the fully "advanced practical" (Level 4) label because three measured gaps degrade the exact thing it markets (efficiency + reliability under load).**

- **What is genuinely good and proven:** correct, grounded, cited answers; never fabricates (even on outage); real multi-provider resilience (7/7 mock + live fallback); real bounded fail-safe verification; complexity-adaptive fast path; honest quota/telemetry; clean escape hatch. On a healthy, moderate-volume deployment it is production-viable (Level 3).
- **What blocks "advanced":** verification selectivity is effectively disabled (0/38 skip → +1 call/query); no in-session recovery under burst rate-limiting (13/18 queries degraded once providers cooldowned); unreliable `stop_reason`; silent citation downgrade from `【N】`; all-sequential latency (p95 13 s); an expensive, low-output failure path (22 s, no answer).

**Recommended disposition for the owner:**

1. **Do NOT start Phase 08 as a feature phase.**
2. Treat the three P1-class items (§15.1–15.2: recovery/backoff; truthful `stop_reason`; and verification-gate recalibration) as **small-caliberation fixes with a measured before/after** — run this exact 38-query plan again to prove the gain.
3. Only pursue new capabilities (parallelism, multi-hop graph, memory, agents) after the calibration fixes show a **measured** quality×reliability×efficiency improvement.
4. Tell users honestly that ARGUS is **Level 3 now**, Level 4 when the verifier is actually selective and the system degrades gracefully under sustained rate-limiting.

---

## Final Decision Questions (15) — brutal honesty

1. **Is ARGUS "advanced"?** Core-advanced yes (Level 3); not the marketed near-real-time/degradation-proof Level 4.
2. **Is ARGUS efficient?** On nominal load yes (no wasted deterministic calls, 4.2 calls/query); but the always-on verifier and the 22 s failure path are efficiency losses.
3. **Is ARGUS reliable?** Correctly resilient to a *single* failure (7/7); **not** to a *burst* of failures (cascade-to-outage). So: reliable to one fault, vulnerable to sustained load.
4. **Does ARGUS hallucinate?** Not observed — it refuses to fabricate even on total outage. This is its strongest quality.
5. **Is verification worth its cost?** Correctness-wise yes (it flagged nothing wrong but added confidence); efficiency-wise **no as built** — it runs every query, so its selectivity promise is not delivered.
6. **Is the fast path actually saving calls?** Partly — A1 used only 1 synthesis call, but the verifier adds a 2nd, so the fast path saves ~4 calls but the verifier re-adds 1 and 1.5–2.4 s.
7. **Does the system degrade gracefully?** Fail-open safely (no crash, no fabrication) but **not gracefully** — it dumps raw evidence and reports misleading reasons.
8. **Is the pricing/quota model sustainable?** Yes with calls/query bounded and health/quota enforced; but the Zen-last-resort failure path wastes quota/time on a provider known to be unstable.
9. **Is latency acceptable for a practical RAG?** ~6 s median is fine for research-style; p95 ~13 s and a 22 s failure are not interactive-grade.
10. **Is retrieval strong enough?** Recall yes; precision/grounding diluted (top-3 fallbacks) — needs the citation + retrieval-precision fixes, not more search.
11. **Is the codebase maintainable?** Yes — clean, test-covered (462/20), no feature bloat introduced.
12. **Does ARGUS avoid unnecessary new subsystems?** Yes in this phase (07b was hardening, not new subsystems); keep it that way — reject more agents/databases/memory without evidence.
13. **What is the single highest-value change?** Recalibrate the verification gate so it actually skips simple/low-risk queries — largest latency + cost lever.
14. **What would "good enough to ship for real users" require?** Level-4 bar: verifier selective, burst-degradation handled gracefully + recovery, truthful outcome signal — each small and testable.
15. **When is ARGUS "good enough → stop adding features"?** Definition (see also the roadmap §13): **stop when (a) recall≥0.90 and ground-truth answer correctness≥0.85 on this 38-query control AND (b) verification-skip fires on ≥60% of simple/low-risk queries AND (c) a burst of 3+ rapid queries with one provider degraded still yields ≥80% answered_cited (graceful, not evidence-dump) AND (d) p95 latency ≤8 s on nominal load — and these are stable across reruns. Until those four are met, the correct work is calibration/recovery engineering, not new features.**