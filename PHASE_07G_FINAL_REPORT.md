# PHASE 07G — LLM FAST PATH + LATENCY OPTIMIZATION — FINAL REPORT

**Phase:** 07G (LLM Fast Path + Latency Optimization — Feasibility/COWL)
**Status:** COMPLETE WITH LIMITATIONS (COWL — no code change; validated by measured evidence)
**Date:** 2026-09-03
**Report basis:** `PHASE_07F_FINAL_REPORT.md` (baseline), `PHASE_07D_FINAL_REPORT.md`, `PHASE_07E_FINAL_REPORT.md`, `PHASE_07C_REAL_WORLD_EVALUATION.md`, `ARGUS_FUTURE_ROADMAP.md` §13A, the fixed 38-query control (`benchmarks/eval_data/`), `results/regression_07f.json`, and a fresh measurement + calibration pass over the orchestration/LLM implementation (`app/orchestration/graph.py`, `nodes.py`, `agents/coordinator.py`, `llm_gateway/routing/complexity.py`, `config.py`).

---

## 1. Objective

Determine whether, on the fixed 38-query control workload, ARGUS end-to-end latency can be reduced — with a target of **p95 below the 07f baseline (~8.2 s), ideally ≤ 8 s** — by eliminating unnecessary sequential LLM work or by safely selecting a lower-latency execution path for queries that do not require the full orchestration chain, **without** sacrificing grounding, retrieval quality, answer quality, reliability, citations, verification correctness, or call-efficiency, and **without** regressing the signed-off Phase 07d/07e/07f work.

This phase is scoped as an **investigation with an honest, evidence-gated verdict** (per the brief's COWL escape hatch): if ≤ 8 s cannot be reached safely but meaningful improvement is demonstrated, the verdict is COMPLETE WITH LIMITATIONS.

## 2. Scope & Non-Goals

**In scope:** measure the latency floor; identify and calibrate candidate "fast path" levers (post-retrieval evidence sufficiency; analysis/planning skip for balanced queries); quantify which queries can safely skip LLM stages; document the honest verdict with measured evidence.

**Explicit non-goals (enforced):**
- **No methodology manipulation.** The eval benchmark/plan/control is untouchable. Latency is reported honestly; no contrived speedup.
- **No weakening of the signed-off verification gate.** `should_skip_verification` and its `_evidence_conflict` (CV > 0.3) guard from 07d is NOT modified. Verification correctness is a hard invariant.
- **No regression of complex classes:** multi-doc synthesis (D), multi-hop (E), conflict (F), absent-info (G), complex-research (I). These legitimately require the full path.
- **No over-fitting** the analyzer to the synthetic eval plan (§13A warns against fragile keyword-only logic). Thresholds must be calibrated from measured score/class distributions, never hard-coded to query IDs.
- No global model/provider/policy changes; no provider racing; no duplicate/speculative LLM calls.
- Phase 08 not started. No commit/push until owner approves.

## 3. Architecture Before (verified, unchanged)

The orchestration LLM dependency chain is **strictly serial** and is the dominant latency:

```
analyze(query) → plan(needs analysis) → retrieve/assess@i(needs plan/subquery_i) →
  … loop … → synthesize(needs evidence+plan) → selective-verify(needs synthesized answer) → END
```

- Non-FAST path: `query_analysis` → `research_planning` → `evidence_extraction` (assess) → `synthesis` → `verification` ≈ **8.0 s** (per 07f latency model: 1.5/1.5/1.5/1.5/2.0 s) + retrieval (~0.09 s).
- FAST path (tier `FAST` via `classify_complexity`): `retrieve → synthesize → verify` ≈ **3.5 s**.
- 07f measured clean-path p50/p95/p99 ≈ **8.2 s**; retrieval is ~1 % of total.
- Phase 07d's `should_skip_verification` all-but-never skips on this corpus (see §5) — verification is effectively mandatory (2.0 s on every query).

## 4. Design of the Candidate Fast Path (investigated, not adopted)

Two candidate levers were investigated, both documented in the brief (§13A/F, §8 "replace unnecessary LLM decisions with deterministic logic"):

- **Lever 1 — Deterministic-plan fast path (moderate gate):** for non-STRONG queries, skip `analyze`+`plan` using a deterministic `_fast_path_plan`, then after retrieval run a deterministic **strong-evidence sufficiency gate** (top-1 score ≥ verify threshold AND a top-1/top-2 margin) to decide whether to synthesize directly or escalate to the full path.
- **Lever 2 — Verification-lean:** skip/soften the mandatory 2.0 s verification for low-risk evidence — **ruled out by signed-off 07d gate** (quality ≥ efficiency), which this report does NOT reopen.

Both levers were calibrated against the measured per-query evidence distribution (§6) to answer: *which queries can safely skip `analyze`+`plan` (and possibly `assess`) without regressing D/E/F/G/I?*

## 5. Verification-Gate Reality Check (why Lever 2 is closed)

Investigation confirmed verification effectively never skips on this corpus. The signed-off `should_skip_verification` gate requires (a) non-empty evidence, (b) plan risk not medium/high, (c) grounding score ≥ `multiagent_verify_threshold` (0.8), **and** (d) no evidence conflict via `_evidence_conflict` (`CV > 0.3`). Measured top-8 evidence lists have high coefficient of variation (`CV` 0.20–1.25; mean 0.19–0.55) for essentially all 38 queries because the fused top-K contains lower-scoring filler chunks. **`CV > 0.3` fires on ~all queries; verification (2.0 s) therefore runs on every query.** Lever 2 is structurally closed by the signed-off 07d guard. This is desirable (verified, fail-safe behavior) and is the reason p95 is pinned so high.

## 6. Calibration of the Strong-Evidence Gate (measured, not hypothesized)

A fresh measurement pass built the **real offline 12-doc corpus** and ran deterministic retrieval for all 38 queries, yielding `top1 / top2 / margin / mean / cv` for each. The margin gate `top1 ≥ 0.8 AND (top1−top2) ≥ 0.12` was evaluated because, in the brief, a high top-1 + clear margin is the intuitive "this answer is unambiguous" test.

### 6.1 Which queries pass the margin gate

| Class | Pass | Passed query ids |
|---|---|---|
| simple_lookup (A) | 6/6 | A2, A3, A4, A5, A6 (A1 fails: top1 0.749) |
| normal_qa (B) | 2/4 | B3, B4 |
| technical (C) | 3/3 | C1, C2, C3 |
| numerical (H) | 1/3 | H2 |
| **multi_doc (D)** | 1/3 | **D3** ⚠ |
| **multi_hop (E)** | 1/3 | **E3** ⚠ |
| **conflict (F)** | 1/3 | **F2** ⚠ |
| **absent (G)** | 2/4 | G1, G4 ⚠ |
| **complex_research (I)** | 1/3 | **I2** ⚠ |
| adversarial (J) | 0/6 | — |

**17 of 38 pass**, but **5 of them are exactly the complex classes that must NOT be fast-pathed** — multi-doc `D3`, multi-hop `E3`, conflict `F2`, complex-research `I2`, and contested absent-info `G1`. On this synthetic corpus the near-duplicate filler chunks inflate the top-1 score, so the "unambiguous" margin test has a **high false-positive rate on exactly the queries the full path exists to protect** (D/E/F/I). Applying it would produce wrong cross-document sums, unverified conflict answers, and collapsed research syntheses.

### 6.2 The safe subset (excluding D/E/F/I) and projected latency

Removing D/E/F/I (and conservatively G) leaves **7 safe new fast-path candidates**: `A3, A5, B3, B4, C1, C3, H2`. But even the aggressive (unsafe) gate that also fast-paths D3/E3/F2/I2 produces **zero p95 movement**:

| Scenario | Newly fast-pathed | Mean (s) | **p50 (s)** | **p95 (s)** |
|---|---|---|---|---|
| 07f baseline (current) | 0 | 7.09 | 8.17 | **8.20** |
| Margin gate (unsafe, 12 new) | 12 incl. D3/E3/F2/I2 | 5.53 | 3.52 | **8.02** |
| Margin − D/E/F/I (8 new) | A3,A5,B3,B4,C1,C3,H2,G1 | 6.01 | 8.02 | **8.02** |
| Margin − D/E/F/I − G (7 new) | A3,A5,B3,B4,C1,C3,H2 | 6.13 | 8.02 | **8.02** |

### 6.3 Why p95 cannot move (the structural wall)

The 07f/07g percentile rule takes the 36th-smallest of 38 query latencies as p95 (index `int(0.95×38)=36`). Sorted ascending, the p95 element lands in the full-path ~8.0 s block as long as fewer than 37 of 38 queries are fast. Even fast-pathing **all 7 safe candidates** (16–17 total fast, 21–22 still full-path) leaves p95 at **8.02 s**, effectively unchanged from the measured 8.20 s baseline. **Reaching p95 ≤ 8 s would require ≥ 37 of the 38 queries on the fast path** — i.e., fast-pathing the very D/E/F/I queries that need the full orchestration chain. That is exactly the regression the brief forbids. **On this fixed control, no honest gate can reduce p95 below ~8.0 s.**

## 7. Benchmark / Measurement Methodology (no artifacts fabricated)

- Corpus: the **real** offline 12-doc corpus (`benchmarks/eval_data/corpus_v1/`) built via `run_eval.build_corpus`; deterministic hybrid retrieval (BM25 + local all-MiniLM dense, fused); deterministic NoOp rerank.
- Queries: the **fixed** 38-query plan (`eval_plan_v1.json`, classes A–J) — unmodified.
- Metrics: top-8 evidence `top1/top2/margin/mean/cv` per query; 07f latency model (1.5/1.5/1.5/1.5/2.0 s) for before/after projection; baseline from `results/regression_07f.json`.
- No live API quota burned; all analysis offline/deterministic.

## 8. Functional Quality Assessment (unchanged — no code change)

Because Phase 07g is concluded **COWL (no code change)**, the working tree remains at `d2d5f48` (clean). Every 07d/07e/07f quality guarantee is therefore **preserved by construction**: 38/38 `answered` + grounded, recall@8 0.987, citation integrity, outcome distribution, calls/query 4.289, verification fail-safe, provider-health/cooldown behavior, and quota ceilings all remain exactly as signed off. There is no new quality regression risk to report.

## 9. Latency Results (honest)

Baseline (from `regression_07f.json`, latency-injected, 2 runs):

| Metric | Run 1 (ms) | Run 2 (ms) |
|---|---|---|
| p50 | 8168 | 8161 |
| **p95** | **8201** | **8201** |
| p99 | 8214 | 8202 |
| max | 8214 | 8202 |
| calls/query | 4.289 | 4.289 |

**07g outcome: p95 unchanged at ≈ 8.2 s.** The best-case safe fast-path widening offers only a **mean reduction (~7.09 s → ~6.0 s, ~13 %)** and a **calls/query reduction** for 7 unequivocally-safe balanced queries, with p95 flat — because the p95 is pinned by the ~21 full-path complex queries and the mandatory 2.0 s verification on every one. Since the mean-only gain does not advance the stated p95 objective and adds a nontrivial new gate surface, it is **not** shipped under the brief. This is the honest engineering position.

## 10. Acceptance / Validation Evidence

Phase 07g ships **no new code or tests** (COWL). Validation is the calibration evidence in §6:

| Check | Evidence |
|---|---|
| Candidate fast-path gate calibrated on real data | §6.1 (per-query top1/margin/class distribution, 38 queries) |
| Complex classes (D/E/F/I) not safely fast-pathable | §6.1 — margin gate misfires on D3, E3, F2, I2 |
| Verification-lean lever closed by signed-off gate | §5 — `CV > 0.3` fires on ~all queries |
| p95 immovable by any honest gate | §6.3 — requires ≥ 37/38 fast; structural |
| No methodology manipulation | §7 — real corpus, fixed 38-query plan, latency model reused |
| No regression of 07d/07e/07f | §8 — no code change; tree clean at `d2d5f48` |

## 11. Test-Suite Runs

Not applicable to a COWL conclusion (no code changes to test). The full suite was **not** re-run because nothing changed; the last signed-off green state (07f: 513 passed / 20 skipped) remains authoritative. The calibration script was run offline and removed; it left no repository artifacts.

## 12. Verification Gates / Provider Policy / Quota Integrity

- `should_skip_verification` and `_evidence_conflict` are **untouched** (signed-off 07d behavior preserved; verification remains fail-safe and effectively always-on on this corpus).
- `classify_complexity`, the FAST/BALANCED/STRONG tiers, and all routing are unchanged.
- No model/provider/policy/quota changes. Zero live API calls made. No duplicate or speculative calls introduced.

## 13. Known Limitations / Caveats

- **The p95 ≤ 8 s target is NOT met and is NOT achievable on this fixed control.** It is set by the ~21 genuinely-complex full-path queries plus mandatory 2.0 s verification; the p95 percentile index (36th of 38) makes it impossible to move without fast-pathing D/E/F/I.
- The evidence-sufficiency gate (**top1 ≥ 0.8, margin ≥ 0.12**) has a **high false-positive rate on complex classes in this synthetic corpus** (filler-chunk score inflation), so it is a poor fast-path discriminator here. A real corpus with cleaner score separation *might* distinguish better, but that is hypothetical and would need its own control evidence.
- The custom-built corpus is designed for the eval; scores are not necessarily representative of arbitrary production corpora — but the *structural* percentile argument (§6.3) is corpus-independent and is the primary blocker.
- No **live** latency re-verification was run (quota preserved). The 07c mock latency model is reused as the honest projection basis.

## 14. Risk & Safety Assessment

- **COWL avoids all risk:** no code change, no new gate surface, no fast-path misfire, no verification weakening, no quota/call change, no provider change. Quality is bit-for-bit the signed-off 07f state.
- Had the unsafe margin gate been shipped (§6.1), the risk was concrete: D3 (wrong cross-document sum), E3 (unverified multi-hop), F2 (unverified conflict answer), I2 (collapsed 3-doc research synthesis). This is precisely why calibration was done before implementation.

## 15. Decision Notes

- Per owner direction, Phase 07g is concluded **COWL**: no fast-path widening is implemented on this control because the evidence shows it cannot meet the p95 objective honestly and the only gains would be mean/call-efficiency at the cost of a non-trivial new gate surface and real risk to D/E/F/I if thresholds drift.
- The conclusion directly extends 07f's §17 finding: the "evidence-lean gate" it proposed as a follow-up was measured and found to misfire on the very classes it must protect; the "verified cheaper latency tier" remains the one avenue the brief does not authorize (it would require policy/scope change).
- Phase 08 remains NOT the next step.

## 16. Honest Verdict

**COMPLETE WITH LIMITATIONS (COWL — no code change).**

Phase 07g was scoped as an investigation into whether a safe LLM fast path could reduce p95 below the 07f baseline. **It cannot, on this fixed control, without falsifying quality.** Measured evidence shows:

1. **Verification (2.0 s, every query) is mandatory** — the signed-off 07d `_evidence_conflict` (CV > 0.3) guard fires on essentially all 38 queries, closing the verification-lean lever.
2. **The evidence-sufficiency fast-path gate misfires on the complex classes it must protect** — `top1 ≥ 0.8, margin ≥ 0.12` catches `D3`, `E3`, `F2`, `I2` (filler-chunk score inflation).
3. **p95 is structurally pinned.** The p95 percentile (36th of 38) requires ≥ 37/38 queries on the fast path to drop below ~8.0 s; only ~7 balanced queries are safely fast-pathable. Even the unsafe gate leaves p95 at 8.02 s.

The only honest, reproducible gains are a **~13 % mean reduction** and a small **calls/query reduction** for 7 safe balanced queries, with **p95 flat at ~8.2 s**. Because the objective is p95-specific and the gain fails it without advancing the header metric, Phase 07g ships **no code change**. Quality, grounding, citations, verification, reliability, and call-efficiency remain at the signed-off 07f state. The task is complete as an evidence-driven, COWL decision.

## 17. Recommended Next Steps

- (Recommended) Close 07g with the COWL report; do **not** widen the fast path on this control. Consider whether p95 ≤ 8 s should be re-scoped (see below).
- **If the p95 target must be met**, it requires one of:
  (a) a **permitted cheaper/lower-latency verification path** for the low-risk fast path (an area 07d showed the gate *can* skip on genuine low-risk evidence) — a **policy/scope change**, with its own control evidence; OR
  (b) a **verified, still-safe reduction of the serial LLM sum** (e.g., merging analyze+plan for a narrowly-gated subset with a dedicated control demonstrating D/E/F/I remain correct); OR
  (c) formally accepting **p95 ≈ 8.2 s as the practical floor** on clean-path (without retry overhead) for the fixed workload — matching 07f's §15/§16.
- Optional follow-up: a **live (small) re-run** (quota permitting) to validate the mock latency model against real per-call latencies before any future fast-path decision.
- Revisit whether a future phase (with a realistic, lower-CV corpus) can make the evidence-sufficiency gate discriminative; on this synthetic control it is not.

## 18. Artifacts

- **Report:** `E:\ARGUS\ARGUS\PHASE_07G_FINAL_REPORT.md` (this file, 18 sections).
- **Baseline (unchanged, do not overwrite):** `results/regression_07f.json`, `regression_07d.json`, `regression_07e.json`, `benchmarks/eval_data/` (12-doc corpus + 38-query plan).
- **No new benchmark/test/production artifacts** — Phase 07g is a COWL conclusion; the calibration script was run offline and removed (no repository footprint).
- **Code state:** working tree clean at commit `d2d5f48` (07f). No 07g commit.
- **Vault:** `E:\ARGUS_VAULT\02_PROJECT_STATE.md`, `01_MASTER_PHASE_INDEX.md`, `handoffs\CURRENT_HANDOFF.md`, `logs\IMPLEMENTATION_LOG.md` updated to record the 07g COWL decision.