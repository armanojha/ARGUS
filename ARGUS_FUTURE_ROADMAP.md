# ARGUS — FUTURE ROADMAP

**Date:** 2026-09-03
**Version:** derived from `PHASE_07C_REAL_WORLD_EVALUATION.md` (evidence-driven, post-Phase 07b).
**Basis:** Level-3 core today; Level-4 (advanced, degradation-proof, near-real-time practical RAG) is the honest target. Roadmap is driven by **measured** quality × reliability × efficiency, not by feature-by-feature projection.

---

## 1. Current Standing

- **What exists and is proven:** grounded + cited answers; never fabricates; real multi-provider circuit-breaker resilience (7/7 mock, live fallback); bounded fail-safe verification; complexity-adaptive fast path; health/quota/capability gating; honest telemetry; 462 tests pass. Verified live on a healthy subset (5/5 correct answers).
- **What was measured as the gap to the "advanced" (Level 4) claim:**
  1. Verification "selectivity" is effectively disabled — skip gate returned False 0/38 → every query pays +1 call +1.5–2.4 s.
  2. Burst rate-limiting cascades to total outage with no in-session recovery (13/18 live queries degraded).
  3. `stop_reason` mislabels outcomes (success and total outage both reported as uninformative reasons).
  4. Full-width `【N】` citations silently downgrade to top-3.
  5. All-sequential LLM calls → p95 ≈13 s; 22 s/2-call wasted failure path on Zen last-resort.
- **Headline verdict:** Level 3 (advanced core) now; Level 4 requires *calibration + recovery engineering*, not new subsystems.

---

## 2. Roadmap Principle

**Rows only if a measured gain is demonstrated by running the same controlled control (38-query `eval_plan_v1`) before and after.** No attractive-but-unverified additions. Capability rows are gated by calibration/robustness rows being proven first (dependency order, §6). "More features" is explicitly rejected until the Level-4 bar (§13) is met.

---

## 3. Priorities

Ordered by quality × reliability × efficiency impact (evidence in the evaluation):

1. **Calibrate the verification gate** so it truly skips simple/low-risk queries (largest latency + cost lever; verified 0/38 today). *Small, measured, high value.*
2. **Graceful burst-degradation + in-session recovery** (cascade-to-outage is real; must become graceful, not evidence-dump).
3. **Truthful outcome/`stop_reason` signal** (observability; required to trust any auto-report and any future metric-gating).
4. **Citation normalization** (`【N】`→`[N]`, small, high grounding value).
5. **Safe parallelism for independent retrieval/evidence calls** (p95 13 s → interactive; only after determinism is preserved).
6. **Faster/cheaper failure path & Zen-last-resort handling** (22 s → fast-fail).

Capability items (multi-hop graph, better retrieval precision, memory, agents) are **deferred** until the calibration/robustness rows above move the metric.

---

## 4. Dependencies

- **Row 1 (verification calibration)** depends on: *none* (pure threshold/gate logic + tests). Earliest.
- **Row 2 (recovery)** depends on: Row 3's truthful outcome signal to verify it works; independent otherwise. Can run after Row 1.
- **Row 3 (outcome signal)** depends on: none; is a pre-requisite for trustworthy telemetry of Rows 1/2.
- **Row 4 (citations)** depends on: none (normalization in `nodes.py` parse). Independent; can land anytime.
- **Row 5 (parallelism)** depends on: Rows 1–3 stable so latency deltas are measurable; needs deterministic output.
- **Row 6 (failure path)** depends on: none; independent.
- **Row 7+ (new capabilities)** depend on: all of Rows 1–6 meeting the re-run bar, else they add surface area on an unproven core.

---

## 5. Phases / Helicopter View

- **Phase 07c (this):** evaluation + roadmap only. No code. ✓ complete.
- **Phase 07d (recommended, small-caliberation):** Rows 1, 3, 4 (+ Row 6 — 20-line fixes, low risk), each with a before/after on `eval_plan_v1`. No new subsystems.
- **Phase 07e (recovery engineering, measured):** Row 2 (graceful burst-degradation + in-session recovery) — the reliability half of Level 4; needs dedicated mock stress + a bounded live burst.
- **Phase 07f (latency, measured):** Row 5 (safe parallelism) + Row 6 completion; target p95 ≤ 8 s nominal.
- **Phase 08+ (ONLY after 07d–07f prove the Level-4 bar):** any genuinely new capability (e.g. evidence-context budget, graph-multi-hop) — gated on approval, driven by measurement, never bundled speculatively.

---

## 6. Phase Dependencies (Order)

```
07c evaluation (done)
   └─> 07d calibration (verification gate, outcome signal, citations, fast-fail)   [proof on eval_plan_v1]
         └─> 07e recovery engineering (burst-degradation, in-session recovery)     [mock stress + bounded live]
              └─> 07f latency (safe parallelism)                                    [p95<=8s nominal]
                   └─> 08+ new capabilities (ONLY if Level-4 bar met, §13)
```

No phase starts before the previous phase's control re-run shows a measured gain.

---

## 7. Capability Tiers

- **Tier 1 — Calibration/robustness (do first, they ARE the gap):** verification-gate recalibration; truthful outcome/`stop_reason`; citation normalization; graceful burst-degradation + recovery; fast-fail last resort.
- **Tier 2 — Efficiency (do next, gated on metrics):** safe parallelism for independent calls; evidence-context budget (clear the always-on top-8 dilution); smarter retrieval-precision for grounding.
- **Tier 3 — New capabilities (deferred, evidence-gated):** graph-based multi-hop retrieval; memory/personalization; multi-agent workflows; better rerankers; near-real-time streaming. Each only after the Level-4 bar and a positive ROI on the control.

---

## 8. Technical Roadmap (Tactical)

- **Best first step:** Rows 1+3+4+6 together in one small 07d pass — they are localized (coordinator gate calibration; `nodes.py` citation normalization; a truthful outcome enums; a fast-fail constant for the last resort) and each is testable in isolation; combined they cut the per-query verification tax, restore trustworthy telemetry, and stop wasting 22 s on a known-failing path.
- **Then Row 2** as its own focused reliability phase (mock burst scenarios → bounded live burst), because it changed the whole evaluated outcome (13/18 degraded).
- **Then Row 5** when latency is the binding constraint on the metric, not a systemic signpost issue.
- **Measurement:** reuse `benchmarks/eval_data/eval_plan_v1.json` + live-results + resilience harness as the fixed control; record mean/median/p95 latency, calls/query, tokens/query, verdict-skip rate, recall@8, answer-correctness, and "burst outcome distribution" before and after each phase.

---

## 9. Milestones

- **M0 (baseline, done):** 38-query control + live healthy n=5 + 7 mock resilience + level verdict (L3). Recorded metrics in `PHASE_07C`.
- **M1 (Phase 07d):** verdict-skip rate >0 and rising (0/38→target ≥60% on simple/low-risk); truthful outcome on all 38; citation-specific (not top-3) on ≥80% of cited answers; failure path <3 s; all 462 tests + new tests green.
- **M2 (Phase 07e):** burst stress (3+ rapid queries, one provider degraded) yields ≥80% `answered_cited` (graceful), 0% evidence-dump-only on recoverable cases; in-session recovery demonstrated on a cooldown-expired provider.
- **M3 (Phase 07f):** p95 latency ≤8 s nominal load; no regression in recall/correctness.
- **M4 (Level-4 bar met, §13):** system earns "advanced practical RAG"; then evaluate capability phases with the same control.

---

## 10. Acceptance Criteria

For **each** phase:
1. Re-run the fixed 38-query control + resilience harness on the same corpus/settings.
2. Show the target metric moved in the right direction (e.g. verdict-skip 0→≥60%; burst outcome; p95; failure-path time) — or the change is reverted.
3. New behavior covered by tests; full suite (462 + new) green; `ruff` clean.
4. No secrets logged/committed; no unrelated files modified; no weakened test.
5. Working tree always left consistent (commit as recovery points when the owner asks).

---

## 11. Risks

- **Quota/cooldown contention during live re-runs** — mitigated by making re-runs small and preferring mock for most stress (only bounded live bursts).
- **Verification-gate recalibration could over-skip and lose the correctness safety-net** — mitigated by keeping the fail-safe verifier on truly high-risk/conflicting evidence and measuring recall/correctness regression.
- **Parallelism could alter result determinism** — mitigated by only parallelizing provably independent steps and keeping stable seeding.
- **Recovery logic could re-introduce dead-wait on a still-down provider** — mitigated by health-backed probe with hard timeouts and the existing 0-repeat guarantee.
- **Scope creep / speculative features** — mitigated by the explicit gate in §13 and the "measure-or-revert" rule.
- **Zen instability (known)** — never treated as reliable; fast-fail and prefer stable primaries.

---

## 12. Strategic Direction

Keep ARGUS a **practical, grounded, reliable RAG** — not a research gadget. The differentiator is "never fabricates + degrades gracefully + efficient." Strategic bets, in order:
1. **Make reliability boringly good** (recovery, truthful signals, fast failure) — this is what separates a research prototype from a product.
2. **Make efficiency real at the margin** (selective verification, safe parallelism) — latency/cost are what users feel.
3. **Only then** expand capability breadth (multi-hop, memory, deeper retrieval), each evidence-gated.
Resist: adding providers, agent frameworks, vector DBs, or memory *purely to grow the feature list*. Success = a system that is trustworthy under real load, not a wide but shallow one.

---

## 13. When to Stop Adding Features — the Definition

> **ARGUS is "good enough to be considered advanced and to stop adding features" when ALL of the following are true AND stable across two consecutive control re-runs:**

1. **Answer quality:** gruth-truth answer-correctness ≥ **0.85** and retrieval recall@8 ≥ **0.90** on the fixed 38-query control.
2. **Efficiency of verification:** the selective gate fires (skips verification) on ≥ **60%** of simple/low-risk queries, with no quality regression — i.e. verification is genuinely *selective*, not mandatory.
3. **Reliability under load:** a burst of 3+ rapid queries with one provider degraded still yields ≥ **80%** `answered_cited` (graceful, real answer) — no evidence-dump-only collapse.
4. **Latency:** p95 ≤ **8 s** on nominal load.
5. **Observability:** every query returns a truthful `outcome ∈ {answered_cited, answered, degraded_evidence, unavailable}` — `stop_reason` never mislabels success as failure or vice versa.
6. **Failure cost:** a degraded-answer path completes in **< 5 s** (no 22 s slow-fail).

**Interpretation:** 1–2 define the *quality + cost* floor; 3 + 4 + 6 define the *reliability + latency* floor under real load; 5 makes 1–4 verifiable automatically. **Until these are met, the correct work is calibration and recovery engineering — NOT new capabilities. Once they are met and stable for two consecutive runs, the owner may stop adding features or may open a deliberate, measurement-gated capability phase; pursuing neither is the correct default.** This is the concrete, unambiguous stop-line the project needs so ARGUS never drifts into feature-chasing.