# Phase 37 Report: LLM Call Minimization & Fast-Path Audit

**Date:** 2026-09-07
**Status:** Complete — no production changes

---

## 1. Executive Summary

Phase 37 investigated whether ARGUS makes more LLM calls than necessary. **Result: Yes — up to 41% of tokens and 2.6 calls per query could theoretically be saved.** However, provider instability (Groq TPM exhaustion, Gemini quota, z.ai timeouts) makes ablation results untrustworthy for production decisions. The existing fast path (Phase 06.5.3) already handles the highest-value case (simple_lookup). No production changes are justified until provider stability improves.

**Production Decision: OPTION E — INFRASTRUCTURE STILL BLOCKED**

---

## 2. Complete LLM Call Inventory

| # | Call | call_type | Avg Latency | Tokens | Required? | Skip Candidate? |
|---|------|-----------|-------------|--------|-----------|----------------|
| 1 | Query Analysis | query_analysis | ~1035ms | ~430 | YES | YES — already skipped for FAST tier |
| 2 | Research Planning | research_planning | ~1500ms | ~360 | YES | YES — already skipped for FAST tier |
| 3 | Evidence Assessment | evidence_extraction | ~1173ms | ~870 | YES | YES — has deterministic short-circuits |
| 4 | Synthesis | synthesis | ~1526ms | ~1210 | YES | NO — final answer production |
| 5 | Verification | verification | ~1334ms | ~870 | CONDITIONAL | YES — 06.5.4 gate; annotation only |

---

## 3. Calls Per Query

| Query | Pattern | Fast? | Calls | Tokens | LLM ms | E2E ms |
|-------|---------|-------|-------|--------|--------|--------|
| P37-Q01 | simple_lookup | Y | 2 | 1823 | 2262 | 3077 |
| P37-Q02 | normal_qa | N | 6 | 6656 | 8450 | 10443 |
| P37-Q03 | technical_explanation | N | 6 | 6755 | 10046 | 12902 |
| P37-Q04 | multi_hop | N | 6 | 3683 | 6186 | 7577 |
| P37-Q05 | numerical | N | 5 | 3624 | 6992 | 8274 |
| P37-Q06 | conflict | N | 7 | 6076 | 8206 | 10159 |
| P37-Q07 | complex_research | N | 7 | 2585 | 26917 | 27984 |
| P37-Q08 | absent_info | N | 6 | 3888 | 7708 | 9095 |

**Average (non-fast):** 6.1 calls, 4895 tokens, 10644ms LLM, 12062ms E2E

---

## 4. Token Usage

| Call Type | Total Tokens | % of Total | Avg per Call |
|-----------|-------------|------------|--------------|
| query_analysis | 3178 | 9.1% | 530 |
| research_planning | 2017 | 5.8% | 202 |
| evidence_extraction | 12354 | 35.2% | 950 |
| synthesis | 9689 | 27.6% | 1211 |
| verification | 7852 | 22.4% | 982 |
| **TOTAL** | **35090** | **100%** | **780** |

**evidence_extraction + synthesis = 62.8% of all tokens.**

---

## 5. LLM Latency Breakdown

| Call Type | Total ms | % of Total | Avg per Call |
|-----------|---------|------------|--------------|
| query_analysis | 6210 | 8.1% | 1035 |
| research_planning | 32420 | 42.2% | 3242 |
| evidence_extraction | 15255 | 19.9% | 1173 |
| synthesis | 12209 | 15.9% | 1526 |
| verification | 10673 | 13.9% | 1334 |
| **TOTAL** | **76767** | **100%** | **1706** |

**research_planning dominates at 42.2%** — but this is skewed by a 21s zai timeout on P37-Q07.

---

## 6. Call Dependency Graph

```
query_analysis → plan → retrieve → assess → [loop] → synthesize → verify
     ↓              ↓                                        ↓
  complexity    subquestions                              annotation
  classification  decomposition                          only
```

---

## 7. Fast-Path Opportunities

### Already Implemented
- **Phase 06.5.3:** Simple lookup fast path — saves 3 calls (analyze, plan, assess)
- **Deterministic short-circuits in assess:** Budget exhaustion, empty retrievals
- **Verification 06.5.4 gate:** Skips verification for low-risk queries (rarely fires)

### Candidates (Not Implemented)
| Candidate | Calls Saved | Token Reduction | Risk |
|-----------|-------------|-----------------|------|
| Skip analyze for MODERATE | 1.2 | 25% | Low — deterministic classifier available |
| Skip plan for simple patterns | 1.0 | 23% | Low — subquestions=[query] is sufficient |
| Skip verify for simple/numerical | 2.6 | 40% | Medium — lose contradiction detection |
| Skip analyze+plan | 1.6 | 33% | Low — combined savings |

---

## 8. Controlled Ablation Results

### Configuration Comparison

| Config | Calls | Δ | Tokens | Δ% | E2E ms | Δ% |
|--------|-------|---|--------|-----|--------|-----|
| A_full (baseline) | 6.6 | — | 4253 | — | 18113 | — |
| B_no_analyze | 5.4 | -1.2 | 3190 | -25% | 9305 | -49% |
| C_no_plan | 5.6 | -1.0 | 3269 | -23% | 9495 | -48% |
| D_no_analyze_plan | 5.0 | -1.6 | 2856 | -33% | 11419 | -37% |
| E_no_verify | 4.0 | -2.6 | 2552 | -40% | 11252 | -38% |
| F_no_analyze_plan_verify | 4.6 | -2.0 | 2508 | -41% | 9177 | -49% |

**WARNING:** E2E latency numbers are UNRELIABLE due to provider instability. Call counts and token counts are deterministic and trustworthy.

---

## 9. Pattern-Specific Results

| Pattern | Best Config | Calls Saved | Token Reduction |
|---------|-------------|-------------|-----------------|
| simple_lookup | F (no analyze+plan+verify) | 1.0 | -37% |
| normal_qa | B (no analyze) | 1.0 | -37% |
| technical_explanation | D (no analyze+plan) | 2.0 | -35% |
| multi_hop | D (no analyze+plan) | 1.0 | -15% |
| numerical | F (no analyze+plan+verify) | 4.0 | -76% |
| conflict | B (no analyze) | 1.0 | -56% |
| complex_research | E (no verify) | 2.0 | -17% |
| absent_info | F (no analyze+plan+verify) | 1.0 | -21% |

---

## 10. Provider TPM Impact

| Optimization | Est. Tokens/Query | Queries Before TPM (8K) | Fallback Reduction |
|-------------|-------------------|------------------------|-------------------|
| Current (full) | ~4895 | ~1.6 | baseline |
| Skip analyze+plan | ~3200 | ~2.5 | moderate |
| Skip verify | ~3000 | ~2.7 | moderate |
| Skip all three | ~2400 | ~3.3 | significant |
| Fast path (existing) | ~1823 | ~4.4 | major |

**Key insight:** Reducing from ~4895 to ~2400 tokens/query would double the number of queries before TPM exhaustion (from ~1.6 to ~3.3). This would significantly reduce fallback frequency.

---

## 11. Fallback Reduction

Current fallback rate: 100% (all queries had fallbacks)

If call minimization were implemented:
- **Skip analyze+plan:** ~33% fewer tokens → ~50% more queries before TPM exhaustion
- **Skip verify:** ~40% fewer tokens → ~67% more queries before TPM exhaustion
- **Skip all three:** ~41% fewer tokens → ~70% more queries before TPM exhaustion

**However:** These estimates assume stable provider behavior. Actual fallback reduction depends on provider quota recovery rates.

---

## 12. Quality Comparison

Quality could not be formally assessed (no gold-standard benchmark). Proxy metrics:

| Config | Avg Answer Length | Notes |
|--------|------------------|-------|
| A_full | 776 chars | Baseline |
| B_no_analyze | 864 chars | +11% longer |
| C_no_plan | 838 chars | +8% longer |
| D_no_analyze_plan | 954 chars | +23% longer |
| E_no_verify | 667 chars | -14% shorter |
| F_no_analyze_plan_verify | 777 chars | Same |

**Answer length is not a quality metric** — longer answers are not necessarily better. Formal quality evaluation is needed.

---

## 13. Implemented Changes

**None.** No production code was modified. All ablation results were produced via monkey-patching that was restored after each run.

---

## 14. Regression Results

Not applicable — no production changes were made.

---

## 15. Production Decision

**OPTION E — INFRASTRUCTURE STILL BLOCKED**

### Justification

1. **Theoretical savings are clear:** Up to 41% token reduction, 2.6 fewer calls per query
2. **Provider instability prevents validation:** Cannot measure quality impact of skipped calls
3. **Existing fast path handles highest value case:** simple_lookup already uses 2 calls
4. **Verification skip is largest opportunity:** But requires quality validation before implementation
5. **No gold-standard quality benchmark:** Cannot measure if skipped calls degrade answers

### What Would Change This Decision

1. **Paid-tier Groq** (higher TPM) → enables clean ablation runs
2. **Quality benchmark** (gold-standard answers) → enables quality impact measurement
3. **Stable providers** → enables reproducible measurements

---

## 16. Recommended Phase 38 Target

**Phase 38: Quality Baseline & Gold-Standard Answers**

Before implementing any fast paths, establish:
1. Gold-standard answers for each query pattern
2. Quality metrics (claim support, citation precision, fact coverage)
3. A quality gate that can detect regressions

This is a prerequisite for safely implementing call minimization.
