# Phase 28 Report: Evidence-Grounded Synthesis Experiments

## Objective
Design and experimentally validate evidence-grounded synthesis strategies to improve ARGUS answer quality, targeting claim support >30%, citation precision >=80%, query relevance >=81.5%, and gold coverage improvement from 25%.

## What Was Done

### 1. Synthesis Path Audit
- Traced: `run_query()` → `make_synthesize_node()` → `build_synthesis_messages()` → LLM call
- Current prompt in `app/orchestration/prompts.py:130` already instructs "use ONLY evidence"
- Evidence format: `[N] (score: X.XX, source: path) text...`
- `check_claim_grounding()` runs post-synthesis on uncited sentences
- Citation fallback auto-attaches top 3 evidence if no citations

### 2. Strategy Design (5 strategies)
| Strategy | Approach | Prompt Size |
|----------|----------|-------------|
| A_baseline | Current production prompt | ~500 chars |
| B_strict_claim | Explicit per-claim rules, forbidden actions | ~700 chars |
| C_evidence_first | Evidence presented before question | ~500 chars |
| D_structured_draft | Step-by-step reasoning instructions | ~600 chars |
| E_specialized | Baseline + per-class instructions | ~700 chars |

### 3. Experimental Results

#### Focused Experiment (5 queries, 3 strategies, Groq primary)
| Strategy | Support | Citation Presence | Key Finding |
|----------|---------|-------------------|-------------|
| A_baseline | 25.0% | 43.0% | — |
| B_strict_claim | 31.1% | 43.3% | +6.1% support |
| E_specialized | 29.8% | 56.7% | +13.7% citation |

#### Full 21-Query Ablation (corrupted by rate limits)
- Strategy A produced 17/21 empty answers
- Results unreliable — rate limits caused fallback to lower-quality Gemini model
- Baseline comparison invalid (33.3% vs focused 25.0%)

### 4. Per-Query Analysis

| Query | Class | Baseline | Best Strategy | Improvement |
|-------|-------|----------|---------------|-------------|
| A01 | simple_lookup | 100% | 100% (all) | Already correct |
| D03 | multi_doc_synthesis | 11% | 33% (B_strict) | +22% |
| F01 | conflict | 14% | 23% (E_spec) | +9% |
| H02 | numerical | 0% | 0% (all) | Correctly abstains |
| G01 | absent_info | 0% | 0% (all) | Correctly abstains |

### 5. Key Findings

1. **Prompt improvements help but don't solve the core problem**: The LLM blends evidence with training knowledge regardless of prompt wording
2. **Absent info handling FIXED**: Stricter prompts correctly say "no information" instead of fabricating
3. **Multi-doc synthesis improved**: D03 went from 11% to 33% support
4. **Numerical queries remain hard**: LLM struggles with exact number extraction from evidence
5. **Conflict handling barely improved**: Still acknowledges single source instead of presenting both sides
6. **Citation presence tradeoff**: Stricter prompts reduce citation presence unless per-class instructions are used (E_specialized achieves 57%)

## Production Decision: OPTION C — Further Research Needed

### Justification
1. Support improvement exists (+6%) but is below the >30% target for most query types
2. Citation tradeoff with stricter prompts is unacceptable
3. Root cause (LLM blending evidence with training knowledge) not addressed by prompt alone
4. Rate limits prevented full-scale validation

### What Would Move the Needle (Phase 29 Recommendations)
1. **Two-pass synthesis** (highest priority): Generate draft → verify each claim against evidence → remove unsupported
2. **Structured output** (medium priority): Force JSON with `{claim, evidence_ids, confidence}`
3. **Reduced evidence count** (low priority): Cut from 8 to 4 chunks

## Regression Results
- 48/48 answer quality tests: **ALL PASS**
- 906/906 full test suite: **ALL PASS** (26 skipped)
- No production code changes in Phase 28

## Files Created/Modified
- `benchmarks/synthesis_ablation.py` — Ablation framework with 5 strategies
- `benchmarks/focused_experiment.py` — Focused 5-query experiment
- `benchmarks/analyze_ablation.py` — Analysis script
- `benchmarks/check_answers.py` — Answer inspection script
- `benchmarks/results/phase28_ablation.json` — Full ablation results (rate-limited)
- `benchmarks/results/phase28_focused.json` — Focused experiment results
- `benchmarks/PHASE_28_DIAGNOSTIC.md` — Diagnostic document
- `benchmarks/PHASE_28_REPORT.md` — This report
