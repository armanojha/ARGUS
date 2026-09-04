# ARGUS Public Release Audit

**Audit Date:** 2026-09-04
**Scope:** Full repository at `E:\ARGUS\ARGUS`
**Tracked files:** 236
**Purpose:** Identify everything that must be addressed before public release on GitHub, LinkedIn, portfolio showcases, and external reviewers.

---

## 1. Executive Summary

ARGUS is a substantial, well-structured project (105 source files, 54 test files, 436 passing tests). The code itself is clean and ready for public scrutiny. However, the repository carries significant **internal development artifacts** that must be removed before public release: 14 internal phase reports, a Groq organization ID leaked in tracked data files, hardcoded developer-local paths, and a README that references an internal vault directory not included in the repository.

**Critical blockers (4):**
1. No LICENSE file — pyproject.toml says "Proprietary" but no LICENSE exists
2. Groq organization ID (`org_01m1br4mkmed6ttx74zj5hhhzt`) exposed in 2 tracked JSON files
3. 14 internal PHASE/FINAL/VALIDATION report files tracked in git root
4. README references `E:\ARGUS_VAULT` (not in repo) as the "source of truth"

**Overall readiness:** The code is production-quality. The packaging is not.

---

## 2. Current Repository Structure

```
E:\ARGUS\ARGUS/
├── .env                          # LIVE SECRETS (gitignored, NOT tracked)
├── .env.example                  # Template (safe)
├── .gitignore                    # Good coverage
├── AGENTS.md                     # Internal AI agent contract
├── ARGUS_FUTURE_ROADMAP.md       # Internal roadmap
├── FINAL_PROJECT_REPORT.md       # Internal close-out report
├── PHASE_06_6_BASELINE_REPORT.md # Internal report
├── PHASE_07_FINAL_REPORT.md      # Internal report
├── PHASE_07B_FINAL_REPORT.md     # Internal report
├── PHASE_07C_REAL_WORLD_EVALUATION.md
├── PHASE_07D_FINAL_REPORT.md
├── PHASE_07E_FINAL_REPORT.md
├── PHASE_07F_FINAL_REPORT.md
├── PHASE_07G_FINAL_REPORT.md
├── PHASE_08_FINAL_REPORT.md
├── PHASE_08_READINESS_AUDIT.md
├── PHASE_KS_FINAL_REPORT.md
├── POST_06_5_VALIDATION_REPORT.md
├── README.md                     # Needs rewrite
├── pyproject.toml                # Needs license fix
├── app/                          # 105 source files — clean
├── benchmarks/                   # Code + eval data — mostly clean
├── configs/                      # 4 YAML configs — clean
├── data/                         # MIXED: 3 tracked JSON files with leaked org ID
├── notebooks/                    # Empty (just .gitkeep)
├── scripts/                      # 5 scripts — has hardcoded path
├── tests/                        # 54 test files — clean
├── TMP/                          # Temporary directory (not tracked, but exists on disk)
├── .Opencode/                    # Agent config (gitignored)
├── .mypy_cache/                  # Cache (gitignored)
├── .pytest_cache/                # Cache (gitignored)
├── .ruff_cache/                  # Cache (gitignored)
├── argus.egg-info/               # Empty (gitignored)
└── app/observability/            # Empty directory (only __pycache__)
```

---

## 3. Files to KEEP

### Source Code (all of `app/`)
All 105 `.py` files in `app/` are production source code. KEEP all.

### Tests (all of `tests/`)
All 54 `.py` files in `tests/` are production tests. KEEP all.

### Benchmarks Code
| File | Reason |
|------|--------|
| `benchmarks/__init__.py` | Package init |
| `benchmarks/models.py` | BenchmarkItem, BenchmarkRunOutput, CorpusContext |
| `benchmarks/metrics.py` | 9 deterministic metrics + aggregation |
| `benchmarks/runner.py` | Corpus build, pipeline composition, scoring, reports |
| `benchmarks/ablation.py` | 7 variant pipelines, delta table, markdown report |
| `benchmarks/data/questions_v1.json` | 110-question benchmark dataset |
| `benchmarks/eval_data/corpus_v1/*.md` | 12 synthetic evaluation documents |

### Configs
| File | Reason |
|------|--------|
| `configs/providers.yaml` | Provider configuration (safe — env-var references only) |
| `configs/model_policy.yaml` | Call-type routing policy (safe) |
| `configs/retrieval_policy.yaml` | Retrieval policy |
| `configs/obsidian.yaml` | Obsidian integration config |

### Scripts
| File | Reason |
|------|--------|
| `scripts/run_obsidian_vault.py` | Obsidian vault ingestion runner |
| `scripts/ingest_knowledge_base.py` | Knowledge base ingestion |
| `scripts/run_tests.ps1` | PowerShell test runner |
| `scripts/run_tests.sh` | Bash test runner |

### Root Files
| File | Reason |
|------|--------|
| `pyproject.toml` | Package definition (needs license fix) |
| `.gitignore` | Comprehensive, well-configured |
| `.env.example` | Safe template (needs path fix) |

### Data (gitkeep placeholders)
| File | Reason |
|------|--------|
| `data/graph/.gitkeep` | Directory structure |
| `data/indexes/.gitkeep` | Directory structure |
| `data/memory/.gitkeep` | Directory structure |
| `data/obsidian_index/.gitkeep` | Directory structure |
| `data/processed/.gitkeep` | Directory structure |
| `data/raw/.gitkeep` | Directory structure |

### Benchmark Eval Data
| File | Reason |
|------|--------|
| `benchmarks/eval_data/eval_plan_v1.json` | Evaluation plan |
| `benchmarks/eval_data/corpus_v1/*.md` | 12 synthetic documents for evaluation |

---

## 4. Files to DELETE

### 4.1 Internal Phase Reports (14 files)

| File | Lines | Why Delete |
|------|-------|------------|
| `AGENTS.md` | ~25 | Internal AI agent operating contract. References `E:\ARGUS_VAULT`. Not for human contributors. |
| `ARGUS_FUTURE_ROADMAP.md` | ~150 | Internal roadmap with detailed metrics, calibration targets, "owner" language. Replace with GitHub Issues/Projects. |
| `FINAL_PROJECT_REPORT.md` | ~110 | Internal close-out report. Exposes git commit hashes, internal workflow, "owner" language. |
| `PHASE_06_6_BASELINE_REPORT.md` | — | Internal baseline report. |
| `PHASE_07_FINAL_REPORT.md` | 128 | Internal phase report. |
| `PHASE_07B_FINAL_REPORT.md` | 125 | Internal audit report. |
| `PHASE_07C_REAL_WORLD_EVALUATION.md` | ~244 | Internal evaluation with provider-specific test results. |
| `PHASE_07D_FINAL_REPORT.md` | 337 | Internal calibration report. |
| `PHASE_07E_FINAL_REPORT.md` | 229 | Internal recovery engineering report. |
| `PHASE_07F_FINAL_REPORT.md` | 161 | Internal parallelism report. |
| `PHASE_07G_FINAL_REPORT.md` | 185 | Internal latency report. References `E:\ARGUS_VAULT`. |
| `PHASE_08_FINAL_REPORT.md` | 140 | Internal audit report. References `E:\ARGUS\ARGUS` path. |
| `PHASE_08_READINESS_AUDIT.md` | ~311 | Internal readiness audit. References `E:\ARGUS_VAULT`. |
| `PHASE_KS_FINAL_REPORT.md` | 219 | Internal report. References `E:/KNOWLEDGE BASE`, `C:\Users\LOQ`. |
| `POST_06_5_VALIDATION_REPORT.md` | 176 | Internal validation report. |

**Impact of deletion:** None — these are internal development artifacts with no code references. No imports, no test dependencies, no configuration references.

**Recommended action:** `git rm` all 14 files. Optionally archive to a private repo or local directory before deletion.

### 4.2 Data Files with Leaked Org ID (3 files)

| File | Size | Why Delete |
|------|------|------------|
| `data/_phase066_default.json` | 40KB | Contains Groq org ID `org_01m1br4mkmed6ttx74zj5hhhzt` in rate-limit error messages (~18+ lines). Git-tracked. |
| `data/_phase066_mm.json` | 32KB | Same org ID exposure. Git-tracked. |
| `data/_phase066_results_default.json` | 37KB | Same org ID exposure (~50+ lines). Git-tracked. |

**Impact of deletion:** None — these are internal phase 06.6 evaluation artifacts. No code imports them.

**Recommended action:** `git rm` all 3 files. Add `data/_phase066*.json` to `.gitignore`.

### 4.3 Empty/Placeholder Directories

| Path | Why Delete |
|------|------------|
| `app/observability/` | Empty directory with only stale `__pycache__/telemetry.cpython-314.pyc`. No source files. Dead code. |
| `argus.egg-info/` | Empty package metadata directory. Build artifact. |
| `notebooks/` | Contains only `.gitkeep`. No actual notebooks. |

**Impact of deletion:** None — empty directories with no code.

**Recommended action:** `rm -rf` these directories. Remove `notebooks/` from `.gitignore` if present. Keep `notebooks/.gitkeep` if you plan to add example notebooks later.

### 4.4 TMP Directory

| Path | Why Delete |
|------|------------|
| `TMP/evidence.db` | 57KB SQLite database in a temporary directory. Not tracked in git (confirmed via `git ls-files TMP/`), but exists on disk. |

**Impact of deletion:** None — temporary file, not tracked.

**Recommended action:** Delete from disk. Add `TMP/` to `.gitignore` (currently only `tmp/` and `temp/` are covered — case-sensitive on Linux).

---

## 5. Files to MOVE / ARCHIVE

### 5.1 Benchmark Eval Scripts

| File | Why Move |
|------|----------|
| `benchmarks/eval_data/regression_07d.py` | Internal regression script for phase 07d |
| `benchmarks/eval_data/regression_07e.py` | Internal regression script for phase 07e |
| `benchmarks/eval_data/regression_07f.py` | Internal regression script for phase 07f |
| `benchmarks/eval_data/resilience_eval.py` | Internal resilience evaluation |
| `benchmarks/eval_data/run_eval.py` | Internal eval runner |

**Why:** These are internal development scripts, not part of the public benchmark harness. They reference specific phase numbering and internal evaluation methodology.

**Recommended action:** Move to `benchmarks/eval_data/internal/` or delete. The public benchmark code (`benchmarks/runner.py`, `benchmarks/ablation.py`, `benchmarks/metrics.py`) is sufficient.

### 5.2 Benchmark Result Files

| File | Why Move |
|------|----------|
| `benchmarks/eval_data/results/adaptivity_analysis.json` | Internal analysis |
| `benchmarks/eval_data/results/live_results.json` | Internal live results |
| `benchmarks/eval_data/results/regression_07d.json` | Internal regression results |
| `benchmarks/eval_data/results/regression_07e.json` | Internal regression results |
| `benchmarks/eval_data/results/regression_07f.json` | Internal regression results |
| `benchmarks/eval_data/results/retrieval_analysis.json` | Internal analysis |
| `benchmarks/eval_data/results/retrieval_recap.json` | Internal recap |
| `benchmarks/eval_data/results/resilience_results.json` | Internal resilience results |

**Why:** Generated result files from internal evaluations. Not needed for public reproducibility.

**Recommended action:** Move to `benchmarks/eval_data/results/internal/` or delete. Keep the synthetic corpus (`corpus_v1/`).

---

## 6. Files Requiring Refactoring

### 6.1 Hardcoded Developer Paths

| File | Line(s) | Issue | Recommended Fix |
|------|---------|-------|-----------------|
| `app/config.py` | 134 | `E:/KNOWLEDGE BASE` as default `knowledge_base_path` | Change to `./knowledge_base` or make required with no default |
| `scripts/ingest_corpus.py` | 4, 39, 41 | `E:/KNOWLEDGE BASE` as default path | Change to `./knowledge_base` or require CLI arg |
| `app/ui/streamlit_app.py` | 8 | `E:/KNOWLEDGE BASE` in docstring | Update docstring |
| `app/ingestion/knowledge_base.py` | 4 | `E:/KNOWLEDGE BASE` in docstring | Update docstring |
| `app/llm_gateway/providers/exceptions.py` | 7 | `E:\\ARGUS_VAULT` in docstring | Remove vault reference |

### 6.2 Developer Identity Exposure

| File | Line | Issue | Recommended Fix |
|------|------|-------|-----------------|
| `PHASE_KS_FINAL_REPORT.md` | 146 | `C:\Users\LOQ\AppData\Local\Programs\Tesseract-OCR` | DELETE file (internal report) |

### 6.3 README.md Quality Issues

| Issue | Severity | Recommended Fix |
|-------|----------|-----------------|
| References `E:\ARGUS_VAULT` (lines 34, 104, 172-173) | CRITICAL | Remove all vault references. Replace with in-repo `docs/` or remove. |
| "Proprietary" license (line 178) | CRITICAL | Add LICENSE file or change to OSS license |
| Internal workflow language (lines 107-114) | HIGH | Rewrite for public contributors |
| No badges | LOW | Add shields.io badges (Python, license, tests) |
| No architecture diagram | LOW | Add Mermaid diagram |
| PowerShell-only examples | LOW | Add bash equivalents |
| Test count outdated (354 vs actual 436) | MEDIUM | Update to current count |
| Phase 12 listed as "not started" (line 149) | MEDIUM | Phase 12 is COMPLETE |

### 6.4 .env.example Cleanup

| File | Line | Issue | Recommended Fix |
|------|------|-------|-----------------|
| `.env.example` | 20 | `E:/KNOWLEDGE BASE` default path | Change to `./knowledge_base` |
| `.env.example` | 43, 47 | References vault decisions (D-005, D-014) | Remove internal decision IDs |

---

## 7. Security & Privacy Audit

### 7.1 Secrets Exposure

| Check | Result |
|-------|--------|
| `.env` tracked in git | **PASS** — not tracked |
| `.env` in git history | **PASS** — never committed |
| Real API keys in source code | **PASS** — only fake test keys (`sk-test-not-a-real-key`) |
| Real API keys in configs | **PASS** — only env-var references |
| Real API keys in tests | **PASS** — only fake values |
| Private keys / certificates | **PASS** — zero matches |
| `sk-` real key patterns | **PASS** — no real keys found |

### 7.2 Identity / Infrastructure Exposure

| Check | Result |
|-------|--------|
| `C:\Users\LOQ` | **FOUND** — `PHASE_KS_FINAL_REPORT.md:146` (delete file) |
| `E:\` local paths | **FOUND** — 34 matches across source/docs (see Section 6.1) |
| `localhost` references | **PASS** — 5 matches, all expected (API defaults) |
| Groq org ID | **CRITICAL** — `data/_phase066*.json` contains `org_01m1br4mkmed6ttx74zj5hhhzt` |

### 7.3 Telemetry / Benchmark Data

| Check | Result |
|-------|--------|
| `data/telemetry/runs.jsonl` | **PASS** — only synthetic test data (`"provider": "scripted"`) |
| `data/benchmark_reports/` | **PASS** — gitignored, contains only aggregate metrics |
| `data/evidence.db` | **PASS** — gitignored |
| `data/memory/memory.db` | **PASS** — gitignored |
| `data/indexes/*.pkl` | **PASS** — gitignored |

---

## 8. Secrets / Credential Exposure

| File | Status | Details |
|------|--------|---------|
| `.env` | **SAFE** | Not tracked, not in history, gitignored |
| `.env.example` | **SAFE** | Empty placeholders only |
| `configs/providers.yaml` | **SAFE** | Env-var references only (`api_key_env: GROQ_API_KEY`) |
| `configs/model_policy.yaml` | **SAFE** | No embedded keys |
| `app/config.py` | **SAFE** | Reads from env vars |
| `tests/test_config.py` | **SAFE** | Uses fake values |
| `tests/llm_gateway/test_openai_compatible.py` | **SAFE** | `sk-test-not-a-real-key` |
| `data/_phase066*.json` | **CRITICAL** | Groq org ID in error messages |

**Verdict:** No API keys or secrets are exposed. One organization ID is leaked in tracked data files.

---

## 9. Model & Provider Configuration Audit

### Current State

| Provider | Configured | API Key Env Var | Primary Model | Fallback |
|----------|-----------|-----------------|---------------|----------|
| Groq | Yes | `GROQ_API_KEY` | `openai/gpt-oss-120b` | `openai/gpt-oss-20b` |
| Gemini | Yes | `GEMINI_API_KEY` | `gemini-2.5-flash-lite` | `gemini-2.5-flash` |
| Cerebras | Yes (key unset) | `CEREBRAS_API_KEY` | `gpt-oss-120b` | — |
| OpenCode Zen | Yes | `OPENCODE_ZEN_API_KEY` | `nemotron-3-ultra-free` | `big-pickle`, `mimo-v2.5-free` |

### Call-Type Routing (`configs/model_policy.yaml`)

| Call Type | Primary | Fallback Chain |
|-----------|---------|----------------|
| `general` | `zen/nemotron-3-ultra-free` | zen→groq→gemini→cerebras |
| `query_analysis` | `zen/mimo-v2.5-free` | zen→groq→cerebras |
| `research_planning` | `zen/big-pickle` | zen→groq→gemini→cerebras |
| `evidence_extraction` | `zen/mimo-v2.5-free` | zen→groq→cerebras |
| `reasoning` | `zen/big-pickle` | zen→groq→cerebras |
| `synthesis` | `zen/big-pickle` | zen→groq→cerebras |
| `verification` | `zen/big-pickle` | zen→groq→groq→cerebras |
| `revision` | `zen/mimo-v2.5-free` | zen→groq→cerebras |

### Public Presentation Recommendation

**DO** hardcode the provider names and model IDs in documentation — these are the supported providers and the current configuration. The architecture is intentionally configuration-driven, so document:
- Which providers are supported (Groq, Gemini, Cerebras, Zen)
- Which models are configured for each call type
- How to change models (edit `configs/model_policy.yaml`)
- Which providers require API keys
- Which providers are optional (Cerebras, Zen)

**DO NOT** hardcode actual API keys. Use `.env.example` as the template.

**Recommended documentation approach:**
1. `.env.example` — shows which env vars to set (already good)
2. `configs/providers.yaml` — already clean, safe to commit
3. `configs/model_policy.yaml` — already clean, safe to commit
4. README — add a "Supported Providers & Models" section explaining the configuration

---

## 10. Benchmark & Generated Artifact Audit

### Tracked Files

| File | Size | Status | Action |
|------|------|--------|--------|
| `data/_phase066_default.json` | 40KB | **CRITICAL** — leaked org ID | DELETE |
| `data/_phase066_mm.json` | 32KB | **CRITICAL** — leaked org ID | DELETE |
| `data/_phase066_results_default.json` | 37KB | **CRITICAL** — leaked org ID | DELETE |
| `benchmarks/data/questions_v1.json` | — | SAFE — benchmark dataset | KEEP |
| `benchmarks/eval_data/corpus_v1/*.md` | — | SAFE — synthetic documents | KEEP |
| `benchmarks/eval_data/results/*.json` | — | Internal results | MOVE/DELETE |
| `benchmarks/eval_data/*.py` | — | Internal scripts | MOVE/DELETE |

### Gitignored Files (confirmed not tracked)

| File | Status |
|------|--------|
| `data/evidence.db` | Gitignored |
| `data/memory/memory.db` | Gitignored |
| `data/indexes/bm25.pkl` | Gitignored |
| `data/indexes/faiss.index` | Gitignored |
| `data/indexes/faiss.ids.pkl` | Gitignored |
| `data/obsidian_index/sync_manifest.pkl` | Gitignored |
| `data/telemetry/runs.jsonl` | Gitignored |
| `data/benchmark_reports/` | Gitignored |

### Recommendation

- **KEEP:** `benchmarks/data/questions_v1.json`, `benchmarks/eval_data/corpus_v1/`
- **DELETE:** `data/_phase066*.json` (leaked org ID, internal data)
- **MOVE/DELETE:** `benchmarks/eval_data/results/`, `benchmarks/eval_data/*.py` (internal scripts/results)
- **ADD to .gitignore:** `data/_phase066*.json`, `TMP/`, `*.db` (global)

---

## 11. Git Hygiene Audit

### .gitignore Coverage

| Pattern | Covered | Notes |
|---------|---------|-------|
| `.env` | Yes | Line 2 |
| `.env.*` (except `.env.example`) | Yes | Lines 3-4 |
| `__pycache__/` | Yes | Line 13 |
| `*.py[cod]` | Yes | Line 14 |
| `.venv/`, `venv/` | Yes | Lines 15-16 |
| `node_modules/` | Yes | Line 19 |
| `.vscode/`, `.idea/` | Yes | Lines 24-25 |
| `*.egg-info/` | Yes | Line 37 |
| `.pytest_cache/` | Yes | Line 38 |
| `.ruff_cache/` | Yes | Line 39 |
| `.Opencode/` | Yes | Line 40 |
| `data/evidence.db` | Yes | Line 43 |
| `data/memory/` | Yes | Line 44 |
| `data/indexes/` | Yes | Line 45 |
| `data/obsidian_index/` | Yes | Line 46 |
| `data/telemetry/` | Yes | Line 47 |
| `data/benchmark_reports/` | Yes | Line 48 |
| `TMP/` | **NO** | Only `tmp/` and `temp/` covered (lines 33-34) |
| `*.db` (global) | **NO** | Only specific paths covered |
| `data/_phase066*.json` | **NO** | These are tracked and need removal |

### Tracked Generated Files

| File | Issue |
|------|-------|
| `data/_phase066_default.json` | Generated evaluation data with leaked org ID |
| `data/_phase066_mm.json` | Same |
| `data/_phase066_results_default.json` | Same |

### Missing .gitignore Patterns

Add to `.gitignore`:
```
TMP/
*.db
data/_phase066*.json
```

---

## 12. README / Documentation Audit

### Current README Assessment

| Section | Quality | Issue |
|---------|---------|-------|
| Title & Description | GOOD | Clear, professional |
| Status | OUTDATED | Says "Phase 12 not started" — it's COMPLETE. Test count says 354, actual is 436. |
| Setup | GOOD | Clear instructions, but PowerShell-only |
| Running the API | GOOD | Clear examples |
| Obsidian ingestion | GOOD | Clear examples |
| Research UI | GOOD | Clear examples |
| Live LLM tests | GOOD | Clear |
| Development | BAD | References `E:\ARGUS_VAULT` as "source of truth" — not in repo |
| Testing & Quality | OUTDATED | Test count wrong |
| Roadmap | OUTDATED | Phase 12 listed as "NEXT" |
| Known limitations | OUTDATED | References vault (P10-01) |
| Contributing | BAD | Directs to `E:\ARGUS_VAULT\00_CLAUDE_BOOT.md` — not in repo |
| License | BAD | "Proprietary — see pyproject.toml" — no LICENSE file exists |

### Missing Documentation

| File | Priority | Purpose |
|------|----------|---------|
| `LICENSE` | **CRITICAL** | Must exist for public release |
| `CONTRIBUTING.md` | HIGH | Standard for open-source |
| `SECURITY.md` | HIGH | Standard for projects handling API keys |
| `CHANGELOG.md` | MEDIUM | Replace internal phase reports |
| `.github/CI.yml` | MEDIUM | Automated testing |
| `docs/architecture.md` | LOW | Architecture documentation |

---

## 13. GitHub Presentation Recommendations

### README Rewrite Checklist

1. **Remove all `E:\ARGUS_VAULT` references** — replace with in-repo equivalents
2. **Update status** — Phase 12 is COMPLETE, 436 tests passing
3. **Add badges** — Python version, license, test status
4. **Add architecture diagram** — Mermaid flowchart showing the pipeline
5. **Add "Supported Providers" section** — document which LLM providers work
6. **Add bash examples** alongside PowerShell
7. **Rewrite "Contributing"** — standard GitHub contribution guidelines
8. **Rewrite "Development"** — remove vault/phase workflow references
9. **Fix license** — add LICENSE file, update pyproject.toml
10. **Add "Quick Start"** section for immediate gratification

### Professional Presentation Score

| Aspect | Current | Target |
|--------|---------|--------|
| First impression | Mixed — internal reports clutter root | Clean, professional |
| Code quality | Excellent | Excellent |
| Documentation | Internal-focused | Public-facing |
| Security | Good (keys safe) | Excellent (org ID removed) |
| Reproducibility | Good (benchmarks) | Excellent (clean benchmarks) |
| Contributing experience | Confusing (vault refs) | Clear (standard GitHub) |

---

## 14. Recommended Final Repository Structure

```
ARGUS/
├── LICENSE                          # NEW — must add
├── README.md                        # REWRITE
├── CHANGELOG.md                     # NEW — summarize phase deliverables
├── CONTRIBUTING.md                  # NEW — standard guidelines
├── SECURITY.md                      # NEW — API key security guidance
├── pyproject.toml                   # FIX — license field
├── .gitignore                       # FIX — add TMP/, *.db, data/_phase066*.json
├── .env.example                     # FIX — remove vault refs, fix default path
│
├── app/                             # KEEP — all 105 source files
├── tests/                           # KEEP — all 54 test files
├── benchmarks/                      # KEEP code, CLEAN data
│   ├── __init__.py
│   ├── models.py
│   ├── metrics.py
│   ├── runner.py
│   ├── ablation.py
│   ├── data/
│   │   └── questions_v1.json        # KEEP
│   └── eval_data/
│       ├── corpus_v1/*.md           # KEEP
│       └── eval_plan_v1.json        # KEEP
│
├── configs/                         # KEEP — all 4 YAML files
├── scripts/                         # KEEP — fix hardcoded path
├── docs/                            # NEW — optional architecture docs
└── .github/                         # NEW — CI/CD workflows
```

**Files to DELETE (17):**
- 14 internal PHASE/FINAL/VALIDATION reports
- 3 data files with leaked org ID

**Files to MOVE (10):**
- 5 internal eval scripts from `benchmarks/eval_data/`
- 8 internal result files from `benchmarks/eval_data/results/`
- (Some may be deleted instead)

---

## 15. Exact Cleanup Checklist

### P0 — MUST FIX BEFORE PUBLIC RELEASE

| # | Action | Files | Command |
|---|--------|-------|---------|
| 1 | Add LICENSE file | `LICENSE` (new) | Create with chosen license text |
| 2 | Fix pyproject.toml license | `pyproject.toml:11` | Change `"Proprietary"` to `"MIT"` or `"Apache-2.0"` |
| 3 | Delete internal reports | 14 `.md` files in root | `git rm AGENTS.md ARGUS_FUTURE_ROADMAP.md FINAL_PROJECT_REPORT.md PHASE_*.md POST_06_5_VALIDATION_REPORT.md` |
| 4 | Delete leaked org ID files | `data/_phase066*.json` | `git rm data/_phase066_default.json data/_phase066_mm.json data/_phase066_results_default.json` |
| 5 | Fix README.md | `README.md` | Remove vault refs, update status, add badges, rewrite Contributing |
| 6 | Fix .env.example | `.env.example` | Remove vault decision refs, fix default path |
| 7 | Update .gitignore | `.gitignore` | Add `TMP/`, `*.db`, `data/_phase066*.json` |

### P1 — SHOULD FIX

| # | Action | Files | Details |
|---|--------|-------|---------|
| 8 | Fix hardcoded paths in source | `app/config.py:134`, `scripts/ingest_corpus.py:4,39,41` | Change `E:/KNOWLEDGE BASE` to `./knowledge_base` |
| 9 | Fix docstring vault refs | `app/ui/streamlit_app.py:8`, `app/ingestion/knowledge_base.py:4`, `app/llm_gateway/providers/exceptions.py:7` | Remove `E:\ARGUS_VAULT` references |
| 10 | Delete empty directories | `app/observability/`, `argus.egg-info/` | `rm -rf` |
| 11 | Move/delete internal eval scripts | `benchmarks/eval_data/regression_*.py`, `resilience_eval.py`, `run_eval.py` | Move or delete |
| 12 | Move/delete internal result files | `benchmarks/eval_data/results/*.json` | Move or delete |
| 13 | Add CONTRIBUTING.md | `CONTRIBUTING.md` (new) | Standard contribution guidelines |
| 14 | Add SECURITY.md | `SECURITY.md` (new) | API key security guidance |
| 15 | Add CHANGELOG.md | `CHANGELOG.md` (new) | Summarize phase deliverables |

### P2 — NICE TO HAVE

| # | Action | Files | Details |
|---|--------|-------|---------|
| 16 | Add architecture diagram | `README.md` | Mermaid flowchart |
| 17 | Add badges | `README.md` | shields.io badges |
| 18 | Add bash examples | `README.md` | Alongside PowerShell |
| 19 | Add `.github/CI.yml` | `.github/workflows/ci.yml` | Automated testing |
| 20 | Add `docs/` directory | `docs/` | Architecture, provider config guide |
| 21 | Add `[project.urls]` | `pyproject.toml` | Homepage, repo, docs URLs |
| 22 | Add `[project.authors]` | `pyproject.toml` | Author attribution |
| 23 | Clean TMP directory | `TMP/` | Delete from disk |

---

## 16. Risks Before Public Release

| Risk | Severity | Mitigation |
|------|----------|------------|
| No LICENSE file | **CRITICAL** | Add before any public sharing |
| Groq org ID exposed | **CRITICAL** | Delete tracked files, scrub from history if needed |
| Internal reports visible | **HIGH** | Delete — exposes development methodology, internal metrics |
| README references non-repo paths | **HIGH** | Rewrite — confusing for public contributors |
| Hardcoded developer paths | **MEDIUM** | Fix — unprofessional but not a security risk |
| No CI/CD | **MEDIUM** | Add GitHub Actions workflow |
| No CONTRIBUTING.md | **MEDIUM** | Add standard guidelines |
| PowerShell-only examples | **LOW** | Add bash equivalents |
| Empty directories | **LOW** | Clean up |
| Test count outdated in README | **LOW** | Update |

---

## 17. Final Public-Release Readiness Score

| Aspect | Score | Notes |
|--------|-------|-------|
| **Code cleanliness** | 9/10 | Excellent. 105 source files, well-organized, type-hinted, tested. Minor: empty `app/observability/` dir. |
| **Security** | 8/10 | API keys safe. One org ID leaked in tracked data files. Developer paths in source (low risk). |
| **Documentation** | 3/10 | README is internal-focused, references non-repo paths, outdated. No LICENSE, CONTRIBUTING, or SECURITY files. |
| **Reproducibility** | 7/10 | Benchmarks exist but some result files are internal. Evaluation corpus is clean. |
| **Git hygiene** | 7/10 | Good .gitignore. 14 internal reports + 3 data files tracked that shouldn't be. |
| **Model/provider transparency** | 8/10 | Configs are clean and well-documented in YAML. README needs a "Supported Providers" section. |
| **GitHub presentation** | 4/10 | Root directory cluttered with internal reports. No badges, no architecture diagram, no CI. |

**Overall readiness: 6/10**

The code is ready. The packaging is not. With the P0 fixes applied (LICENSE, delete internal reports, fix README, remove org ID), the score jumps to **8.5/10** — suitable for public release, LinkedIn, and portfolio showcases.
