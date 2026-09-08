# ARGUS — LinkedIn Material

## One-Line Description

Evidence-grounded AI research intelligence with visible verification, conflicts, and provenance.

## Short Description

ARGUS is an AI research system that makes the research process observable. Instead of hiding how an answer is produced, ARGUS shows every step: query analysis, evidence retrieval, claim verification, conflict detection, and grounded synthesis — all visible in an interactive Brain UI.

## Technical Description

ARGUS is a full-stack AI research system built with:

- **Hybrid retrieval** — BM25 + FAISS dense search with configurable fusion
- **Evidence graph** — NetworkX MultiDiGraph with 6 node types, 8 edge types
- **Deterministic verification** — Every claim checked against source evidence
- **Conflict detection** — Contradictions identified with temporal/entity/unit awareness
- **Grounded synthesis** — Answers generated from verified evidence with inline citations
- **Multi-model routing** — 6 LLM providers with call-type routing and fallback
- **Brain UI** — Interactive pipeline visualization showing how ARGUS thinks

The system is built with Python 3.11+, FastAPI, LangGraph, FAISS, and D3.js.

## Key Achievements

- **906 tests passing** with 2 pre-existing failures across retrieval, verification, graph, memory, and orchestration
- **Conflict detection** reduced false positive rate from 80% to 0% through query-aware filtering
- **Evidence traceability** — trace from any answer citation back to the source document
- **Brain UI** — 25/25 spec requirements implemented in a single-file HTML/JS/D3.js application
- **43 development phases** with detailed benchmark reports and honest infrastructure assessment

## Suggested LinkedIn Post

> I built ARGUS — an AI research system that shows you how it thinks.
>
> Most RAG systems give you an answer and hide the process. ARGUS does the opposite: every step is visible.
>
> 🔍 Query Analysis — What kind of question is this?
> 📋 Research Planning — What evidence do we need?
> 📄 Evidence Retrieval — Hybrid BM25 + dense search
> ✅ Verification — Every claim checked against sources
> ⚠️ Conflict Detection — When sources disagree, you see it
> 🧠 Grounded Synthesis — Answer from verified evidence, not model opinion
>
> The Brain UI lets you click any stage and understand what happened, why it matters, and how it affected the final answer.
>
> 906 tests. 6 LLM providers. Deterministic verification. Evidence provenance.
>
> Built with Python, FastAPI, LangGraph, FAISS, and D3.js.
>
> #AI #RAG #EvidenceBasedAI #MachineLearning #OpenSource

## Project Tags

`ai` `rag` `evidence-grounded` `verification` `conflict-detection` `llm` `retrieval` `python` `fastapi` `langgraph` `faiss` `d3.js` `open-source`

## Repository URL

https://github.com/armanojha/ARGUS

## Key Metrics for Posts

| Metric | Value |
|--------|-------|
| Tests | 906 passed, 2 pre-existing failures |
| Phases | 43 completed |
| Providers | 6 LLM providers |
| Node types | 9 pipeline stages |
| Edge types | 8 graph edge types |
| Conflict FP rate | 80% → 0% |
| Brain UI views | 10 views |
