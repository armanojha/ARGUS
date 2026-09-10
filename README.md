# ARGUS

### An Iterative, Evidence-Driven Research RAG System

> **ARGUS is not a general-purpose RAG chatbot.**
>
> It is an experimental research system built around a different idea: **retrieval should be an iterative investigation, not a single search followed by generation.**

ARGUS takes a complex question, decomposes what needs to be known, retrieves evidence using multiple retrieval strategies, evaluates that evidence, detects gaps and conflicts, adapts its research strategy when necessary, and finally produces a grounded answer with traceable evidence.

The entire process is observable through an interactive **Brain UI**, allowing the research process to be inspected rather than hidden behind a final answer.

---

## The Idea

Most RAG systems follow roughly:

```text
Question
   ↓
Retrieve
   ↓
Generate
   ↓
Answer
```

That works for straightforward questions.

But difficult research questions are rarely that simple.

A serious question may require:

* multiple searches
* different retrieval strategies
* evidence from different documents
* verification of individual claims
* identifying missing information
* resolving contradictory evidence
* revisiting the research strategy
* deciding when enough evidence has actually been gathered

ARGUS is built around this loop:

```text
                    ┌──────────────────────┐
                    │      User Query      │
                    └──────────┬───────────┘
                               ↓
                    ┌──────────────────────┐
                    │    Query Analysis    │
                    └──────────┬───────────┘
                               ↓
                    ┌──────────────────────┐
                    │   Research Planning  │
                    └──────────┬───────────┘
                               ↓
                    ┌──────────────────────┐
                    │   Evidence Retrieval │
                    └──────────┬───────────┘
                               ↓
                    ┌──────────────────────┐
                    │ Evidence Assessment  │
                    └──────────┬───────────┘
                               ↓
                    ┌──────────┴───────────┐
                    │                      │
               Gaps / Conflicts       Sufficient?
                    │                      │
                    ↓                      ↓
             Adapt Strategy            Synthesize
                    │                      │
                    └──────→ Research ←────┘
                               ↓
                    ┌──────────────────────┐
                    │ Verified Synthesis   │
                    └──────────┬───────────┘
                               ↓
                    ┌──────────────────────┐
                    │ Answer + Provenance  │
                    └──────────────────────┘
```

The important part is the **feedback loop**.

ARGUS does not treat retrieval as a one-shot operation. Research can continue when the evidence is insufficient, contradictory, or reveals a new information gap.

---

# Why ARGUS Exists

I built ARGUS as a way to move beyond learning RAG techniques individually and actually understand what happens when they are combined into a complete research pipeline.

The project became an engineering experiment around questions such as:

* How should lexical and semantic retrieval cooperate?
* How should a system decide what to search for next?
* How can retrieved evidence be evaluated before synthesis?
* How should contradictions be represented?
* How can evidence provenance survive the entire pipeline?
* How can a system adapt its research strategy based on what it discovers?
* How can the internal research process be made inspectable?
* Where should deterministic algorithms be used, and where should LLM reasoning be used?
* How can all of this remain testable and measurable?

ARGUS is the result of that experimentation.

---

# Architecture

ARGUS is organized as a stateful research pipeline rather than a simple retrieve-and-generate chain.

```text
                         ┌─────────────────────┐
                         │      Brain UI       │
                         │ Visualization /     │
                         │ Research Inspection │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │     FastAPI API     │
                         └──────────┬──────────┘
                                    │
                                    ▼
                  ┌─────────────────────────────────┐
                  │       LangGraph Orchestrator    │
                  │                                 │
                  │ Analyze → Plan → Retrieve       │
                  │       → Assess → Stop           │
                  │       → Research Again          │
                  │       → Synthesize              │
                  └───────┬───────────┬─────────────┘
                          │           │
             ┌────────────┘           └──────────────┐
             ▼                                       ▼
   ┌────────────────────┐                  ┌────────────────────┐
   │ Retrieval Pipeline │                  │ Evidence Pipeline  │
   │                    │                  │                    │
   │ BM25               │                  │ Verification       │
   │ Dense / FAISS      │                  │ Contradictions     │
   │ Hybrid Fusion      │                  │ Gap Detection      │
   │ Multi-Query        │                  │ Confidence         │
   │ Reranking          │                  │ Provenance          │
   └────────────────────┘                  └────────────────────┘
             │                                       │
             └────────────────┬──────────────────────┘
                              ▼
                   ┌──────────────────────┐
                   │ Knowledge / Memory   │
                   │                      │
                   │ Evidence Store       │
                   │ Knowledge Graph      │
                   │ Persistent Memory    │
                   └──────────────────────┘
                              │
                              ▼
                   ┌──────────────────────┐
                   │ LLM Gateway          │
                   │ Multi-provider        │
                   │ routing + fallback   │
                   └──────────────────────┘
```

---

# Research Loop

## 1. Query Analysis

The first stage determines what kind of question ARGUS is dealing with.

The query analysis layer identifies properties such as:

* query pattern
* entities
* temporal references
* research complexity
* retrieval requirements
* risk characteristics

ARGUS supports a broad set of query patterns including:

```text
exact_term
conceptual
entity_relationship
historical
comparative
causal
procedural
numerical
technical_explanation
multi_doc_synthesis
conflict
complex_research
multi_hop
absent_info
adversarial
...
```

The purpose is not simply classification.

The classification influences **how the system should investigate the question**.

---

# 2. Research Planning

Instead of immediately searching the original question, ARGUS creates a research plan.

The planner can:

* decompose complex questions
* generate targeted sub-queries
* identify required evidence
* establish research budgets
* determine iteration limits
* select retrieval strategies

For a complex question, the research plan may therefore become:

```text
Original Question
       │
       ├── What is the phenomenon?
       │
       ├── What caused it?
       │
       ├── What evidence supports it?
       │
       ├── What evidence contradicts it?
       │
       └── What information is still missing?
```

This turns retrieval into an **evidence acquisition process**.

---

# 3. Hybrid Retrieval

ARGUS combines two fundamentally different retrieval paradigms.

### BM25

Lexical retrieval is useful when exact terms matter.

It performs well for:

* names
* technical terminology
* exact phrases
* identifiers
* numerical queries

### Dense Retrieval

FAISS-based vector retrieval captures semantic similarity.

It is useful when:

* wording differs
* concepts are expressed indirectly
* semantic similarity matters more than exact terms

### Hybrid Fusion

ARGUS combines both signals rather than assuming one retrieval method is universally superior.

Conceptually:

```text
                 Query
                   │
          ┌────────┴────────┐
          ↓                 ↓
       BM25             Dense Search
          │                 │
          └────────┬────────┘
                   ↓
             Score Fusion
                   ↓
            Candidate Set
```

Fusion weights can vary according to the detected query pattern.

This allows the retrieval layer to behave differently for an exact-term lookup versus a conceptual or causal question.

---

# 4. Multi-Query Retrieval

Complex research questions often cannot be answered by one query.

ARGUS decomposes research into multiple sub-queries and executes them with bounded concurrency.

```text
Research Question
       │
       ├── Query A ──→ Retrieval
       ├── Query B ──→ Retrieval
       ├── Query C ──→ Retrieval
       └── Query D ──→ Retrieval
                         │
                         ▼
                 Evidence Pool
```

This increases coverage while allowing each query to target a specific evidence requirement.

---

# 5. Reranking

Initial retrieval is optimized for candidate generation.

It should not automatically determine what enters the final context.

ARGUS therefore separates:

```text
Candidate Generation
        ↓
Retrieval
        ↓
Reranking
        ↓
Evidence Selection
        ↓
LLM Context
```

The reranking layer is pluggable, allowing stronger rerankers to be introduced without restructuring the retrieval architecture.

---

# 6. Evidence Selection

Retrieving many relevant chunks does not mean sending all of them to the LLM.

ARGUS performs evidence selection using:

* semantic deduplication
* diversity selection
* relevance
* evidence coverage
* token-budget constraints

The objective is to construct a **small but informative evidence set** rather than blindly maximizing retrieved context.

---

# 7. Evidence Verification

Before synthesis, ARGUS evaluates whether the collected evidence actually supports the research requirements.

Evidence can be classified as:

| Status           | Meaning                                      |
| ---------------- | -------------------------------------------- |
| **Supported**    | Evidence directly supports the claim         |
| **Partial**      | Evidence provides incomplete support         |
| **Contradicted** | Evidence conflicts with the claim            |
| **Unsupported**  | Sufficient supporting evidence was not found |

Verification considers factors such as:

* evidence coverage
* source quality
* cross-source agreement
* temporal relevance
* confidence

This separates **retrieval relevance** from **evidence validity**.

A chunk can be relevant to a question without actually proving the claim being investigated.

---

# 8. Contradiction Detection

Contradiction is one of the more difficult problems in evidence-grounded systems.

ARGUS does not simply mark two different statements as contradictory.

The contradiction pipeline considers:

### Entity overlap

Are both sources talking about the same entity?

### Metric overlap

Are they actually discussing the same quantity?

### Numerical differences

Do numerical values disagree after accounting for units?

```text
$2.1 billion
       vs
$2,100 million
```

These should not automatically become contradictions.

### Temporal context

Two different values may both be correct if they refer to different periods.

```text
Revenue in 2022: $4B
Revenue in 2025: $7B
```

Different ≠ contradictory.

### Conflict categories

ARGUS distinguishes between:

```text
GENUINE_CONTRADICTION
DIFFERENT_TIMEFRAME
DIFFERENT_SOURCE
POSSIBLE_CONTRADICTION
IRRELEVANT_DIFFERENCE
```

This is important because a research system should not treat every disagreement between documents as a factual contradiction.

---

# 9. Query-Aware Conflict Filtering

Not every conflict discovered in a corpus matters to the current question.

ARGUS can therefore evaluate conflict signals against the query context.

For example:

```text
Corpus
 ├── Conflict A → relevant
 ├── Conflict B → irrelevant
 ├── Conflict C → historical
 └── Conflict D → relevant
```

Only conflicts relevant to the research question should influence the research process.

This reduces the risk of derailing research because of unrelated disagreements elsewhere in the evidence set.

---

# 10. Adaptive Research

This is one of the core ideas of ARGUS.

A fixed RAG pipeline behaves like:

```text
Retrieve → Retrieve → Retrieve
```

ARGUS can instead change **what it does next based on what it discovers**.

The research strategy is represented explicitly and snapshotted across iterations.

```text
Iteration 1
───────────
Strategy A
   ↓
Retrieve
   ↓
Assess
   ↓
Discover:
  • contradiction
  • evidence gap
  • retrieval gain stalled
   ↓
Strategy Mutation
   ↓
Iteration 2
───────────
Strategy B
```

### Conflict-driven adaptation

When unresolved critical contradictions remain, ARGUS can prioritize a contradiction-resolution query in the next research iteration.

```text
Contradiction detected
        ↓
Identify conflict
        ↓
Generate resolution query
        ↓
Prioritize next retrieval
```

### Gain-stall escalation

When additional retrieval produces diminishing information gain, ARGUS can increase retrieval depth for the next iteration, subject to a cap.

```text
Low marginal gain
       ↓
Increase retrieval budget
       ↓
Search deeper
```

### Gap prioritization

When the evidence assessor identifies a high-priority information gap, the corresponding research query can be moved ahead of lower-priority work.

```text
Evidence gap
     ↓
Gap priority
     ↓
Next research query
```

The important distinction is that these are **runtime strategy mutations**, not merely configuration options.

---

# 11. Research Strategy State

ARGUS explicitly represents research strategy rather than hiding it inside individual nodes.

A strategy can contain:

```text
retrieval mode
queries
source priorities
verification depth
iteration budget
evidence budget
mutation history
```

Each iteration can therefore be inspected as a state transition:

```text
Strategy₀
   │
   │ conflict detected
   ▼
Strategy₁
   │
   │ evidence gain stalled
   ▼
Strategy₂
```

This makes the research loop observable and testable.

---

# 12. Research Sufficiency & Stopping

More retrieval is not automatically better.

ARGUS evaluates whether research should continue.

Stopping decisions can depend on:

* evidence sufficiency
* claim support
* unresolved contradictions
* marginal information gain
* research budget
* iteration limits
* user-directed stopping

The goal is:

> **Continue when additional research is justified. Stop when it isn't.**

This creates an explicit research-control problem rather than an arbitrary fixed number of retrieval passes.

---

# 13. Grounded Synthesis

Once research reaches a sufficient state, ARGUS synthesizes the answer from the accumulated evidence.

The synthesis layer is designed around:

* evidence grounding
* citation traceability
* conflict awareness
* safe degradation
* claim-level support

The desired flow is:

```text
Evidence
   ↓
Verified Claims
   ↓
Conflict Awareness
   ↓
Synthesis
   ↓
Citations
   ↓
Grounding Validation
```

The final answer is therefore not just generated from retrieved text.

It is generated from the **research state produced by the investigation**.

---

# 14. Reasoning & Evidence Graph

ARGUS maintains a graph representation of the research process.

The graph can represent concepts such as:

```text
Query
Research Task
Hypothesis
Evidence
Claim
Counterclaim
Inference
Decision
Answer
```

Alongside the underlying evidence graph containing entities, claims, events, documents, chunks, and sources.

A simplified reasoning chain can look like:

```text
QUERY
  │
  ▼
RESEARCH TASK
  │
  ▼
HYPOTHESIS
  │
  ├───────────────┐
  ▼               ▼
EVIDENCE       COUNTERCLAIM
  │               │
  └───────┬───────┘
          ▼
       INFERENCE
          │
          ▼
       DECISION
          │
          ▼
        ANSWER
```

This makes it possible to inspect not only **what evidence was retrieved**, but also how that evidence contributed to the research process.

---

# 15. Provenance

ARGUS maintains evidence provenance through the pipeline.

A citation can be traced through:

```text
Answer
  ↓
Claim
  ↓
Evidence
  ↓
Chunk
  ↓
Document
  ↓
Source
```

This is fundamental to the system's design.

The goal is not merely to produce citations.

The goal is to preserve the **lineage of evidence** behind the answer.

---

# 16. Knowledge Graph

ARGUS uses a NetworkX-based evidence graph containing:

### Node types

```text
Entity
Claim
Event
Document
Chunk
Source
```

### Relationship types

Examples include:

```text
supports
contradicts
derived_from
relates_to
```

The graph provides a structured representation of relationships between evidence and extracted knowledge.

It also gives the Brain UI something substantially richer to visualize than a simple document list.

---

# 17. Memory

ARGUS includes a persistent multi-layer memory architecture backed by SQLite.

Memory can preserve information across research sessions and support:

* previous research context
* memory promotion
* versioning
* reuse of prior information

Memory is deliberately separated from the core retrieval pipeline so that the research engine does not depend on persistent memory to function.

---

# 18. Multi-Provider LLM Gateway

ARGUS does not hard-code the research system around one LLM provider.

The LLM Gateway separates:

```text
Research Logic
      ↓
LLM Gateway
      ↓
Provider Router
      ↓
Model
```

Different call types can be routed independently:

```text
Query Analysis
Research Planning
Evidence Extraction
Verification
Synthesis
```

Provider fallback chains provide resilience when a provider is unavailable or rate-limited.

This also allows the orchestration layer to remain independent from individual model providers.

---

# 19. Configuration Modes

ARGUS provides explicit operating modes.

| Mode       | Purpose                                                              |
| ---------- | -------------------------------------------------------------------- |
| `baseline` | Historical/default behavior                                          |
| `research` | Enables adaptive research                                            |
| `verified` | Enables verification-oriented capabilities                           |
| `full`     | Combines research, verification, memory and multi-agent capabilities |

Explicit configuration overrides can take precedence over mode defaults.

Experimental capabilities such as multimodal processing, BGE-M3 retrieval, and Obsidian integration remain opt-in rather than silently changing the baseline system.

---

# The Brain UI

The Brain UI is not intended to be another chatbot interface.

It exists to answer:

> **"What did ARGUS actually do to arrive at this answer?"**

The interface exposes the internal research process through interactive visualization.

### Pipeline

Shows the progression of the research state.

### Node Inspector

Selecting a node exposes information about:

* what stage executed
* what it received
* what it produced
* why the stage exists
* how it contributed to the research process

### Evidence Trace

Allows evidence to be followed back toward its source.

### Conflict Visualization

Shows detected conflicts and their associated evidence.

### Reasoning Trace

Displays the relationship between:

```text
Question
→ Research Tasks
→ Hypotheses
→ Evidence
→ Claims
→ Counterclaims
→ Inferences
→ Decisions
→ Answer
```

### Knowledge Graph

Provides an interactive representation of entities, claims, events and evidence relationships.

The objective is to make the system **inspectable rather than opaque**.

---

# Evaluation

ARGUS includes a fixed end-to-end evaluation benchmark designed to measure research-system behavior rather than simply counting tests.

The deterministic benchmark uses:

* **38 evaluation cases**
* **12-document fixed corpus**
* real BM25 retrieval
* real FAISS retrieval
* MiniLM embeddings
* deterministic contradiction detection
* deterministic evidence-abstention detection
* isolated temporary storage

Current benchmark measurements include approximately:

| Metric                      |      Result |
| --------------------------- | ----------: |
| Recall@8                    |  **0.9865** |
| Precision@8                 |  **0.1546** |
| Gold-fact coverage          |  **~0.904** |
| Contradiction recall        |  **1.0000** |
| Contradiction precision     | **~0.0789** |
| Abstention-trigger accuracy |  **1.0000** |

These numbers should **not** be interpreted as proof that ARGUS is generally intelligent.

The benchmark is intentionally fixed and deterministic. It measures specific components under controlled conditions.

Live LLM evaluation is separated because provider availability, latency, rate limits and model behavior introduce additional variables.

---

# Engineering Philosophy

ARGUS follows several design principles.

### 1. Deterministic where possible

If a problem can be reliably solved with deterministic logic, it should not automatically become an LLM call.

### 2. LLMs where semantic reasoning is useful

Model calls are used where semantic interpretation or generation provides genuine value.

### 3. Retrieval ≠ verification

Finding relevant evidence and determining whether it supports a claim are treated as different problems.

### 4. Relevance ≠ truth

A highly relevant document can still contain information that is outdated, incomplete or contradictory.

### 5. More evidence ≠ better research

The system needs mechanisms for evidence selection, sufficiency and stopping.

### 6. Research should be stateful

If the system discovers something important, that discovery should be capable of influencing what happens next.

### 7. Observability is part of the architecture

The research process should be inspectable, not reconstructed after the fact.

### 8. Measure before claiming

Benchmark numbers, test results and capabilities are documented separately from aspirational goals.

---

# What Makes ARGUS Different?

ARGUS is **not** trying to compete with general-purpose AI assistants.

It is an engineering exploration of what happens when a RAG system is treated as a **research process** rather than a retrieval component.

The central distinction is:

```text
Traditional RAG

Question
   ↓
Retrieve
   ↓
Generate


ARGUS

Question
   ↓
Analyze
   ↓
Plan
   ↓
Retrieve
   ↓
Assess
   ↓
     ┌───────────────┐
     │ Enough?       │
     └───────┬───────┘
             │ No
             ▼
       Detect gaps /
       conflicts / gain
             │
             ▼
       Adapt strategy
             │
             └──────→ Retrieve again
                          │
                         Yes
                          ↓
                      Verify
                          ↓
                      Synthesize
                          ↓
                  Explain + Trace
```

The difference is not simply **"more RAG components."**

It is the feedback loop connecting them.

---

# Project Structure

```text
ARGUS/
│
├── app/
│   ├── api/                    # FastAPI endpoints
│   ├── config.py               # Configuration and operating modes
│   │
│   ├── evidence/               # Evidence storage and provenance
│   ├── graph/                  # Evidence / reasoning graph
│   ├── ingestion/              # Document ingestion pipeline
│   ├── llm_gateway/            # Multi-provider LLM routing
│   ├── memory/                 # Persistent memory
│   │
│   ├── orchestration/          # LangGraph research workflow
│   ├── retrieval/              # BM25 + dense + hybrid retrieval
│   ├── reranking/              # Candidate reranking
│   ├── verification/           # Claim/evidence verification
│   │
│   └── ui/
│       └── brain/              # Interactive Brain UI
│
├── benchmarks/
│   ├── eval_data/              # Fixed evaluation corpus
│   └── e2e_intelligence.py     # End-to-end benchmark
│
├── configs/
│   ├── providers.yaml
│   ├── model_policy.yaml
│   └── retrieval_policy.yaml
│
├── tests/                      # Unit + integration tests
├── docs/                       # Architecture and correctness docs
│
├── knowledge_base/             # Local research corpus
│
├── pyproject.toml
└── README.md
```

---

# Tech Stack

| Layer             | Technology                      |
| ----------------- | ------------------------------- |
| Language          | Python                          |
| API               | FastAPI                         |
| Orchestration     | LangGraph                       |
| Lexical Retrieval | BM25                            |
| Dense Retrieval   | FAISS + Sentence Transformers   |
| Embeddings        | `all-MiniLM-L6-v2`              |
| Reranking         | Pluggable reranker architecture |
| Evidence Store    | SQLite                          |
| Knowledge Graph   | NetworkX                        |
| Memory            | SQLite                          |
| Frontend          | HTML / JavaScript / D3 / Canvas |
| Testing           | pytest                          |
| Linting           | Ruff                            |
| CI                | GitHub Actions                  |
| LLMs              | Multi-provider gateway          |

---

# Installation

### Requirements

* Python 3.11+
* At least one supported LLM provider API key for live research
* Git

### Clone

```bash
git clone https://github.com/armanojha/ARGUS.git
cd ARGUS
```

### Environment

```bash
python -m venv .venv
```

Windows:

```bash
.\\.venv\\Scripts\\activate
```

Install:

```bash
pip install -e ".[core,retrieval,graph,dev-test]"
```

Configure environment variables:

```bash
cp .env.example .env
```

Then add the required provider credentials.

---

# Running ARGUS

Start the API:

```bash
uvicorn app.api.main:app --reload
```

Then open:

```text
http://localhost:8000/brain
```

The Brain UI provides the primary interface for inspecting ARGUS's research pipeline.

---

# Testing

Run the complete test suite:

```bash
pytest tests/ -v
```

Run with coverage:

```bash
pytest tests/ --cov=app --cov-report=term-missing
```

Lint:

```bash
ruff check .
```

---

# End-to-End Evaluation

Run the deterministic intelligence benchmark:

```bash
python benchmarks/e2e_intelligence.py
```

Specify retrieval depth:

```bash
python benchmarks/e2e_intelligence.py --top-k 8
```

The benchmark operates against an isolated evaluation corpus and does not modify the normal knowledge base.

---

# Current Status

ARGUS is an **experimental research and engineering project**.

The system is functional enough to demonstrate the core research architecture, but it is not presented as a production-ready autonomous research platform.

The project is intentionally being developed around measurable behavior rather than feature count.

---

# Known Limitations

ARGUS still has important limitations.

### Contradiction precision

The contradiction detector currently has high recall on the fixed benchmark but substantially lower precision.

This is a known research problem rather than a metric being hidden.

### Live evaluation

Full LLM evaluation is more variable than deterministic evaluation because of:

* provider availability
* model behavior
* rate limits
* latency
* token usage

### Embedding model

`all-MiniLM-L6-v2` was selected for practicality and speed rather than maximum retrieval quality.

### Production hardening

ARGUS is not currently designed for:

* large-scale distributed deployment
* enterprise authentication
* horizontal scaling
* production-grade observability infrastructure
* guaranteed provider availability

### Experimental integrations

Some capabilities remain intentionally optional and are not required by the core research loop.

---

# Research Direction

The long-term direction of ARGUS is not simply to keep adding RAG components.

The interesting problem is:

> **How far can an evidence-grounded retrieval system go when research itself becomes an adaptive control loop?**

Areas of continued experimentation include:

* stronger semantic contradiction verification
* better research-policy adaptation
* improved evidence selection
* stronger reranking
* deeper claim-level verification
* richer reasoning traces
* improved evaluation methodology
* more robust abstention
* better source-quality modeling
* more informative research visualization

---

# Why the Project Is Called ARGUS

ARGUS is named after **Argus Panoptes**, the many-eyed watcher from Greek mythology. i remembered this name from reading certain novel
and then after doing some research I found some RAG based projects with th esame name so i chose this as well. plus its'cool. 
The metaphor fits the project's purpose:

**ARGUS observes the research process from multiple perspectives — retrieval, evidence, conflicts, verification, reasoning and provenance — rather than looking only at the final answer.**

---

# A Note on the Project

ARGUS started as an attempt to understand RAG by building one from the ground up.

It evolved into something more specific:

**an experiment in turning retrieval-augmented generation into an observable, iterative research process.**

The project is therefore less about claiming that every component is state-of-the-art and more about exploring how advanced RAG techniques interact when placed inside a single research loop.

That is the problem ARGUS is built to investigate.
And I am going to continue this research as well in the future.

---

# License

MIT License.

See [`LICENSE`](LICENSE).

---

## Documentation

* [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — Detailed system architecture
* [`docs/CORRECTNESS.md`](docs/CORRECTNESS.md) — Correctness and engineering evolution
* [`CHANGELOG.md`](CHANGELOG.md) — Project history
* [`CONTRIBUTING.md`](CONTRIBUTING.md) — Contribution guidelines

---

<p align="center">

**ARGUS**

*Research should be a process, not a single retrieval.*

</p>
