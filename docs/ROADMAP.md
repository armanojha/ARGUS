# ARGUS Roadmap

## Now (Complete)

These capabilities are implemented, tested, and available:

- **Hybrid Retrieval** — BM25 + FAISS with configurable fusion weights
- **Adaptive Research** — 20 query patterns with evidence need planning
- **Evidence Verification** — Deterministic claim-support checking with confidence scoring
- **Conflict Detection** — Pairwise contradiction detection with temporal/entity/metric awareness
- **Query-Aware Filtering** — Suppresses irrelevant conflicts based on user intent
- **Grounded Synthesis** — Answer generation with citations, conflict acknowledgment, safe degradation
- **Multi-Model Routing** — 6 LLM providers with per-call-type routing and fallback
- **Persistent Memory** — 6-layer SQLite memory with promotion and versioning
- **Brain UI** — Interactive pipeline visualization, node inspection, evidence tracing, demo mode
- **Evidence Traceability** — Every citation traces back to source document, chunk, and retrieval metadata

## Next

These are natural extensions of the existing architecture:

### Streaming Research Events

Add SSE/WebSocket endpoints for real-time stage-by-stage updates. This would allow the Brain UI to show actual backend progress instead of simulating it.

**Complexity:** Medium. Requires async event emission from the LangGraph state machine.

### Richer Per-Stage Timing

Expose start/end timestamps for individual pipeline stages. Currently only aggregate telemetry is available.

**Complexity:** Low. Instrumentation around existing node functions.

### Brain UI Component Modularization

Break the 3,996-line single-file HTML into reusable components. This would make the UI easier to maintain and extend.

**Complexity:** Medium. Requires a build step or module system.

### Stronger Synthesis Validation

Add more rigorous post-synthesis checks: citation accuracy, claim-evidence alignment, conflict acknowledgment completeness.

**Complexity:** Low-Medium. Extensions to existing deterministic checks.

### Research Session Persistence

Persist research results so users can revisit previous queries and answers.

**Complexity:** Medium. Requires schema design and storage layer.

## Later

These require more significant architectural work:

### Advanced Entity Linking

Improve entity extraction and linking across documents. Currently basic keyword matching; could benefit from NER models or knowledge base integration.

**Complexity:** High. Requires NER infrastructure or external knowledge base.

### Graph-Assisted Retrieval

Use the evidence graph to guide retrieval. Instead of only vector/BM25 search, traverse entity relationships to find relevant evidence.

**Complexity:** High. Requires graph traversal integration with retrieval pipeline.

### Additional Multimodal Workflows

Extend beyond OCR to image understanding, chart interpretation, and table analysis.

**Complexity:** High. Requires vision models or specialized parsers.

### Multi-Language Support

Extend retrieval and synthesis to non-English documents.

**Complexity:** High. Requires multilingual embeddings and models.

## What Will NOT Be Built

These were considered and rejected:

| Feature | Reason |
|---------|--------|
| Replace BM25 with pure vector search | Hybrid outperforms both individually |
| Replace embeddings with BGE-M3 as default | Requires significant VRAM; not default-worthy |
| Autonomous model selection | Explicit configuration is more predictable |
| Commercial production deployment | ARGUS is a research project |
| Real-time web search | ARGUS queries a local corpus by design |
| Perfect accuracy / zero hallucination | Impossible to guarantee; grounding reduces but doesn't eliminate |
