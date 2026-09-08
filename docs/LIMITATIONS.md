# ARGUS Known Limitations

ARGUS is a research system built and validated through 43 development phases. This document honestly describes what works, what doesn't, and what's intentionally left for future work.

## Provider Infrastructure

### Free-Tier Rate Limits

ARGUS supports 6 LLM providers, most with free tiers. These limits affect:

- **Live E2E synthesis latency** — Rapid sequential queries exhaust rate limits
- **Benchmark cleanliness** — Provider fallbacks contaminate controlled experiments
- **Demo reliability** — Live demos may hit rate limits during presentations

**Mitigation:** Demo Mode provides a reliable offline demonstration. Core retrieval, verification, and conflict detection are independently tested without LLM dependency.

### Provider Instability

Some providers have inconsistent availability:

- **Z.ai (Zhipu AI)** — Occasional timeouts
- **Zen (OpenCode)** — 403 errors on some models
- **Cerebras** — Requires API key not always available
- **Groq** — 8K tokens per minute limit on free tier

**Mitigation:** ARGUS routes through fallback chains. When a provider fails, the next in the chain is used.

### Provider-Dependent Evaluation

Some benchmarks could not be run cleanly due to provider instability:

- **Phase 36 (Clean Provider Benchmark)** — All runs had provider fallbacks
- **Phase 37 (LLM Call Minimization)** — Infrastructure-blocked
- **Phase 40 (Conflict-Aware Synthesis)** — Provider contamination

These results were not used to justify production changes.

## Backend Limitations

### No Streaming Research Events

The backend processes queries synchronously. There are no WebSocket or SSE endpoints for real-time stage-by-stage updates.

**Impact:** The Brain UI's progressive pipeline building is a UI simulation, not actual backend streaming.

**Future work:** Add SSE endpoints for research event streaming.

### Per-Stage Timing Not Exposed

The backend does not expose start/end timestamps for individual pipeline stages. Only aggregate telemetry (total duration, total calls, total tokens) is available.

**Impact:** The research timeline in the Brain UI is approximate, derived from LLM call routing decisions.

**Future work:** Add per-stage timing instrumentation.

### No Persistent Research History

Research results are not persisted. Each query is stateless. There is no "research history" beyond the current session.

**Future work:** Add research session persistence.

### Synchronous Ingestion

Document ingestion processes files sequentially. Large corpora may take significant time to ingest.

**Future work:** Parallel ingestion for large corpora.

## Brain UI Limitations

### Single-File Architecture

The Brain UI is a 3,996-line single HTML file. This was intentional for simplicity (no build step, no framework), but:

- Adding features requires modifying a large file
- No component reuse across views
- No TypeScript type safety

**Future work:** Component modularization if the UI grows significantly.

### No Real-Time Collaboration

The Brain UI is single-user. There is no multi-user or collaboration support.

### Canvas Rendering Limits

The graph uses Canvas 2D rendering. For graphs with thousands of nodes, performance may degrade. The current evidence graph typically has tens to low hundreds of nodes, which works well.

### Demo Mode is Static

Demo mode shows a pre-recorded research result. It does not execute the actual pipeline. This is clearly labeled "DEMO MODE — Pre-recorded data."

## Experimental Components

The following components were evaluated but intentionally not promoted to production:

| Component | Status | Reason |
|-----------|--------|--------|
| BGE-M3 embeddings | Experimental, behind `ARGUS_BGE_M3_ENABLED` | Requires significant VRAM; tested but not default |
| Multi-agent debate | Behind `ARGUS_MULTIAGENT_ENABLED` | Validated but adds latency |
| Chart extraction | Behind `ARGUS_MULTIMODAL_CHART_EXTRACTION_ENABLED` | Placeholder for vision model |
| Two-pass synthesis | Behind `ARGUS_VERIFIED_SYNTHESIS_ENABLED` | Needs stable providers to validate |

These are not bugs — they are intentionally feature-flagged experimental work.

## Evaluation Limitations

### No Human Evaluation

Answer quality has not been evaluated by human annotators. All evaluation is automated (citation grounding, claim support, conflict detection).

### Synthetic Gold Standards

Relevance judgments and gold-standard answers were created synthetically, not by domain experts.

### Limited Corpus Size

Evaluation used corpora of 12-20 documents. Real-world performance on larger corpora has not been measured.

### Provider-Contaminated Benchmarks

Some benchmark results were affected by provider fallbacks and could not be trusted as clean measurements.

## What ARGUS Does NOT Do

- **Real-time web search** — ARGUS queries a local document corpus, not the internet
- **Image understanding** — OCR extracts text from images, but ARGUS does not "see" images
- **Perfect accuracy** — ARGUS verifies evidence but can still make mistakes
- **Zero hallucinations** — Grounded synthesis reduces hallucination but does not eliminate it
- **Commercial production use** — ARGUS is a research project, not a production system
- **Streaming research events** — Backend processes queries synchronously
- **Per-stage timing** — Only aggregate telemetry is available
- **Research history** — Each query is stateless
