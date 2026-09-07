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

- Z.ai (Zhipu AI) — Occasional timeouts
- Zen (OpenCode) — 403 errors on some models
- Cerebras — Requires API key not always available

**Mitigation:** ARGUS routes through fallback chains. When a provider fails, the next in the chain is used.

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

## Brain UI Limitations

### Single-File Architecture

The Brain UI is a 3,993-line single HTML file. This was intentional for simplicity (no build step, no framework), but:

- Adding features requires modifying a large file
- No component reuse across views
- No TypeScript type safety

**Future work:** Component modularization if the UI grows significantly.

### No Real-Time Collaboration

The Brain UI is single-user. There is no multi-user or collaboration support.

### Canvas Rendering Limits

The graph uses Canvas 2D rendering. For graphs with thousands of nodes, performance may degrade. The current evidence graph typically has tens to low hundreds of nodes, which works well.

## Experimental Components

The following components were evaluated but intentionally not promoted to production:

| Component | Status | Reason |
|-----------|--------|--------|
| BGE-M3 embeddings | Experimental, behind `ARGUS_BGE_M3_ENABLED` | Requires significant VRAM; tested but not default |
| Multi-agent debate | Behind `ARGUS_MULTIAGENT_ENABLED` | Validated but adds latency |
| Chart extraction | Behind `ARGUS_MULTIMODAL_CHART_EXTRACTION_ENABLED` | Placeholder for vision model |
| Two-pass synthesis | Behind `ARGUS_VERIFIED_SYNTHESIS_ENABLED` | Needs stable providers to validate |

These are not bugs — they are intentionally feature-flagged experimental work.

## What ARGUS Does NOT Do

- **Real-time web search** — ARGUS queries a local document corpus, not the internet
- **Image understanding** — OCR extracts text from images, but ARGUS does not "see" images
- **Perfect accuracy** — ARGUS verifies evidence but can still make mistakes
- **Zero hallucinations** — Grounded synthesis reduces hallucination but does not eliminate it
- **Commercial production use** — ARGUS is a research project, not a production system

## Validation Honesty

Some benchmark results were affected by provider instability:

- **Phase 36 (Clean Provider Benchmark)** — Infrastructure-limited; all runs had provider fallbacks
- **Phase 37 (LLM Call Minimization)** — Infrastructure-blocked; could not run clean experiments
- **Phase 40 (Conflict-Aware Synthesis)** — Provider contamination prevented trustworthy ablation

These results were not used to justify production changes. Core components (retrieval, verification, conflict detection) are independently validated with deterministic tests.
