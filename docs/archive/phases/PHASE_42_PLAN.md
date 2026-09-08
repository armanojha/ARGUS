# Phase 42 — ARGUS Brain UI Finalization

## Objective

Build and polish the FINAL ARGUS Brain UI so that a third person can open the project and immediately understand:
- What ARGUS is
- How ARGUS works
- Why each component exists
- How information flows through the system
- What evidence was retrieved
- How evidence was verified
- How conflicts were detected
- How the final answer was produced

## Existing UI Audit

### Current Architecture
- **Single file:** `app/ui/brain/index.html` (2,142 lines)
- **Framework:** Vanilla JavaScript, no build step
- **Graph:** D3.js v7 force-directed, Canvas 2D rendering
- **Styling:** CSS custom properties, dark theme
- **Served:** FastAPI static file at `/brain`

### Current Views (8)
1. **Brain** — Force-directed knowledge graph (Canvas + D3)
2. **Research** — Chat-style Q&A with citations
3. **Knowledge Base** — Upload/ingest documents
4. **Memory** — Persistent machine memory
5. **Documents** — Browse graph documents
6. **Evidence** — Knowledge graph summary
7. **Obsidian Brain** — Obsidian vault
8. **Settings** — System status

### Current Strengths
- Clean dark theme with glassmorphism
- Force-directed graph with clustering
- Node selection with details panel
- Document viewer modal
- Search with focus-on-select
- Keyboard navigation (arrow keys between neighbors)
- Research chat with citations and source trails

### Current Gaps (per spec)
- No research pipeline visualization
- No node inspector with explanations
- No evidence/citation tracing
- No conflict visualization
- No architecture view
- No demo mode
- No research timeline
- No "why this exists" explanations

## Reference UI Analysis (Smart Second Brain)

### Key Lessons (without copying)
1. **Pixi.js WebGL** — Batched sprite rendering for performance
2. **Topic hulls** — Colored convex hulls behind clusters
3. **Lasso selection** — Freehand region selection
4. **Stale-while-revalidate** — Show cached data immediately
5. **Granularity slider** — Adjust clustering density
6. **Dashed vs solid edges** — Inferred vs explicit connections
7. **Right-click context menu** — Focus, open, hide actions
8. **Immerse/drill-down** — Re-segment regions

### What ARGUS Does Better
- Richer edge semantics (supports/contradicts)
- Confidence visualization
- Verification status display
- Keyboard arrow-key navigation
- Document viewer modal
- Research chat with citations
- Self-contained dark theme
- Multi-brain architecture

## Final Architecture

### Design Decision: Evolve, Don't Rewrite

The existing single-file architecture is retained. New features are added as additions to the existing HTML file. This follows the spec: "Do NOT rewrite the entire frontend merely for aesthetics."

### New View Structure

```
Sidebar Navigation:
├── Research Pipeline (NEW - default view)
│   └── Shows ARGUS processing flow for current/recent query
├── Knowledge Graph (RENAMED from Brain)
│   └── Force-directed evidence graph
├── Research Chat (EXISTING)
│   └── Chat-style Q&A
├── Knowledge Base (EXISTING)
│   └── Upload/ingest
├── Memory (EXISTING)
│   └── Persistent memory
├── Documents (EXISTING)
│   └── Browse documents
├── Evidence (EXISTING)
│   └── Graph summary
├── Architecture (NEW)
│   └── System component diagram
├── Obsidian Brain (EXISTING)
│   └── Vault integration
└── Settings (EXISTING)
    └── System status
```

### Pipeline View Architecture

The Pipeline view is the new default. It shows:

```
┌─────────────────────────────────────────────┐
│  ARGUS BRAIN                                │
│  Evidence-Grounded Research Intelligence    │
├─────────────────────────────────────────────┤
│  [New Research] [Search] [History] [Status] │
├─────────────────────────────────────────────┤
│                                             │
│  ┌─────────────────────────────────────┐    │
│  │  🔍 QUERY                           │    │
│  │  "Compare revenue growth of Acme    │    │
│  │   and Globex in 2024 vs 2025"       │    │
│  └─────────────────────────────────────┘    │
│                    │                         │
│                    ▼                         │
│  ┌─────────────────────────────────────┐    │
│  │  📊 ANALYSIS                        │    │
│  │  Pattern: comparison                │    │
│  │  Entities: Acme, Globex             │    │
│  │  Time: 2024-2025                    │    │
│  └─────────────────────────────────────┘    │
│                    │                         │
│                    ▼                         │
│  ┌─────────────────────────────────────┐    │
│  │  📋 RESEARCH PLAN                   │    │
│  │  Strategy: hybrid (BM25 + dense)    │    │
│  │  Sub-queries: 3                     │    │
│  │  Iteration budget: 3                │    │
│  └─────────────────────────────────────┘    │
│                    │                         │
│          ┌─────────┼─────────┐              │
│          ▼         ▼         ▼              │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐    │
│  │ EVIDENCE │ │ EVIDENCE │ │ EVIDENCE │    │
│  │ Chunk 1  │ │ Chunk 2  │ │ Chunk 3  │    │
│  │ Score: 92│ │ Score: 87│ │ Score: 81│    │
│  └──────────┘ └──────────┘ └──────────┘    │
│          │         │         │              │
│          └─────────┼─────────┘              │
│                    ▼                         │
│  ┌─────────────────────────────────────┐    │
│  │  ✅ VERIFICATION                    │    │
│  │  Coverage: 85%                      │    │
│  │  Supported claims: 4                │    │
│  │  Confidence: HIGH                   │    │
│  └─────────────────────────────────────┘    │
│                    │                         │
│                    ▼                         │
│  ┌─────────────────────────────────────┐    │
│  │  ⚠️ CONFLICT DETECTION              │    │
│  │  1 conflict found                   │    │
│  │  Type: DIFFERENT_TIMEFRAME          │    │
│  │  Status: ACKNOWLEDGED               │    │
│  └─────────────────────────────────────┘    │
│                    │                         │
│                    ▼                         │
│  ┌─────────────────────────────────────┐    │
│  │  🧠 SYNTHESIS                       │    │
│  │  Method: grounded                   │    │
│  │  Citations: 4                       │    │
│  │  Grounding: VERIFIED                │    │
│  └─────────────────────────────────────┘    │
│                    │                         │
│                    ▼                         │
│  ┌─────────────────────────────────────┐    │
│  │  💬 ANSWER                          │    │
│  │  Acme Corporation reported revenue  │    │
│  │  of $45B in 2024 and $52B in 2025,  │    │
│  │  representing 15.6% growth [1][2].  │    │
│  │  Globex reported $12B in 2024 and   │    │
│  │  $14B in 2025, representing 16.7%   │    │
│  │  growth [3][4].                      │    │
│  └─────────────────────────────────────┘    │
│                                             │
├─────────────────────────────────────────────┤
│  Research Timeline                          │
│  00:00 Query analyzed                       │
│  00:02 Research plan generated              │
│  00:05 Retrieval completed (3 chunks)       │
│  00:08 Evidence verified                    │
│  00:10 Conflict detected                    │
│  00:12 Synthesis completed                  │
└─────────────────────────────────────────────┘
```

### Node Types (Pipeline)

| Node | Color | Icon | Shape |
|------|-------|------|-------|
| Query | #6ea8fe | 🔍 | Rounded rect |
| Analysis | #8b9cf9 | 📊 | Rounded rect |
| Plan | #a78bfa | 📋 | Rounded rect |
| Sub-query | #c4b5fd | 🔎 | Small rect |
| Evidence | #34d399 | 📄 | Card |
| Verification | #fbbf24 | ✅ | Rounded rect |
| Conflict | #f87171 | ⚠️ | Rounded rect |
| Synthesis | #f472b6 | 🧠 | Rounded rect |
| Answer | #6ea8fe | 💬 | Large card |

### Inspector System

Every node type gets a specialized inspector with:

1. **WHAT IS THIS?** — Simple explanation
2. **WHY DOES ARGUS NEED IT?** — Purpose explanation
3. **WHAT DID IT RECEIVE?** — Input data
4. **WHAT DID IT PRODUCE?** — Output data
5. **HOW DOES IT AFFECT THE FINAL ANSWER?** — Impact

### Evidence Traceability

Click an answer citation to trace:
```
Answer Claim → Citation → Evidence Chunk → Document → Source
```

### Conflict Visualization

Show conflict nodes with:
- Conflicting evidence (Source A vs Source B)
- Conflict type (GENUINE_CONTRADICTION, DIFFERENT_TIMEFRAME, etc.)
- Relevance to query
- Detection status
- Safe resolution rules

### Architecture View

Show ARGUS components as a flow diagram:
```
Query → Query Analysis → Policy Router → Retrieval → Planner → Reranker → Evidence Verification → Conflict Detection → Synthesis → Evaluation → Answer
```

Clicking a component explains:
- What it does
- Why it exists
- What enters it
- What leaves it

### Demo Mode

Static recorded research result for offline demonstration. Clearly labeled "DEMO MODE". Showcases:
- Query
- Retrieval
- Evidence
- Verification
- Conflict
- Synthesis
- Answer

## Implementation Plan

### Step 1: Add Pipeline Data Model
- Add pipeline node/edge types to JavaScript
- Add pipeline state management
- Add pipeline data transformation from OrchestrationResult

### Step 2: Add Pipeline Canvas Renderer
- Vertical flow layout (top-to-bottom)
- Stage-based horizontal alignment
- Animated edges for active processing
- Node states (idle, processing, complete, warning)

### Step 3: Add Pipeline View
- New section in HTML
- Pipeline canvas container
- Stage labels
- Timeline display
- Empty state with example queries

### Step 4: Enhance Node Inspector
- Add "WHY THIS EXISTS" sections
- Add technical detail toggle
- Add evidence tracing
- Add citation linking

### Step 5: Add Conflict Visualization
- Conflict node rendering
- Conflict type badges
- Source comparison
- Resolution status

### Step 6: Add Architecture View
- Component flow diagram
- Component descriptions
- Input/output explanation

### Step 7: Add Demo Mode
- Static recorded data
- Demo mode indicator
- Pre-loaded research result

### Step 8: Improve States
- Empty state with example queries
- Error state with retry
- Loading state with progress
- Provider status display

### Step 9: Visual QA
- Spacing/alignment check
- Typography check
- Graph readability
- Inspector usability
- Responsive behavior

### Step 10: Documentation
- README update
- Architecture diagram
- Brain UI usage docs

## Backend Changes

**None expected.** All required data is already exposed by existing API endpoints:
- `/api/v1/query` — Returns OrchestrationResult with full pipeline data
- `/api/v1/brain/graph` — Returns knowledge graph
- `/api/v1/brain/status` — Returns memory status
- `/api/v1/knowledge-base/status` — Returns corpus stats
- `/api/v1/telemetry` — Returns run traces

If any adapter is needed, it will be a thin UI-layer transformation, not a backend modification.

## Testing

- Run existing test suite: `python -m pytest tests/ -x -q --ignore=tests/llm_gateway/test_phase07e_recovery.py`
- Target: 925 passed, 26 skipped, 0 failures
- Add UI tests where applicable
- Visual QA: inspect all states manually
