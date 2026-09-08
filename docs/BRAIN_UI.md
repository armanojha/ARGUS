# ARGUS Brain UI

The Brain is the primary interface for ARGUS. It visualizes how the system thinks through research questions.

**Open:** `http://localhost:8000/brain`

## Views

### Pipeline View (Default)

The pipeline view shows the research flow as an interactive canvas. Each node represents a processing stage:

| Node | Color | What It Shows |
|------|-------|---------------|
| Query | Blue | The user's research question |
| Analysis | Indigo | Detected pattern, entities, time window |
| Plan | Purple | Research strategy, sub-queries, budgets |
| Sub-query | Light Purple | Individual search queries |
| Evidence | Green | Retrieved chunks with scores |
| Verification | Yellow | Claim-support status, confidence |
| Conflict | Red | Contradictions between sources |
| Synthesis | Pink | Answer generation, outcome |
| Answer | Blue | Final answer with citations |

**Interactions:**
- **Click** — Open node inspector
- **Double-click** — Zoom and focus on node
- **Hover** — Show tooltip with description
- **Drag** — Pan the canvas
- **Scroll** — Zoom in/out
- **Arrow keys** — Navigate between connected nodes

### Architecture View

Interactive SVG diagram of ARGUS components. Click any component to see:

- What it does
- Why ARGUS needs it
- What goes in / what comes out

### Knowledge Graph

Force-directed graph of the evidence graph. Nodes represent entities, claims, events, documents, chunks, and sources. Edge types include supports, contradicts, derived_from, relates_to, and more.

**Filters:** Toggle visibility by node type using the filter chips.

### Research Chat

Ask questions in natural language. ARGUS plans, retrieves, verifies, and synthesizes an answer with citations.

### Other Views

- **Knowledge Base** — Upload documents, trigger ingestion
- **Memory** — View persistent memory records and layers
- **Documents** — Browse graph documents and sources
- **Evidence** — Knowledge graph statistics
- **Obsidian Brain** — Obsidian vault integration
- **Settings** — System health, telemetry, configuration

## Node Inspector

Click any pipeline node to open the inspector panel. It shows:

### What is this?

A simple explanation of the stage's purpose.

### Why does ARGUS need it?

Why this stage exists in the pipeline and what would happen without it.

### Data Sections

Each node type has specific data:

- **Query** — Original question text
- **Analysis** — Pattern, entities, time window, risk level
- **Plan** — Objective, sub-queries, strategy, budgets
- **Evidence** — Source, score, section, text preview
- **Verification** — Status, confidence, reasoning, breakdown
- **Conflict** — Type, confidence, entities, metrics, resolution
- **Synthesis** — Outcome, stop reason, iterations, warnings
- **Answer** — Full text, citations, evidence trail, telemetry, timeline

### Technical Details

Click "Show technical details" to see implementation-level information about input, output, and impact on the final answer.

## Evidence Traceability

The answer node inspector provides an Evidence Trail:

```
Answer
  ↓
Citation [1] — Source path, text excerpt, score
  ↓
Evidence Chunk — Full text, retrieval metadata
  ↓
Source Document — Original file
```

Click any citation chip or evidence card to open the source document.

## Conflict Visualization

Conflict nodes show:

- **Type** — Classification of the disagreement
- **Confidence** — How certain ARGUS is about the conflict
- **Entities** — Which entities are involved
- **Metrics** — Which measurements disagree
- **Resolution** — How ARGUS handled the conflict

## Demo Mode

No API key required. Click **Load Demo** on the empty pipeline view to see a pre-recorded research result.

The demo showcases:
- Multi-source comparison query
- 4 evidence chunks with scores
- Verification with confidence breakdown
- Conflict detection (DIFFERENT_TIMEFRAME)
- Grounded synthesis with citations

Clearly labeled "DEMO MODE — Pre-recorded data" to distinguish from live research.

## Research Timeline

**Approximate** — derived from available telemetry and routing information; does not represent exact backend stage timings.

```
Analysis    — 850ms (1 call — groq)
Plan        — 1200ms (1 call — groq)
Retrieval   — 320ms (4 chunks retrieved)
Verification — 950ms (1 call — gemini)
Synthesis   — 2100ms (1 call — nvidia_nim)
```

## Search

Use the search bar to find nodes by label, summary, or type. Results show type badges and connection counts. Click a result to focus and zoom on that node.

## Keyboard Navigation

| Key | Action |
|-----|--------|
| Arrow keys | Move to nearest neighbor |
| `+` / `-` | Zoom in/out |
| `0` | Fit to view |
| `Escape` | Clear selection / close panel |
| `Enter` | Send research question |

## Accessibility

- ARIA labels on all interactive elements
- Keyboard navigation for graph exploration
- Non-color status indicators (icons + labels)
- Skip-navigation link
- `prefers-reduced-motion` support
- Readable contrast ratios

## Responsive Design

The Brain UI adapts to different screen sizes:

- **Desktop** (1024px+) — Full layout with sidebar
- **Tablet** (768px-1024px) — Narrower panels
- **Mobile** (< 768px) — Collapsible sidebar, full-width panels

## Technical Details

- **Rendering:** Canvas 2D with D3.js force simulation
- **Graph:** Force-directed layout with cluster forces
- **State:** Vanilla JavaScript, no framework
- **Styling:** CSS custom properties, dark theme
- **Served:** FastAPI static file at `/brain`
- **Dependencies:** D3.js v7 (CDN), no build step
