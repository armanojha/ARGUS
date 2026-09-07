# Phase 41 — Conflict Signal Precision: Diagnostic Data

## Detection Architecture

### Contradiction Signal Schema

Each signal now includes structured metadata:

```python
{
    "severity": 0.8,           # 0.0-1.0 severity score
    "confidence": "HIGH",      # HIGH/MEDIUM/LOW
    "conflict_type": "GENUINE_CONTRADICTION",  # See taxonomy
    "description": "Evidence [1] and [2] present different values...",
    "evidence_indices": [1, 2],
    "entity_overlap": ["acme", "corporation"],
    "metric_overlap": ["revenue"],
    "timeframe_i": [2023],
    "timeframe_j": [2025],
    "resolved": false,
    "critical": true
}
```

### Conflict Type Distribution

From direct testing on C1 (revenue query):

| Type | Count | Filtered |
|------|-------|----------|
| DIFFERENT_TIMEFRAME | 2 | 0 (kept: metric matches query) |
| GENUINE_CONTRADICTION | 1 | 0 (kept: metric matches query) |
| **Total** | **3** | **0** |

### Confidence Distribution

| Level | Count | Criteria |
|-------|-------|----------|
| HIGH | 1 | Same entity + same timeframe + severity ≥ 0.8 |
| MEDIUM | 2 | Same entity + severity ≥ 0.5 |
| LOW | 0 | — |

## Temporal Extraction

### Patterns Detected
- Year: `\b(20[0-9]{2})\b` — matches 2023, 2025, etc.
- Quarter: `\bQ([1-4])\b` — matches Q1-Q4
- As-of: `\bas of\b.*?\b(20[0-9]{2})\b` — matches "as of 2025"

### Examples
| Text | Years | Quarters |
|------|-------|----------|
| "Revenue in 2023 was $3.1B" | {2023} | {} |
| "Q3 2025 metrics" | {2025} | {Q3} |
| "As of January 2025, revenue..." | {2025} | {} |

## Unit Normalization

### Canonical Units
| Input | Canonical | Multiplier (to billions) |
|-------|-----------|------------------------|
| billion, bn, B | B | 1.0 |
| million, mn, M | M | 0.001 |
| thousand, K | K | 0.000001 |
| percent, %, pct | % | 1.0 |

### Normalization Examples
| Input | Value | Unit | Normalized |
|-------|-------|------|-----------|
| "$3.1 billion" | 3.1 | B | 3.1 |
| "$3,100 million" | 3100.0 | M | 3.1 |
| "15%" | 15.0 | % | 15.0 |
| "0.15" | 0.15 | "" | 0.15 |

**Key**: 3.1B == 3100M after normalization (not a contradiction)

## Entity Extraction

### Heuristic
- Capitalized words (3+ chars)
- Excluded: "the", "and", "for", "with", "from", "this", "that", "report", "annual", "fiscal", "total", "combined"

### Examples
| Text | Entities |
|------|----------|
| "Acme Corporation reported..." | {acme, corporation} |
| "The Ohio plant produced..." | {ohio, plant} |
| "New York region showed..." | {york, region, york} |

## Query-Aware Filtering Rules

### Rule 1: DIFFERENT_TIMEFRAME
```python
if conflict_type == "DIFFERENT_TIMEFRAME":
    if not historical_intent and not (metric_overlap & query_metrics):
        continue  # Filter out
```

**Effect**: Keeps DIFFERENT_TIMEFRAME when query asks about the same metric

### Rule 2: LOW Confidence
```python
if confidence == "LOW" and not conflict_intent:
    continue  # Filter out
```

**Effect**: Only shows LOW confidence when query explicitly asks about conflicts

### Rule 3: Metric Relevance
```python
if query_metrics:
    if not (metric_overlap & query_metrics):
        continue  # Filter out
else:
    if not conflict_intent and not historical_intent:
        continue  # Filter out all conflicts
```

**Effect**: Non-metric queries get zero conflicts (unless conflict/historical intent)

## False Positive Analysis

### Before Filtering (80% FP rate)

| Query | Signals | Why FP |
|-------|---------|--------|
| What products does Acme make? | 8 | Revenue chunks retrieved alongside product info |
| Where are Acme plants located? | 9 | Location chunks mention employee counts |
| What industry is Acme in? | 8 | Industry description includes revenue figures |

### After Filtering (0% FP rate)

| Query | Signals | Why Filtered |
|-------|---------|-------------|
| What products does Acme make? | 0 | No metric keyword in query |
| Where are Acme plants located? | 0 | No metric keyword in query |
| How many employees does Acme have? | 0 | Metric "employees" in query but no conflict on employees |
| What industry is Acme in? | 0 | No metric keyword in query |

## Degraded Synthesis Safety

### Before Phase 41
```
Synthesis is temporarily unavailable, so I could not produce a polished answer.
Here is the grounded evidence I retrieved:
- Evidence text [1]
- Evidence text [2]
```

### After Phase 41
```
Synthesis is temporarily unavailable, so I could not produce a polished answer.
Here is the grounded evidence I retrieved:
- Evidence text [1]
- Evidence text [2]

IMPORTANT: The evidence contains the following conflicts:
  - Evidence [1] and [2] present different values for revenue: [3.1, 2023] vs [4.7, 2025]
These conflicts could not be fully resolved during synthesis.
```

## Synthesis Prompt Structure

### Before Phase 41
```
--- CONTRADICTION ALERT ---
The retrieved evidence contains contradictions. When synthesizing:
1. Acknowledge the conflict explicitly in your answer
2. Present both sides with their respective sources
3. If one source is more authoritative or recent, note that
4. Do NOT present contradictory claims as settled fact
- Severity high: Source A says X, Source B says not-X
--- END CONTRADICTION ALERT ---
```

### After Phase 41
```
--- RELEVANT CONFLICTS DETECTED ---
The retrieved evidence contains conflicts relevant to this query.

SAFE RESOLUTION RULES:
1. Present BOTH sides with their respective evidence citations [X] and [Y]
2. State clearly that the sources disagree
3. Do NOT select one side as 'correct' unless evidence establishes
   clear authority (newer date, official source, etc.)
4. If one source is more authoritative, explain WHY
5. If authority is unclear, say: 'The available evidence does not
   establish which value is correct'
6. NEVER present contradictory claims as settled fact

Conflict details:
- Severity 0.8 | Confidence HIGH | Type: GENUINE_CONTRADICTION
  Evidence [1] and [2] present different values for metric(s) revenue
  Entity: ['acme'] | Metric: ['revenue'] | Timeframes: [2023] vs [2025]
--- END CONFLICTS ---
```
