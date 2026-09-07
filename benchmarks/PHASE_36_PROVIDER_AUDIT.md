# Phase 36 Provider Configuration Audit

**Date:** 2026-09-07

---

## Provider Overview

| Provider | Enabled | API Key | Status |
|----------|---------|---------|--------|
| groq | Yes | Set | Active — primary for query_analysis, evidence_extraction |
| gemini | Yes | Set | Active — primary for research_planning, verification |
| cerebras | Yes | Empty | No API key — acts as dead fallback |
| zen | Yes | Set | Active — last-resort fallback |
| zai | Yes | Set | Active — fallback |
| nvidia_nim | Yes | Set | Active — primary for synthesis |

## Routing Chains (per call_type)

| call_type | Primary | F1 | F2 | F3 | F4 |
|-----------|---------|----|----|----|----|
| query_analysis | groq/gpt-oss-20b | gemini/flash-lite | cerebras/gpt-oss-120b | zai/glm-4.5-flash | zen/lightning-free |
| research_planning | gemini/flash-lite | groq/gpt-oss-120b | cerebras/gpt-oss-120b | zai/glm-4.5-flash | zen/big-pickle |
| evidence_extraction | groq/gpt-oss-120b | gemini/flash-lite | cerebras/gpt-oss-120b | zai/glm-4.5-flash | zen/mimo-v2.5-free |
| synthesis | nvidia_nim/nemotron-3-ultra | groq/gpt-oss-120b | gemini/flash-lite | cerebras/gpt-oss-120b | zai/glm-4.5-flash |
| verification | gemini/flash-lite | groq/gpt-oss-20b | cerebras/gpt-oss-120b | zai/glm-4.5-flash | zen/mimo-v2.5-free |

## Provider Quotas (Free Tier)

| Provider | RPM | RPD | TPM | TPD |
|----------|-----|-----|-----|-----|
| groq | 30 | 1,000 | 8,000 | 200,000 |
| gemini | 15 | 1,000 | 250,000 | 1,000,000 |
| cerebras | 10 | 500 | 50,000 | 1,000,000 |
| zen | unlimited | unlimited | unlimited | unlimited |
| zai | unlimited | unlimited | unlimited | unlimited |
| nvidia_nim | unlimited | unlimited | unlimited | unlimited |

## Phase 35 Contamination Analysis

Every query hit fallbacks because:

1. **Groq TPM exhaustion**: Primary for query_analysis + evidence_extraction. With 8K TPM, ~5 calls of ~500 tokens each = 2,500 tokens. 9 queries = 22,500 tokens. Within 200K TPD but tight for rapid sequential calls.

2. **Gemini free-tier exhaustion**: Primary for research_planning + verification. 15 RPM limit means we can only do ~15 calls/minute. With multiple queries in succession, this limit is hit.

3. **z.ai timeout**: One 21s timeout on P35-Q08 verification. Provider infrastructure issue.

## Clean Benchmark Strategy

For Phase 36, the goal is to avoid ALL fallbacks. This requires:

1. **Respect rate limits**: Add delays between queries if needed
2. **Use appropriate models**: Lighter models for structured output, heavier for synthesis
3. **Limit call count**: Stay within provider capacity
4. **Monitor telemetry**: Track fallback_count and rate_limit_events per query

### Recommended Configuration for Clean Benchmark

- **query_analysis**: groq/gpt-oss-20b (light, fast)
- **research_planning**: gemini/flash-lite (primary, reliable for structured output)
- **evidence_extraction**: groq/gpt-oss-120b (primary, but watch TPM)
- **synthesis**: groq/gpt-oss-120b (skip nvidia_nim for benchmark — avoids extra provider complexity)
- **verification**: gemini/flash-lite (primary, reliable)

### Critical Constraint

Groq's 8,000 TPM is the tightest limit. Each evidence_extraction call uses ~1,000-1,500 tokens. With 8 queries × ~2 iterations × 1 evidence_extraction each = ~16 calls. At ~1,200 tokens/call = ~19,200 tokens. This exceeds 8,000 TPM if done rapidly.

**Mitigation**: Either slow down between calls, or switch evidence_extraction to gemini for the benchmark.
