"""Reproduce the real-world failure: irrelevant evidence + false contradictions."""
import asyncio
import json
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.orchestration.graph import run_query


async def main():
    query = "What evidence supports the claim that AI adoption is accelerating?"
    print(f"QUERY: {query}")
    print("=" * 80)

    result = await run_query(query)

    print(f"\nOUTCOME: {result.outcome}")
    print(f"CONFIDENCE: {result.confidence}")
    print(f"LOOP COUNT: {result.loop_count}")
    print(f"TOKENS USED: {result.total_tokens}")

    print(f"\n--- EVIDENCE ({len(result.evidence)} chunks) ---")
    for i, ev in enumerate(result.evidence):
        score = ev.get("score", "N/A")
        text_preview = ev.get("text", "")[:150]
        source = ev.get("source_path", ev.get("source", "unknown"))
        print(f"  [{i+1}] score={score} source={source}")
        print(f"      {text_preview}...")

    print(f"\n--- CONTRADICTION SIGNALS ({len(result.contradiction_signals)}) ---")
    for i, cs in enumerate(result.contradiction_signals):
        print(f"  [{i+1}] type={cs.get('type', '?')}")
        print(f"      shared_terms={cs.get('shared_terms', [])}")
        print(f"      confidence={cs.get('confidence', '?')}")
        evidence_a = cs.get("evidence_a", {})
        evidence_b = cs.get("evidence_b", {})
        print(f"      A: {str(evidence_a.get('text', ''))[:100]}")
        print(f"      B: {str(evidence_b.get('text', ''))[:100]}")

    print(f"\n--- SYNTHESIS ---")
    print(f"Answer preview: {str(result.answer)[:300] if result.answer else 'None'}")

    print(f"\n--- RAW RESULT KEYS ---")
    print(json.dumps({k: type(v).__name__ for k, v in result.__dict__.items()}, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
