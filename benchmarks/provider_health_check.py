#!/usr/bin/env python3
"""Lightweight provider health check for Phase 32.1.
Tests each provider with a minimal request to verify availability.
"""
import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


async def check_provider(name, router, tier="strong"):
    """Send a minimal request to check provider health."""
    from app.llm_gateway.providers.models import Message, MessageRole
    messages = [
        Message(role=MessageRole.SYSTEM, content="Reply with exactly: OK"),
        Message(role=MessageRole.USER, content="Reply with exactly: OK"),
    ]
    t0 = time.time()
    try:
        response = await router.complete(
            messages, temperature=0.0, timeout=15,
            call_type="synthesis", request_id=None, query="health check", tier=tier
        )
        latency = (time.time() - t0) * 1000
        content = response.content or ""
        usage = getattr(response, "usage", None)
        prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
        completion_tokens = getattr(usage, "completion_tokens", 0) or 0
        model = getattr(response, "model", "unknown")
        provider = getattr(response, "provider", "unknown")
        return {
            "name": name, "status": "OK", "latency_ms": latency,
            "model": model, "provider": provider,
            "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
            "response_preview": content[:50],
        }
    except Exception as e:
        latency = (time.time() - t0) * 1000
        error_str = str(e)
        is_rate_limit = "RATE_LIMIT" in error_str or "429" in error_str or "quota" in error_str.lower()
        return {
            "name": name, "status": "RATE_LIMITED" if is_rate_limit else "ERROR",
            "latency_ms": latency, "error": error_str[:200],
        }


async def main():
    from app.llm_gateway import get_router
    router = get_router()

    print("=" * 60)
    print("Provider Health Check — Phase 32.1")
    print("=" * 60)

    providers = [
        ("Groq (GPT-OSS-120B)", "strong"),
        ("Gemini (flash-lite)", "strong"),
        ("Z.ai (glm-4.5-flash)", "strong"),
        ("Zen (nemotron)", "strong"),
    ]

    results = []
    for name, tier in providers:
        print(f"\n  Testing {name}...", end=" ", flush=True)
        r = await check_provider(name, router, tier)
        results.append(r)
        status = r["status"]
        if status == "OK":
            print(f"OK ({r['latency_ms']:.0f}ms, {r.get('model', '?')})")
        else:
            print(f"{status} ({r['latency_ms']:.0f}ms)")
            if "error" in r:
                print(f"    Error: {r['error'][:120]}")
        await asyncio.sleep(1)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    ok_count = sum(1 for r in results if r["status"] == "OK")
    rl_count = sum(1 for r in results if r["status"] == "RATE_LIMITED")
    err_count = sum(1 for r in results if r["status"] == "ERROR")

    for r in results:
        icon = {"OK": "[OK]", "RATE_LIMITED": "[RL]", "ERROR": "[ERR]"}.get(r["status"], "?")
        print(f"  {icon} {r['name']}: {r['status']}")

    print(f"\n  Available: {ok_count}  Rate-limited: {rl_count}  Error: {err_count}")

    if ok_count == 0:
        print("\n  ❌ NO PROVIDERS AVAILABLE — Benchmark deferred.")
        return False
    elif rl_count > 0:
        print(f"\n  ⚠️  {rl_count} provider(s) rate-limited. Benchmark may be contaminated.")
        return ok_count >= 1
    else:
        print("\n  ✅ All providers healthy. Benchmark can proceed.")
        return True


if __name__ == "__main__":
    ok = asyncio.run(main())
    sys.exit(0 if ok else 1)
