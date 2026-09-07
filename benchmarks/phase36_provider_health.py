#!/usr/bin/env python3
"""Phase 36: Provider Health Check.

Tests each provider/model combination before the clean benchmark.
If any primary provider is unhealthy, STOP the benchmark.

Usage:
    python -m benchmarks.phase36_provider_health
"""
from __future__ import annotations

import io
import json
import os
import sys
import time
from pathlib import Path
from dataclasses import dataclass, field

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")


@dataclass
class HealthResult:
    provider: str
    model: str
    call_type: str
    success: bool
    latency_ms: float
    error: str | None = None
    status_code: int | None = None
    response_length: int = 0
    tokens_used: int = 0


def test_provider(provider_name: str, model: str, call_type: str, api_key: str, base_url: str) -> HealthResult:
    """Test a single provider/model with a minimal generation request."""
    import httpx

    if not api_key:
        return HealthResult(
            provider=provider_name, model=model, call_type=call_type,
            success=False, error="No API key configured"
        )

    messages = [{"role": "user", "content": "Say exactly: OK"}]

    # Build request based on provider format
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": 10,
        "temperature": 0.0,
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    # Gemini uses a different auth format
    if provider_name == "gemini":
        headers["Authorization"] = f"Bearer {api_key}"

    url = f"{base_url.rstrip('/')}/chat/completions"

    t0 = time.perf_counter()
    try:
        with httpx.Client(timeout=30.0) as client:
            resp = client.post(url, json=payload, headers=headers)
            latency_ms = (time.perf_counter() - t0) * 1000

            if resp.status_code == 200:
                data = resp.json()
                content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                usage = data.get("usage", {})
                total_tokens = usage.get("total_tokens", 0)
                return HealthResult(
                    provider=provider_name, model=model, call_type=call_type,
                    success=True, latency_ms=latency_ms,
                    response_length=len(content),
                    tokens_used=total_tokens,
                )
            else:
                return HealthResult(
                    provider=provider_name, model=model, call_type=call_type,
                    success=False, latency_ms=latency_ms,
                    error=f"HTTP {resp.status_code}: {resp.text[:200]}",
                    status_code=resp.status_code,
                )
    except httpx.TimeoutException:
        latency_ms = (time.perf_counter() - t0) * 1000
        return HealthResult(
            provider=provider_name, model=model, call_type=call_type,
            success=False, latency_ms=latency_ms, error="Timeout"
        )
    except Exception as e:
        latency_ms = (time.perf_counter() - t0) * 1000
        return HealthResult(
            provider=provider_name, model=model, call_type=call_type,
            success=False, latency_ms=latency_ms, error=str(e)
        )


def run_health_check():
    print("=" * 70)
    print("ARGUS Phase 36: Provider Health Check")
    print("=" * 70)

    # Load API keys from .env
    env_path = Path(__file__).resolve().parent.parent / ".env"
    env_vars = {}
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env_vars[k.strip()] = v.strip()

    # Providers to test for the benchmark
    providers = [
        ("groq", "openai/gpt-oss-20b", "query_analysis", "GROQ_API_KEY",
         "https://api.groq.com/openai/v1"),
        ("groq", "openai/gpt-oss-120b", "evidence_extraction", "GROQ_API_KEY",
         "https://api.groq.com/openai/v1"),
        ("groq", "openai/gpt-oss-120b", "synthesis", "GROQ_API_KEY",
         "https://api.groq.com/openai/v1"),
        ("gemini", "gemini-3.5-flash-lite", "research_planning", "GEMINI_API_KEY",
         "https://generativelanguage.googleapis.com/v1beta/openai/"),
        ("gemini", "gemini-3.5-flash-lite", "verification", "GEMINI_API_KEY",
         "https://generativelanguage.googleapis.com/v1beta/openai/"),
    ]

    results = []
    all_healthy = True

    print(f"\nTesting {len(providers)} provider/model combinations...\n")

    for provider_name, model, call_type, key_env, base_url in providers:
        api_key = env_vars.get(key_env, "")
        result = test_provider(provider_name, model, call_type, api_key, base_url)
        results.append(result)

        status = "✓ OK" if result.success else "✗ FAIL"
        latency_str = f"{result.latency_ms:.0f}ms"
        error_str = f" — {result.error}" if result.error else ""

        print(f"  {status}  {provider_name:<10} {model:<35} {call_type:<22} {latency_str:>8}{error_str}")

        if not result.success:
            all_healthy = False

    # Summary
    print("\n" + "-" * 70)
    healthy_count = sum(1 for r in results if r.success)
    print(f"\nHealthy: {healthy_count}/{len(results)}")

    if all_healthy:
        print("\n✓ All providers healthy. Benchmark can proceed.")
    else:
        print("\n✗ Some providers unhealthy. Benchmark may be contaminated.")
        failed = [r for r in results if not r.success]
        for r in failed:
            print(f"  FAILED: {r.provider}/{r.model} ({r.call_type}): {r.error}")

    # Save results
    output_dir = Path("benchmarks/results")
    output_dir.mkdir(exist_ok=True)

    report = {
        "benchmark": "phase36_provider_health",
        "all_healthy": all_healthy,
        "results": [
            {
                "provider": r.provider,
                "model": r.model,
                "call_type": r.call_type,
                "success": r.success,
                "latency_ms": round(r.latency_ms, 2),
                "error": r.error,
                "response_length": r.response_length,
                "tokens_used": r.tokens_used,
            }
            for r in results
        ],
    }

    out_path = output_dir / "phase36_provider_health.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved to {out_path}")

    return all_healthy


if __name__ == "__main__":
    healthy = run_health_check()
    sys.exit(0 if healthy else 1)
