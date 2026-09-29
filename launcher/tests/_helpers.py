"""Shared helpers for the ktl_env test suite (not a test module)."""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
import ktl_env  # noqa: E402


def mk_resolved(**kw):
    base = dict(
        client="opencode",
        model_key="qwen",
        api_model="qwen3.8-27b",
        context=262144,
        max_output=65536,
        cost={"input": 0.45, "output": 3.2, "cache_read": 0.05},
        base_url="https://ex.dev",
        client_key="sk-test-123",
        key_is_placeholder=False,
        key_source="env",
        url_source="cli",
        shell="bash",
        reveal=False,
    )
    base.update(kw)
    return ktl_env.Resolved(**base)


def placeholder_resolved(**kw):
    kw.setdefault("client_key", ktl_env.PLACEHOLDER)
    kw.setdefault("key_is_placeholder", True)
    kw.setdefault("key_source", "placeholder")
    return mk_resolved(**kw)
