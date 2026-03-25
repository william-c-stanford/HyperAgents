"""Shared pytest configuration and fixtures.

Detects whether tests are running inside a Docker container and exposes
helpers that other test modules can use.
"""
import os
import pytest


def in_docker() -> bool:
    """Return True if the current process is running inside a Docker container."""
    return os.path.exists("/.dockerenv") or os.getenv("DOCKER_CONTAINER") == "1"


def proxy_preconfigured() -> bool:
    """Return True if ANTHROPIC_BASE_URL is already set (host proxy mode)."""
    base_url = os.getenv("ANTHROPIC_BASE_URL", "")
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    return bool(base_url) and api_key == "ccproxy-oauth"


def ollama_preconfigured() -> bool:
    """Return True if a reachable Ollama instance is available.

    Probes OLLAMA_API_BASE (default http://localhost:11434) via /api/tags.
    Works for both local Ollama and Docker --network=host forwarding.
    """
    import httpx

    base = os.getenv("OLLAMA_API_BASE", "http://localhost:11434")
    try:
        resp = httpx.get(f"{base}/api/tags", timeout=2.0)
        return resp.status_code == 200
    except Exception:
        return False
