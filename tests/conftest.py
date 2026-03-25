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
