"""Integration tests for Claude OAuth via ccproxy.

These tests make REAL LLM calls. They require:
  - ANTHROPIC_AUTH_MODE=oauth in .env (or environment)
  - Claude Code credentials at ~/.claude/.credentials.json
    (written automatically when you log into Claude Code)

Included in the default test suite. Tests are skipped (with a warning) when
OAuth is not configured — no extra flags needed.

Run with:
  python -m pytest tests/ -v                      # full suite
  python -m pytest tests/test_oauth_integration.py -v  # OAuth only

For Docker (proxy pre-configured via env vars):
  docker run --network=host \
    -e ANTHROPIC_BASE_URL=http://127.0.0.1:8765/claude \
    -e ANTHROPIC_API_KEY=ccproxy-oauth \
    <image> python -m pytest tests/test_oauth_integration.py -v
"""
import json
import os
import sys
import warnings
import pytest
from dotenv import load_dotenv

load_dotenv()

# Ensure the tests/ directory is on sys.path so conftest helpers are importable
# when this file is collected directly (not via a full suite run).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from conftest import proxy_preconfigured

_OAUTH_UP = os.getenv("ANTHROPIC_AUTH_MODE") == "oauth" or proxy_preconfigured()

# Emit a visible warning when OAuth is unavailable so the skip is not silent.
if not _OAUTH_UP:
    warnings.warn(
        "OAuth not configured — Claude OAuth integration tests will be skipped. "
        "To run them: set ANTHROPIC_AUTH_MODE=oauth and run `ccproxy auth login claude_api`",
        UserWarning,
        stacklevel=1,
    )

pytestmark = pytest.mark.skipif(
    not _OAUTH_UP,
    reason="ANTHROPIC_AUTH_MODE=oauth not set and no pre-configured proxy — skipping OAuth integration tests",
)

_CREDS_PATH = os.path.expanduser("~/.claude/.credentials.json")


def _read_oauth_token() -> str | None:
    """Read the Claude Code OAuth token from the credentials file."""
    try:
        with open(_CREDS_PATH) as f:
            return json.load(f).get("claudeAiOauth", {}).get("accessToken", "")
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        return None


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def ccproxy_running():
    """Ensure the Anthropic proxy is available for the whole module.

    Three modes:
    1. Pre-configured (Docker / CI): ANTHROPIC_BASE_URL already points at a
       running proxy — yield immediately, nothing to start or stop.
    2. Local OAuth, ccproxy already running on port: reuse it.
    3. Local OAuth, ccproxy not running: start it from credentials file.
    """
    from conftest import proxy_preconfigured

    # Mode 1: proxy URL already injected (e.g. Docker --network=host)
    if proxy_preconfigured():
        yield
        return

    from utils.ccproxy_manager import (
        is_ccproxy_running,
        start_ccproxy,
        stop_ccproxy,
        setup_ccproxy_env,
        _patch_ccproxy_oauth_header,
    )

    if not _read_oauth_token():
        pytest.skip(
            "No OAuth token found in ~/.claude/.credentials.json\n"
            "Log in with Claude Code to generate credentials."
        )

    port = int(os.getenv("CCPROXY_PORT", "8765"))

    # Mode 2: ccproxy already running locally
    if is_ccproxy_running(port):
        setup_ccproxy_env(port)
        yield
        return

    # Mode 3: start ccproxy locally
    _patch_ccproxy_oauth_header()
    try:
        proc = start_ccproxy(port)
    except RuntimeError as e:
        pytest.skip(f"Could not start ccproxy: {e}")
        return

    setup_ccproxy_env(port)
    yield
    stop_ccproxy(proc)


# ---------------------------------------------------------------------------
# Credential / proxy health tests
# ---------------------------------------------------------------------------

class TestCcproxyHealth:
    def test_ccproxy_binary_found(self):
        from conftest import proxy_preconfigured
        if proxy_preconfigured():
            pytest.skip("Proxy pre-configured via env vars — binary not required")
        from utils.ccproxy_manager import is_ccproxy_available
        assert is_ccproxy_available(), (
            "ccproxy binary not found — install with: pip install ccproxy-api"
        )

    def test_claude_credentials_present(self):
        """Claude Code has written OAuth credentials to disk."""
        from conftest import proxy_preconfigured
        if proxy_preconfigured():
            pytest.skip("Proxy pre-configured via env vars — credentials file not required")
        assert os.path.exists(_CREDS_PATH), (
            f"{_CREDS_PATH} not found — log in with Claude Code first"
        )
        token = _read_oauth_token()
        assert token, "claudeAiOauth.accessToken missing in credentials file"

    def test_ccproxy_serving(self, ccproxy_running):
        """Proxy endpoint is reachable (either local ccproxy or forwarded host proxy)."""
        import httpx
        from conftest import proxy_preconfigured
        base_url = os.getenv("ANTHROPIC_BASE_URL", "")
        if proxy_preconfigured():
            # Derive the health URL from ANTHROPIC_BASE_URL (strip /claude suffix)
            health_base = base_url.rstrip("/claude").rstrip("/")
            health_url = f"{health_base}/health/live"
        else:
            from utils.ccproxy_manager import is_ccproxy_running
            port = int(os.getenv("CCPROXY_PORT", "8765"))
            assert is_ccproxy_running(port), f"ccproxy not responding on port {port}"
            return
        try:
            resp = httpx.get(health_url, timeout=5.0)
            assert resp.status_code == 200, f"Proxy health check failed: {resp.status_code}"
        except Exception as e:
            pytest.fail(f"Proxy health check unreachable at {health_url}: {e}")

    def test_anthropic_base_url_set(self, ccproxy_running):
        base_url = os.getenv("ANTHROPIC_BASE_URL", "")
        assert base_url, "ANTHROPIC_BASE_URL not set after ccproxy start"
        assert "127.0.0.1" in base_url

    def test_anthropic_api_key_set(self, ccproxy_running):
        assert os.getenv("ANTHROPIC_API_KEY") == "ccproxy-oauth"


# ---------------------------------------------------------------------------
# Real LLM call tests
# ---------------------------------------------------------------------------

class TestLiveClaudeCall:
    def test_basic_response(self, ccproxy_running):
        """A simple call returns a non-empty string."""
        from agent.llm import get_response_from_llm, CLAUDE_MODEL
        response, _, _ = get_response_from_llm(
            "Reply with the single word: hello", model=CLAUDE_MODEL
        )
        assert isinstance(response, str)
        assert len(response) > 0

    def test_response_is_coherent(self, ccproxy_running):
        """Model understands the prompt and responds sensibly."""
        from agent.llm import get_response_from_llm, CLAUDE_MODEL
        response, _, _ = get_response_from_llm(
            "What is 2 + 2? Reply with just the number.",
            model=CLAUDE_MODEL,
        )
        assert "4" in response

    def test_message_history_preserved(self, ccproxy_running):
        """Multi-turn conversation works correctly."""
        from agent.llm import get_response_from_llm, CLAUDE_MODEL
        _, history, _ = get_response_from_llm(
            "My name is TestBot. Remember it.", model=CLAUDE_MODEL
        )
        response, _, _ = get_response_from_llm(
            "What is my name?", model=CLAUDE_MODEL, msg_history=history
        )
        assert "TestBot" in response

    def test_max_tokens_respected(self, ccproxy_running):
        """Response stays within the requested token budget."""
        from agent.llm import get_response_from_llm, CLAUDE_MODEL
        response, _, _ = get_response_from_llm(
            "Count from 1 to 1000.", model=CLAUDE_MODEL, max_tokens=50
        )
        assert len(response) < 500

    def test_temperature_zero_deterministic(self, ccproxy_running):
        """Two calls with temperature=0 return the same answer."""
        from agent.llm import get_response_from_llm, CLAUDE_MODEL
        prompt = "What is the capital of France? Reply with just the city name."
        r1, _, _ = get_response_from_llm(prompt, model=CLAUDE_MODEL, temperature=0.0)
        r2, _, _ = get_response_from_llm(prompt, model=CLAUDE_MODEL, temperature=0.0)
        assert r1.strip() == r2.strip()


# ---------------------------------------------------------------------------
# Proxy health endpoint
# ---------------------------------------------------------------------------

class TestCcproxyProxy:
    def test_proxy_health_endpoint(self, ccproxy_running):
        """Proxy /health/live returns 200."""
        import httpx
        from conftest import proxy_preconfigured
        base_url = os.getenv("ANTHROPIC_BASE_URL", "")
        if proxy_preconfigured():
            health_url = base_url.rstrip("/claude").rstrip("/") + "/health/live"
        else:
            port = int(os.getenv("CCPROXY_PORT", "8765"))
            health_url = f"http://127.0.0.1:{port}/health/live"
        resp = httpx.get(health_url, timeout=5.0)
        assert resp.status_code == 200
