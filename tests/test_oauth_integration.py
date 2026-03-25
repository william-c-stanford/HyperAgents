"""Integration tests for Claude OAuth via ccproxy.

These tests make REAL LLM calls. They require:
  - ANTHROPIC_AUTH_MODE=oauth in .env (or environment)
  - Claude Code credentials at ~/.claude/.credentials.json
    (written automatically when you log into Claude Code)

Run with:
  python -m pytest tests/test_oauth_integration.py -v

Skip automatically if OAuth is not configured.
"""
import json
import os
import pytest
from dotenv import load_dotenv

load_dotenv()

# Skip all tests in this module if OAuth mode is not configured
pytestmark = pytest.mark.skipif(
    os.getenv("ANTHROPIC_AUTH_MODE") != "oauth",
    reason="ANTHROPIC_AUTH_MODE=oauth not set — skipping OAuth integration tests",
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
    """Ensure ccproxy is started once for the whole module.

    Reads credentials from ~/.claude/.credentials.json directly (no
    subprocess auth check) to avoid entry-point discovery issues in pytest.
    """
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

    if is_ccproxy_running(port):
        setup_ccproxy_env(port)
        yield
        return

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
        from utils.ccproxy_manager import is_ccproxy_available
        assert is_ccproxy_available(), (
            "ccproxy binary not found — install with: pip install ccproxy-api"
        )

    def test_claude_credentials_present(self):
        """Claude Code has written OAuth credentials to disk."""
        assert os.path.exists(_CREDS_PATH), (
            f"{_CREDS_PATH} not found — log in with Claude Code first"
        )
        token = _read_oauth_token()
        assert token, "claudeAiOauth.accessToken missing in credentials file"

    def test_ccproxy_serving(self, ccproxy_running):
        from utils.ccproxy_manager import is_ccproxy_running
        port = int(os.getenv("CCPROXY_PORT", "8765"))
        assert is_ccproxy_running(port), f"ccproxy not responding on port {port}"

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
        """ccproxy /health/live returns 200."""
        import httpx
        port = int(os.getenv("CCPROXY_PORT", "8765"))
        resp = httpx.get(f"http://127.0.0.1:{port}/health/live", timeout=5.0)
        assert resp.status_code == 200
