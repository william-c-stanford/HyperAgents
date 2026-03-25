"""Integration tests for local LLM inference via Ollama.

These tests make REAL LLM calls against a locally running Ollama instance.
They require:
  - Ollama installed and running (https://ollama.com)
  - The target model pulled: ollama pull qwen3.5:9b-q4_K_M

Included in the default test suite. Tests are skipped (with a warning) when
Ollama is not reachable — no extra flags needed.

Run with:
  python -m pytest tests/ -v                     # full suite
  python -m pytest tests/test_ollama_integration.py -v  # Ollama only

For Docker (with --network=host):
  docker run --network=host <image> \
    python -m pytest tests/test_ollama_integration.py -v

To use a different model or host:
  OLLAMA_MODEL=ollama_chat/qwen3.5:9b-q5_K_M \
  OLLAMA_API_BASE=http://localhost:11434 \
  python -m pytest tests/test_ollama_integration.py -v
"""
import os
import sys
import warnings
import pytest

# Ensure the tests/ directory is on sys.path so conftest helpers are importable
# when this file is collected directly (not via a full suite run).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from conftest import ollama_preconfigured

_OLLAMA_UP = ollama_preconfigured()

# Emit a visible warning when Ollama is unavailable so the skip is not silent.
if not _OLLAMA_UP:
    _base = os.getenv("OLLAMA_API_BASE", "http://localhost:11434")
    warnings.warn(
        f"Ollama not reachable at {_base} — "
        "Ollama integration tests will be skipped. "
        "To run them: start Ollama and run `ollama pull qwen3.5:9b-q4_K_M`",
        UserWarning,
        stacklevel=1,
    )

pytestmark = pytest.mark.skipif(
    not _OLLAMA_UP,
    reason="Ollama not reachable — start Ollama and pull the model, then re-run",
)

_OLLAMA_BASE = os.getenv("OLLAMA_API_BASE", "http://localhost:11434")


def _get_ollama_model_tag() -> str:
    """Extract the bare model tag from the OLLAMA_MODEL constant (strips provider prefix)."""
    from agent.llm import OLLAMA_MODEL
    # e.g. "ollama_chat/qwen2.5-coder:7b-instruct-q4_K_M" → "qwen2.5-coder:7b-instruct-q4_K_M"
    return OLLAMA_MODEL.split("/", 1)[-1]


def _list_pulled_models() -> list[str]:
    """Return model names from Ollama's /api/tags endpoint."""
    import httpx
    resp = httpx.get(f"{_OLLAMA_BASE}/api/tags", timeout=5.0)
    resp.raise_for_status()
    return [m["name"] for m in resp.json().get("models", [])]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def ollama_running():
    """Verify Ollama is up and the target model is pulled.

    Skips (not errors) when the model hasn't been pulled yet, with an
    actionable pull command.
    """
    target = _get_ollama_model_tag()
    pulled = _list_pulled_models()

    # Ollama may store the model without the quantization suffix in the name
    # list; check for prefix match as well as exact match.
    model_present = any(
        name == target or name.startswith(target.split(":")[0])
        for name in pulled
    )

    if not model_present:
        warnings.warn(
            f"Ollama model '{target}' not pulled — "
            f"run: ollama pull {target}",
            UserWarning,
            stacklevel=1,
        )
        pytest.skip(
            f"Model '{target}' not pulled. Run: ollama pull {target}\n"
            f"Available models: {pulled or '(none)'}"
        )

    yield


# ---------------------------------------------------------------------------
# Health / connectivity tests
# ---------------------------------------------------------------------------

class TestOllamaHealth:
    def test_ollama_endpoint_reachable(self):
        """Ollama /api/tags returns 200."""
        import httpx
        resp = httpx.get(f"{_OLLAMA_BASE}/api/tags", timeout=5.0)
        assert resp.status_code == 200, f"Unexpected status: {resp.status_code}"

    def test_target_model_pulled(self, ollama_running):
        """The configured model is present in the pulled-models list."""
        target = _get_ollama_model_tag()
        pulled = _list_pulled_models()
        assert any(
            name == target or name.startswith(target.split(":")[0])
            for name in pulled
        ), f"Model '{target}' not found. Pulled: {pulled}"


# ---------------------------------------------------------------------------
# Live LLM call tests
# ---------------------------------------------------------------------------

class TestLiveOllamaCall:
    def test_basic_response(self, ollama_running):
        """A simple call returns a non-empty string."""
        from agent.llm import get_response_from_llm, OLLAMA_MODEL
        response, _, _ = get_response_from_llm(
            "Reply with the single word: hello", model=OLLAMA_MODEL
        )
        assert isinstance(response, str)
        assert len(response) > 0

    def test_response_is_coherent(self, ollama_running):
        """Model understands the prompt and responds sensibly."""
        from agent.llm import get_response_from_llm, OLLAMA_MODEL
        response, _, _ = get_response_from_llm(
            "What is 2 + 2? Reply with just the number.",
            model=OLLAMA_MODEL,
        )
        assert "4" in response

    def test_message_history_preserved(self, ollama_running):
        """Multi-turn conversation works correctly."""
        from agent.llm import get_response_from_llm, OLLAMA_MODEL
        _, history, _ = get_response_from_llm(
            "My name is TestBot. Remember it.", model=OLLAMA_MODEL
        )
        response, _, _ = get_response_from_llm(
            "What is my name?", model=OLLAMA_MODEL, msg_history=history
        )
        assert "TestBot" in response

    def test_max_tokens_respected(self, ollama_running):
        """Response stays within the requested token budget."""
        from agent.llm import get_response_from_llm, OLLAMA_MODEL
        response, _, _ = get_response_from_llm(
            "Count from 1 to 1000.", model=OLLAMA_MODEL, max_tokens=50
        )
        assert len(response) < 500

    def test_temperature_zero_deterministic(self, ollama_running):
        """Two calls with temperature=0 return the same answer."""
        from agent.llm import get_response_from_llm, OLLAMA_MODEL
        prompt = "What is the capital of France? Reply with just the city name."
        r1, _, _ = get_response_from_llm(prompt, model=OLLAMA_MODEL, temperature=0.0)
        r2, _, _ = get_response_from_llm(prompt, model=OLLAMA_MODEL, temperature=0.0)
        # Use case-insensitive comparison to tolerate minor Metal threading variation
        assert r1.strip().lower() == r2.strip().lower()
