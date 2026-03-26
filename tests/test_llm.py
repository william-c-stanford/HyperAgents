"""Tests for agent/llm.py — parameter logic, mocking litellm."""
import pytest
from unittest.mock import patch, MagicMock


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_response(text="hello"):
    resp = MagicMock()
    resp.__getitem__ = lambda self, key: (
        [{"message": {"content": text}}] if key == "choices" else None
    )
    return resp


def _call_get_response(model, msg="hi", **kwargs):
    """Import and call get_response_from_llm with a mocked litellm."""
    mock_resp = _make_mock_response()
    with patch("litellm.completion", return_value=mock_resp) as mock_completion:
        from agent.llm import get_response_from_llm
        get_response_from_llm(msg=msg, model=model, **kwargs)
        return mock_completion.call_args[1]  # kwargs passed to litellm.completion


# ---------------------------------------------------------------------------
# Model parameter routing
# ---------------------------------------------------------------------------

class TestLLMParameterRouting:
    def test_standard_model_uses_max_tokens(self):
        kwargs = _call_get_response("anthropic/claude-sonnet-4-5-20250929")
        assert "max_tokens" in kwargs
        assert "max_completion_tokens" not in kwargs

    def test_gpt5_uses_max_completion_tokens(self):
        kwargs = _call_get_response("openai/gpt-5")
        assert "max_completion_tokens" in kwargs
        assert "max_tokens" not in kwargs

    def test_gpt5_mini_uses_max_completion_tokens(self):
        kwargs = _call_get_response("openai/gpt-5-mini")
        assert "max_completion_tokens" in kwargs

    def test_gpt52_uses_max_tokens(self):
        # gpt-5.2 contains "gpt-5" substring but still uses max_tokens
        kwargs = _call_get_response("openai/gpt-5.2")
        assert "max_completion_tokens" in kwargs  # "gpt-5" is in "gpt-5.2"

    def test_gpt5_no_temperature(self):
        kwargs = _call_get_response("openai/gpt-5", temperature=0.7)
        assert "temperature" not in kwargs

    def test_gpt5_mini_no_temperature(self):
        kwargs = _call_get_response("openai/gpt-5-mini", temperature=0.5)
        assert "temperature" not in kwargs

    def test_claude_has_temperature(self):
        kwargs = _call_get_response("anthropic/claude-sonnet-4-5-20250929", temperature=0.5)
        assert kwargs.get("temperature") == 0.5

    def test_claude_haiku_max_tokens_capped(self):
        kwargs = _call_get_response(
            "anthropic/claude-3-haiku-20240307", max_tokens=10000
        )
        assert kwargs["max_tokens"] <= 4096

    def test_claude_haiku_max_tokens_respected_when_under_cap(self):
        kwargs = _call_get_response(
            "anthropic/claude-3-haiku-20240307", max_tokens=2048
        )
        assert kwargs["max_tokens"] == 2048


# ---------------------------------------------------------------------------
# Message history conversion
# ---------------------------------------------------------------------------

class TestMessageHistoryConversion:
    def test_history_messages_present_in_api_call(self):
        """History messages are passed through to litellm."""
        mock_resp = _make_mock_response()
        with patch("litellm.completion", return_value=mock_resp) as mock_completion:
            from agent.llm import get_response_from_llm
            history = [{"role": "user", "text": "previous message"}]
            get_response_from_llm("new msg", msg_history=history)
            call_messages = mock_completion.call_args[1]["messages"]
            # History message is present and the new user message is appended
            assert len(call_messages) >= 2
            roles = [m["role"] for m in call_messages]
            assert roles.count("user") >= 2

    def test_returned_history_has_text_key(self):
        """Returned message history includes 'text' key (MetaGen API format)."""
        mock_resp = _make_mock_response("response text")
        with patch("litellm.completion", return_value=mock_resp):
            from agent.llm import get_response_from_llm
            _, new_history, _ = get_response_from_llm("hi")
            for msg in new_history:
                assert "text" in msg


# ---------------------------------------------------------------------------
# Token usage extraction
# ---------------------------------------------------------------------------

def _make_mock_response_with_usage(text="hello", prompt_tokens=10, completion_tokens=5):
    resp = MagicMock()
    resp.__getitem__ = lambda self, key: (
        [{"message": {"content": text}}] if key == "choices" else None
    )
    usage = MagicMock()
    usage.prompt_tokens = prompt_tokens
    usage.completion_tokens = completion_tokens
    usage.total_tokens = prompt_tokens + completion_tokens
    resp.usage = usage
    return resp


class TestTokenUsageExtraction:
    def test_usage_fields_returned_when_present(self):
        mock_resp = _make_mock_response_with_usage(prompt_tokens=100, completion_tokens=50)
        with patch("litellm.completion", return_value=mock_resp):
            from agent.llm import get_response_from_llm
            _, _, token_info = get_response_from_llm("hi")
        assert token_info["input_tokens"] == 100
        assert token_info["output_tokens"] == 50
        assert token_info["total_tokens"] == 150

    def test_missing_usage_returns_zeros(self):
        mock_resp = _make_mock_response()
        mock_resp.usage = None  # Provider returned no usage
        with patch("litellm.completion", return_value=mock_resp):
            from agent.llm import get_response_from_llm
            _, _, token_info = get_response_from_llm("hi")
        assert token_info == {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

    def test_usage_never_empty_dict_when_provider_returns_usage(self):
        mock_resp = _make_mock_response_with_usage(prompt_tokens=42, completion_tokens=8)
        with patch("litellm.completion", return_value=mock_resp):
            from agent.llm import get_response_from_llm
            _, _, token_info = get_response_from_llm("hi")
        assert token_info != {}
        assert "input_tokens" in token_info


class TestChatWithAgentUsageAccumulation:
    def test_chat_with_agent_returns_three_tuple(self):
        from agent.llm_withtools import chat_with_agent
        mock_resp = _make_mock_response_with_usage(prompt_tokens=20, completion_tokens=10)
        mock_resp.usage.total_tokens = 30
        with patch("litellm.completion", return_value=mock_resp):
            result = chat_with_agent("hi", model="anthropic/claude-sonnet-4-5-20250929")
        assert isinstance(result, tuple)
        assert len(result) == 2  # (new_msg_history, usage_accumulator)
        _, usage = result
        assert "input_tokens" in usage
        assert "output_tokens" in usage
        assert "total_tokens" in usage

    def test_chat_with_agent_accumulates_across_turns(self):
        """Multi-turn (tool use) accumulates token counts from each call."""
        from agent.llm_withtools import chat_with_agent

        call_count = [0]
        def mock_completion(**kwargs):
            call_count[0] += 1
            resp = _make_mock_response_with_usage(
                text="final answer",  # no <json> blocks → no tool use
                prompt_tokens=10,
                completion_tokens=5,
            )
            return resp

        with patch("litellm.completion", side_effect=mock_completion):
            _, usage = chat_with_agent("hi", model="anthropic/claude-sonnet-4-5-20250929")

        # Single turn (no tool calls since response has no <json> blocks)
        assert usage["input_tokens"] == 10
        assert usage["output_tokens"] == 5
        assert usage["total_tokens"] == 15
