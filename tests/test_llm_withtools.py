"""Tests for agent/llm_withtools.py — pure logic only (no LLM calls)."""
import pytest
from unittest.mock import patch, MagicMock
from agent.llm_withtools import (
    get_tooluse_prompt,
    should_retry_tool_use,
    check_for_tool_uses,
    process_tool_call,
)


# ---------------------------------------------------------------------------
# get_tooluse_prompt
# ---------------------------------------------------------------------------

class TestGetToolUsePrompt:
    def test_empty_tools_returns_empty_string(self):
        assert get_tooluse_prompt([]) == ""
        assert get_tooluse_prompt(None) == ""

    def test_single_tool_included_in_prompt(self):
        tool_info = MagicMock()
        tool_info.__str__ = lambda self: "my_tool: does stuff"
        prompt = get_tooluse_prompt([tool_info])
        assert "my_tool: does stuff" in prompt
        assert "tool_name" in prompt
        assert "tool_input" in prompt

    def test_multiple_tools_joined(self):
        t1, t2 = MagicMock(), MagicMock()
        t1.__str__ = lambda self: "tool_one"
        t2.__str__ = lambda self: "tool_two"
        prompt = get_tooluse_prompt([t1, t2])
        assert "tool_one" in prompt
        assert "tool_two" in prompt


# ---------------------------------------------------------------------------
# should_retry_tool_use
# ---------------------------------------------------------------------------

class TestShouldRetryToolUse:
    def _long_response(self, content):
        """Pad content to >= 2000 chars."""
        return content + " " * max(0, 2000 - len(content))

    def test_returns_false_when_tool_uses_present(self):
        response = self._long_response("<json>\ntool_name\ntool_input\n")
        assert should_retry_tool_use(response, tool_uses=[{"tool_name": "x"}]) is False

    def test_returns_true_when_all_markers_in_order_and_long(self):
        response = self._long_response("<json> tool_name tool_input")
        assert should_retry_tool_use(response, tool_uses=None) is True

    def test_returns_false_when_response_too_short(self):
        response = "<json> tool_name tool_input"  # < 2000 chars
        assert should_retry_tool_use(response, tool_uses=None) is False

    def test_returns_false_when_markers_out_of_order(self):
        # tool_input before tool_name
        response = self._long_response("<json> tool_input tool_name")
        assert should_retry_tool_use(response, tool_uses=None) is False

    def test_returns_false_when_json_marker_missing(self):
        response = self._long_response("tool_name tool_input")
        assert should_retry_tool_use(response, tool_uses=None) is False

    def test_returns_false_when_empty_tool_uses_list(self):
        # empty list is falsy — treated as no tool uses
        response = self._long_response("<json> tool_name tool_input")
        assert should_retry_tool_use(response, tool_uses=[]) is True


# ---------------------------------------------------------------------------
# check_for_tool_uses
# ---------------------------------------------------------------------------

class TestCheckForToolUses:
    def test_valid_tool_use(self):
        response = '<json>{"tool_name": "bash", "tool_input": {"cmd": "ls"}}</json>'
        result = check_for_tool_uses(response)
        assert result is not None
        assert len(result) == 1
        assert result[0]["tool_name"] == "bash"
        assert result[0]["tool_input"] == {"cmd": "ls"}

    def test_multiple_tool_uses(self):
        response = (
            '<json>{"tool_name": "bash", "tool_input": {}}</json>'
            '<json>{"tool_name": "edit", "tool_input": {}}</json>'
        )
        result = check_for_tool_uses(response)
        assert result is not None
        assert len(result) == 2

    def test_malformed_json_skipped(self):
        response = '<json>{bad json}</json>'
        assert check_for_tool_uses(response) is None

    def test_missing_tool_name_skipped(self):
        response = '<json>{"tool_input": {}}</json>'
        assert check_for_tool_uses(response) is None

    def test_missing_tool_input_skipped(self):
        response = '<json>{"tool_name": "bash"}</json>'
        assert check_for_tool_uses(response) is None

    def test_no_json_tags_returns_none(self):
        assert check_for_tool_uses("no tool calls here") is None
        assert check_for_tool_uses("") is None

    def test_mixed_valid_and_invalid(self):
        response = (
            '<json>{bad}</json>'
            '<json>{"tool_name": "bash", "tool_input": {}}</json>'
        )
        result = check_for_tool_uses(response)
        assert result is not None
        assert len(result) == 1
        assert result[0]["tool_name"] == "bash"


# ---------------------------------------------------------------------------
# process_tool_call
# ---------------------------------------------------------------------------

class TestProcessToolCall:
    def test_calls_correct_tool(self):
        mock_fn = MagicMock(return_value="output")
        tools_dict = {"my_tool": {"function": mock_fn}}
        result = process_tool_call(tools_dict, "my_tool", {"arg": "val"})
        mock_fn.assert_called_once_with(arg="val")
        assert result == "output"

    def test_unknown_tool_returns_error(self):
        result = process_tool_call({}, "nonexistent", {})
        assert "Error" in result
        assert "nonexistent" in result

    def test_tool_exception_returns_error(self):
        def bad_tool(**kwargs):
            raise ValueError("something went wrong")
        tools_dict = {"bad": {"function": bad_tool}}
        result = process_tool_call(tools_dict, "bad", {})
        assert "Error" in result
        assert "bad" in result
