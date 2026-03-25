"""Tests for utils/common.py."""
import os
import json
import tempfile
import pytest
from utils.common import extract_jsons, file_exist_and_not_empty, load_json_file


# ---------------------------------------------------------------------------
# extract_jsons
# ---------------------------------------------------------------------------

class TestExtractJsons:
    def test_xml_tag_format(self):
        response = '<json>{"key": "value"}</json>'
        result = extract_jsons(response)
        assert result == [{"key": "value"}]

    def test_markdown_code_block_format(self):
        response = '```json\n{"key": "value"}\n```'
        result = extract_jsons(response)
        assert result == [{"key": "value"}]

    def test_multiple_json_blocks(self):
        response = '<json>{"a": 1}</json> some text <json>{"b": 2}</json>'
        result = extract_jsons(response)
        assert result == [{"a": 1}, {"b": 2}]

    def test_both_formats_in_one_response(self):
        response = '<json>{"a": 1}</json>\n```json\n{"b": 2}\n```'
        result = extract_jsons(response)
        assert {"a": 1} in result
        assert {"b": 2} in result

    def test_no_json_returns_none(self):
        assert extract_jsons("no json here") is None
        assert extract_jsons("") is None

    def test_malformed_json_skipped(self):
        response = '<json>{bad json}</json><json>{"good": true}</json>'
        result = extract_jsons(response)
        assert result == [{"good": True}]

    def test_all_malformed_returns_none(self):
        response = '<json>{bad}</json>```json\nalso bad\n```'
        assert extract_jsons(response) is None

    def test_nested_json(self):
        response = '<json>{"outer": {"inner": [1, 2, 3]}}</json>'
        result = extract_jsons(response)
        assert result == [{"outer": {"inner": [1, 2, 3]}}]

    def test_whitespace_around_json(self):
        response = '<json>  \n  {"key": "value"}  \n  </json>'
        result = extract_jsons(response)
        assert result == [{"key": "value"}]


# ---------------------------------------------------------------------------
# file_exist_and_not_empty
# ---------------------------------------------------------------------------

class TestFileExistAndNotEmpty:
    def test_nonexistent_file(self):
        assert file_exist_and_not_empty("/tmp/definitely_does_not_exist_xyz.txt") is False

    def test_existing_nonempty_file(self):
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            f.write("some content")
            path = f.name
        try:
            assert file_exist_and_not_empty(path) is True
        finally:
            os.unlink(path)

    def test_existing_empty_file(self):
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            path = f.name
        try:
            assert file_exist_and_not_empty(path) is False
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# load_json_file
# ---------------------------------------------------------------------------

class TestLoadJsonFile:
    def test_loads_valid_json(self):
        data = {"key": "value", "number": 42}
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            json.dump(data, f)
            path = f.name
        try:
            result = load_json_file(path)
            assert result == data
        finally:
            os.unlink(path)

    def test_raises_on_invalid_json(self):
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            f.write("{not valid json}")
            path = f.name
        try:
            with pytest.raises(json.JSONDecodeError):
                load_json_file(path)
        finally:
            os.unlink(path)
