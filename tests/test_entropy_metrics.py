"""Tests for analysis/entropy_metrics.py."""
import math
import pathlib
import textwrap

import pytest


# ---------------------------------------------------------------------------
# Helpers to build synthetic file trees
# ---------------------------------------------------------------------------

def _write_files(tmp_path: pathlib.Path, files: dict[str, str]) -> None:
    """Write a dict of {rel_path: content} under tmp_path."""
    for rel, content in files.items():
        fpath = tmp_path / rel
        fpath.parent.mkdir(parents=True, exist_ok=True)
        fpath.write_text(content, encoding="utf-8")


# ---------------------------------------------------------------------------
# Structural Entropy
# ---------------------------------------------------------------------------

class TestComputeStructuralEntropy:
    def test_single_file_entropy_is_zero(self, tmp_path):
        from analysis.entropy_metrics import compute_structural_entropy
        _write_files(tmp_path, {"a.py": "x = 1\n" * 50})
        result = compute_structural_entropy(tmp_path)
        assert result["h_struct"] == 0.0
        assert result["file_count"] == 1

    def test_two_equal_files_entropy_is_one(self, tmp_path):
        from analysis.entropy_metrics import compute_structural_entropy
        # Equal-size files → p_i = 0.5 → H = 1.0 bit
        content = "x = 1\n" * 100
        _write_files(tmp_path, {"a.py": content, "b.py": content})
        result = compute_structural_entropy(tmp_path)
        assert result["h_struct"] == pytest.approx(1.0, abs=0.05)
        assert result["file_count"] == 2

    def test_entropy_increases_with_more_equal_files(self, tmp_path):
        from analysis.entropy_metrics import compute_structural_entropy
        content = "x = 1\n" * 50
        _write_files(tmp_path, {f"f{i}.py": content for i in range(4)})
        result = compute_structural_entropy(tmp_path)
        # H = log2(4) = 2.0 for 4 equal-size files
        assert result["h_struct"] == pytest.approx(2.0, abs=0.1)

    def test_excludes_outputs_dir(self, tmp_path):
        from analysis.entropy_metrics import compute_structural_entropy
        content = "x = 1\n" * 50
        _write_files(tmp_path, {
            "agent.py": content,
            "outputs/should_be_excluded.py": content * 10,
        })
        result = compute_structural_entropy(tmp_path)
        assert result["file_count"] == 1

    def test_excludes_pycache(self, tmp_path):
        from analysis.entropy_metrics import compute_structural_entropy
        content = "x = 1\n" * 50
        _write_files(tmp_path, {
            "agent.py": content,
            "__pycache__/agent.cpython-311.pyc": content,
        })
        # __pycache__ is excluded; .pyc is not .py so ignored anyway
        result = compute_structural_entropy(tmp_path)
        assert result["file_count"] == 1

    def test_empty_repo_returns_zero(self, tmp_path):
        from analysis.entropy_metrics import compute_structural_entropy
        result = compute_structural_entropy(tmp_path)
        assert result["h_struct"] == 0.0
        assert result["total_tokens"] == 0

    def test_total_tokens_is_sum(self, tmp_path):
        from analysis.entropy_metrics import compute_structural_entropy
        _write_files(tmp_path, {"a.py": "pass\n", "b.py": "pass\n"})
        result = compute_structural_entropy(tmp_path)
        assert result["total_tokens"] > 0
        assert result["file_count"] == 2


# ---------------------------------------------------------------------------
# Coupling Entropy
# ---------------------------------------------------------------------------

class TestComputeCouplingEntropy:
    def test_no_imports_entropy_is_zero(self, tmp_path):
        from analysis.entropy_metrics import compute_coupling_entropy
        _write_files(tmp_path, {
            "a.py": "x = 1\n",
            "b.py": "y = 2\n",
        })
        result = compute_coupling_entropy(tmp_path)
        assert result["h_couple"] == 0.0
        assert result["edge_count"] == 0

    def test_single_importer(self, tmp_path):
        from analysis.entropy_metrics import compute_coupling_entropy
        _write_files(tmp_path, {
            "utils.py": "def helper(): pass\n",
            "main.py": "from utils import helper\n",
        })
        result = compute_coupling_entropy(tmp_path)
        # Only one node has fan-out > 0 → r_i = 1.0 → H = 0.0
        assert result["h_couple"] == 0.0
        assert result["edge_count"] == 1

    def test_two_equal_importers_entropy_is_one(self, tmp_path):
        from analysis.entropy_metrics import compute_coupling_entropy
        _write_files(tmp_path, {
            "utils.py": "def helper(): pass\n",
            "main.py": "from utils import helper\n",
            "other.py": "from utils import helper\n",
        })
        result = compute_coupling_entropy(tmp_path)
        # Each of main.py and other.py has fan-out=1, total=2 → r_i=0.5 → H=1.0
        assert result["h_couple"] == pytest.approx(1.0, abs=0.05)

    def test_ast_parse_error_file_skipped(self, tmp_path):
        from analysis.entropy_metrics import compute_coupling_entropy
        _write_files(tmp_path, {
            "good.py": "x = 1\n",
            "broken.py": "def (\n",  # invalid syntax
        })
        # Should not raise; broken file is skipped
        result = compute_coupling_entropy(tmp_path)
        assert "h_couple" in result

    def test_node_count_matches_file_count(self, tmp_path):
        from analysis.entropy_metrics import compute_coupling_entropy
        _write_files(tmp_path, {
            "a.py": "import b\n",
            "b.py": "x = 1\n",
            "c.py": "import a\nimport b\n",
        })
        result = compute_coupling_entropy(tmp_path)
        assert result["node_count"] == 3


# ---------------------------------------------------------------------------
# reconstruct_and_analyze
# ---------------------------------------------------------------------------

class TestReconstructAndAnalyze:
    def test_no_patch_returns_baseline_entropy(self, tmp_path):
        from analysis.entropy_metrics import reconstruct_and_analyze
        _write_files(tmp_path, {"a.py": "x = 1\n" * 50, "b.py": "y = 2\n" * 50})
        result = reconstruct_and_analyze(tmp_path, patch_file=None)
        assert "h_struct" in result
        assert "h_couple" in result
        assert result["h_struct"] is not None

    def test_missing_patch_file_returns_baseline(self, tmp_path):
        from analysis.entropy_metrics import reconstruct_and_analyze
        _write_files(tmp_path, {"a.py": "x = 1\n"})
        result = reconstruct_and_analyze(tmp_path, patch_file=tmp_path / "nonexistent.diff")
        assert result["h_struct"] is not None

    def test_empty_patch_file_returns_baseline(self, tmp_path):
        from analysis.entropy_metrics import reconstruct_and_analyze
        _write_files(tmp_path, {"a.py": "x = 1\n"})
        patch = tmp_path / "empty.diff"
        patch.write_text("")
        result = reconstruct_and_analyze(tmp_path, patch_file=patch)
        assert result["h_struct"] is not None

    def test_no_temp_dirs_left_behind(self, tmp_path, tmp_path_factory):
        import os
        import tempfile
        from analysis.entropy_metrics import reconstruct_and_analyze

        _write_files(tmp_path, {"a.py": "x = 1\n"})
        tmp_before = set(os.listdir(tempfile.gettempdir()))
        reconstruct_and_analyze(tmp_path)
        tmp_after = set(os.listdir(tempfile.gettempdir()))
        # No new hyperagents_entropy_ dirs left behind
        new_dirs = {d for d in (tmp_after - tmp_before) if d.startswith("hyperagents_entropy_")}
        assert len(new_dirs) == 0

    def test_returns_all_expected_keys(self, tmp_path):
        from analysis.entropy_metrics import reconstruct_and_analyze
        _write_files(tmp_path, {"a.py": "x = 1\n"})
        result = reconstruct_and_analyze(tmp_path)
        for key in ("h_struct", "h_couple", "file_count", "total_tokens", "node_count", "edge_count"):
            assert key in result, f"Missing key: {key}"
