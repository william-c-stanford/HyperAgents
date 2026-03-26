"""Entropy metrics for HyperAgents codebase analysis.

Implements Structural Entropy (H_struct) and Coupling Entropy (H_couple) from
*Entropic Selection for Self-Improving Agents* (Stanford, 2026):

  H_struct(R) = -Σ p_i log2(p_i)    where p_i = tokens(f_i) / total_tokens
  H_couple(G) = -Σ r_i log2(r_i)    where r_i = fan-out(v_i) / Σ fan-out

Token counts use tiktoken with the ``cl100k_base`` encoding (GPT-3.5/4 vocabulary)
as a consistent, provider-agnostic measure of file size.

Inclusion scope: all .py files reachable from the repo root, excluding:
  - outputs/, venv_nat/, .venv/, venv/, __pycache__/, tests/

The exclusion of tests/ may warrant reconsideration for the full paper experiments.

Usage:
    from analysis.entropy_metrics import reconstruct_and_analyze
    metrics = reconstruct_and_analyze("/path/to/repo", "/path/to/model_patch.diff")
"""

from __future__ import annotations

import ast
import logging
import math
import os
import pathlib
import shutil
import subprocess
import tempfile
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default exclusion dirs (relative names checked against each path component)
# ---------------------------------------------------------------------------
_DEFAULT_EXCLUDE_DIRS = frozenset({
    "outputs", "venv_nat", ".venv", "venv", "__pycache__", "tests",
    ".git", "node_modules",
    # Exclude analysis/ permanently so instrumentation scripts never affect
    # the entropy baseline.  This directory is maintained separately from the
    # agent codebase and must not be evolved.
    "analysis",
})


# ---------------------------------------------------------------------------
# tiktoken encoder (lazy, cached)
# ---------------------------------------------------------------------------
_encoder = None


def _get_encoder():
    global _encoder
    if _encoder is None:
        import tiktoken
        _encoder = tiktoken.get_encoding("cl100k_base")
    return _encoder


def _count_tokens(file_content: str) -> int:
    """Return token count for a Python source file using cl100k_base encoding."""
    try:
        return len(_get_encoder().encode(file_content))
    except Exception as exc:
        logger.warning("tiktoken encode failed, falling back to char proxy: %s", exc)
        return max(1, len(file_content) // 4)


# ---------------------------------------------------------------------------
# File collection
# ---------------------------------------------------------------------------

def _collect_py_files(
    repo_root: pathlib.Path,
    exclude_dirs: frozenset[str] = _DEFAULT_EXCLUDE_DIRS,
) -> list[pathlib.Path]:
    """Return all .py files under repo_root, respecting exclusions."""
    results = []
    for dirpath, dirnames, filenames in os.walk(repo_root):
        # Prune excluded directories in-place so os.walk skips them
        dirnames[:] = [
            d for d in dirnames
            if d not in exclude_dirs and not d.startswith(".")
        ]
        for fname in filenames:
            if fname.endswith(".py"):
                results.append(pathlib.Path(dirpath) / fname)
    return results


# ---------------------------------------------------------------------------
# Structural Entropy
# ---------------------------------------------------------------------------

def compute_structural_entropy(
    repo_root: pathlib.Path | str,
    exclude_dirs: frozenset[str] = _DEFAULT_EXCLUDE_DIRS,
) -> dict:
    """Compute H_struct for the Python codebase at repo_root.

    Returns:
        {
            "h_struct": float,         # Shannon entropy in bits; 0.0 if ≤1 file
            "file_count": int,          # number of .py files analyzed
            "total_tokens": int,        # sum of all file token counts
        }
    """
    repo_root = pathlib.Path(repo_root)
    py_files = _collect_py_files(repo_root, exclude_dirs)

    sizes: list[int] = []
    for fpath in py_files:
        try:
            content = fpath.read_text(encoding="utf-8", errors="replace")
            sizes.append(_count_tokens(content))
        except Exception as exc:
            logger.warning("Could not read %s: %s — skipping", fpath, exc)

    # Remove zero-size files (excluded from entropy by convention)
    sizes = [s for s in sizes if s > 0]
    total = sum(sizes)

    if total == 0 or len(sizes) <= 1:
        return {"h_struct": 0.0, "file_count": len(sizes), "total_tokens": total}

    h = -sum((s / total) * math.log2(s / total) for s in sizes)
    return {"h_struct": h, "file_count": len(sizes), "total_tokens": total}


# ---------------------------------------------------------------------------
# Coupling Entropy
# ---------------------------------------------------------------------------

def _build_module_index(
    py_files: list[pathlib.Path],
    repo_root: pathlib.Path,
) -> dict[str, pathlib.Path]:
    """Map dotted module names to their file paths.

    e.g. ``agent.llm`` → ``<repo_root>/agent/llm.py``
    """
    index: dict[str, pathlib.Path] = {}
    for fpath in py_files:
        try:
            rel = fpath.relative_to(repo_root)
        except ValueError:
            continue
        parts = list(rel.parts)
        if parts[-1].endswith(".py"):
            parts[-1] = parts[-1][:-3]  # strip .py
        module_name = ".".join(parts)
        index[module_name] = fpath
        # Also index by final component name for bare ``import X`` statements
        index[parts[-1]] = fpath
    return index


def _extract_fan_out(fpath: pathlib.Path, module_index: dict[str, pathlib.Path]) -> int:
    """Count the number of local repo modules imported by fpath."""
    try:
        source = fpath.read_text(encoding="utf-8", errors="replace")
        tree = ast.parse(source, filename=str(fpath))
    except SyntaxError as exc:
        logger.warning("AST parse failed for %s: %s — skipping", fpath, exc)
        return 0
    except Exception as exc:
        logger.warning("Could not parse %s: %s — skipping", fpath, exc)
        return 0

    local_imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                name = alias.name
                if name in module_index or name.split(".")[0] in module_index:
                    local_imports.add(name)
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.level == 0:  # absolute import
                name = node.module
                if name in module_index or name.split(".")[0] in module_index:
                    local_imports.add(name)

    return len(local_imports)


def compute_coupling_entropy(
    repo_root: pathlib.Path | str,
    exclude_dirs: frozenset[str] = _DEFAULT_EXCLUDE_DIRS,
) -> dict:
    """Compute H_couple for the Python codebase at repo_root.

    Returns:
        {
            "h_couple": float,   # Shannon entropy; 0.0 if no local imports
            "node_count": int,   # number of .py files (graph nodes)
            "edge_count": int,   # total fan-out edges across all nodes
        }
    """
    repo_root = pathlib.Path(repo_root)
    py_files = _collect_py_files(repo_root, exclude_dirs)
    module_index = _build_module_index(py_files, repo_root)

    fan_outs: list[int] = [_extract_fan_out(f, module_index) for f in py_files]
    total_edges = sum(fan_outs)

    if total_edges == 0:
        return {"h_couple": 0.0, "node_count": len(py_files), "edge_count": 0}

    h = -sum(
        (fo / total_edges) * math.log2(fo / total_edges)
        for fo in fan_outs
        if fo > 0
    )
    return {"h_couple": h, "node_count": len(py_files), "edge_count": total_edges}


# ---------------------------------------------------------------------------
# Reconstruction helper
# ---------------------------------------------------------------------------

def reconstruct_and_analyze(
    repo_root: pathlib.Path | str,
    patch_file: Optional[pathlib.Path | str] = None,
) -> dict:
    """Apply patch to a temp copy of repo_root and compute entropy metrics.

    If patch_file is None, empty, or missing, entropy is computed on the
    unpatched repo (baseline generation).

    Returns a merged metrics dict:
        {
            "h_struct": float | None,
            "h_couple": float | None,
            "file_count": int | None,
            "total_tokens": int | None,
            "node_count": int | None,
            "edge_count": int | None,
        }

    On any error, the affected metric is None rather than raising.
    """
    repo_root = pathlib.Path(repo_root)
    patch_path = pathlib.Path(patch_file) if patch_file else None
    has_patch = patch_path is not None and patch_path.exists() and patch_path.stat().st_size > 0

    tmp_dir: Optional[str] = None
    try:
        tmp_dir = tempfile.mkdtemp(prefix="hyperagents_entropy_")
        tmp_path = pathlib.Path(tmp_dir)

        # Copy repo into temp dir
        shutil.copytree(str(repo_root), str(tmp_path / "repo"))
        analysis_root = tmp_path / "repo"

        # Apply patch if present
        if has_patch:
            result = subprocess.run(
                ["git", "apply", "--whitespace=fix", str(patch_path)],
                cwd=str(analysis_root),
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                logger.warning(
                    "git apply failed (returncode=%d): %s — analyzing unpatched repo",
                    result.returncode,
                    result.stderr.strip(),
                )

        # Compute metrics
        try:
            struct = compute_structural_entropy(analysis_root)
        except Exception as exc:
            logger.warning("compute_structural_entropy failed: %s", exc)
            struct = {"h_struct": None, "file_count": None, "total_tokens": None}

        try:
            couple = compute_coupling_entropy(analysis_root)
        except Exception as exc:
            logger.warning("compute_coupling_entropy failed: %s", exc)
            couple = {"h_couple": None, "node_count": None, "edge_count": None}

        return {**struct, **couple}

    except Exception as exc:
        logger.warning("reconstruct_and_analyze failed: %s", exc)
        return {
            "h_struct": None,
            "h_couple": None,
            "file_count": None,
            "total_tokens": None,
            "node_count": None,
            "edge_count": None,
        }
    finally:
        if tmp_dir and os.path.exists(tmp_dir):
            try:
                shutil.rmtree(tmp_dir)
            except Exception as exc:
                logger.warning("Failed to clean up temp dir %s: %s", tmp_dir, exc)
