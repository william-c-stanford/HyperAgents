"""Backfill entropy metrics for existing output directories.

Usage:
    python analysis/compute_entropy_backfill.py --output-dir outputs/generate_20260325_212836_252984/
    python analysis/compute_entropy_backfill.py --output-dir outputs/run/ --force

Computes Structural Entropy and Coupling Entropy for each generation in the given
output directory and writes them to each gen's metadata.json.  Token usage (c_mod)
cannot be backfilled because the raw LLM responses are not persisted.

archive.jsonl is NOT updated — it is append-only.  Only metadata.json is touched.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

# Allow running from repo root or from analysis/ subdirectory
_repo_root = pathlib.Path(__file__).parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from analysis.entropy_metrics import reconstruct_and_analyze
from utils.gl_utils import update_node_metadata


# ---------------------------------------------------------------------------
# Gen discovery
# ---------------------------------------------------------------------------

def _discover_generations(output_dir: pathlib.Path) -> list[str]:
    """Return genids found in output_dir as strings, in natural order."""
    gens = []
    for entry in sorted(output_dir.iterdir()):
        if entry.is_dir() and entry.name.startswith("gen_"):
            genid = entry.name[len("gen_"):]
            gens.append(genid)
    return gens


def _coerce_genid(genid_str: str):
    """Return int if numeric, else string (for 'initial')."""
    if genid_str == "initial":
        return "initial"
    try:
        return int(genid_str)
    except ValueError:
        return genid_str


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Backfill entropy metrics for an existing output directory."
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Path to the run output directory (e.g. outputs/generate_20260325_212836_252984/)",
    )
    parser.add_argument(
        "--repo-root",
        default=None,
        help="Path to the repo root for entropy reconstruction. Defaults to CWD.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing metrics in metadata.json.",
    )
    args = parser.parse_args()

    output_dir = pathlib.Path(args.output_dir).resolve()
    if not output_dir.is_dir():
        print(f"Error: output directory not found: {output_dir}")
        sys.exit(1)

    repo_root = pathlib.Path(args.repo_root).resolve() if args.repo_root else pathlib.Path.cwd()

    genid_strs = _discover_generations(output_dir)
    if not genid_strs:
        print(f"No gen_* directories found in {output_dir}")
        sys.exit(0)

    print(f"Found {len(genid_strs)} generation(s) in {output_dir}")
    print(f"Repo root for entropy analysis: {repo_root}")
    print()

    results = []
    for genid_str in genid_strs:
        genid = _coerce_genid(genid_str)
        gen_dir = output_dir / f"gen_{genid_str}"
        metadata_file = gen_dir / "metadata.json"

        # Check if already processed
        if metadata_file.exists() and not args.force:
            with open(metadata_file) as f:
                existing = json.load(f)
            if "metrics" in existing and existing["metrics"] is not None:
                print(f"  gen_{genid_str}: already has metrics (skip; use --force to overwrite)")
                results.append((genid_str, existing["metrics"]))
                continue

        # Find patch file
        patch_file = gen_dir / "agent_output" / "model_patch.diff"
        has_patch = patch_file.exists() and patch_file.stat().st_size > 0

        try:
            metrics = reconstruct_and_analyze(
                repo_root,
                patch_file=patch_file if has_patch else None,
            )
        except Exception as exc:
            print(f"  gen_{genid_str}: ERROR during entropy computation: {exc}")
            results.append((genid_str, None))
            continue

        # c_mod not backfillable — set to null
        metrics["c_mod"] = None

        # Write to metadata.json
        update_node_metadata(str(output_dir), genid, {"metrics": metrics})
        results.append((genid_str, metrics))

        h_struct = f"{metrics['h_struct']:.4f}" if metrics.get("h_struct") is not None else "null"
        h_couple = f"{metrics['h_couple']:.4f}" if metrics.get("h_couple") is not None else "null"
        patch_info = f"(patched)" if has_patch else "(baseline)"
        print(f"  gen_{genid_str} {patch_info}: H_struct={h_struct}  H_couple={h_couple}  files={metrics.get('file_count')}  tokens={metrics.get('total_tokens')}")

    # Summary table
    print()
    print(f"{'gen':<12} {'H_struct':>10} {'H_couple':>10} {'files':>7} {'tokens':>10}")
    print("-" * 55)
    for genid_str, metrics in results:
        if metrics is None:
            print(f"{'gen_' + genid_str:<12} {'ERROR':>10}")
            continue
        h_struct = f"{metrics['h_struct']:.4f}" if metrics.get("h_struct") is not None else "null"
        h_couple = f"{metrics['h_couple']:.4f}" if metrics.get("h_couple") is not None else "null"
        files = str(metrics.get("file_count", ""))
        tokens = str(metrics.get("total_tokens", ""))
        print(f"{'gen_' + genid_str:<12} {h_struct:>10} {h_couple:>10} {files:>7} {tokens:>10}")


if __name__ == "__main__":
    main()
