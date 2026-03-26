"""NeurIPS-quality metric trajectory plots for entropy and token usage.

Reads per-generation metrics from archive.jsonl and produces publication-quality
plots of:
  - Structural Entropy (H_struct)       — teal-cyan
  - Coupling Entropy   (H_couple)       — magenta
  - Input tokens   (C_mod component)   — gold
  - Output tokens  (C_mod component)   — rose
  - Total tokens   (C_mod)             — violet

Single-run usage:
    python -m analysis.plot_metrics --path outputs/generate_20260325_212836_252984/

Compare multiple runs (ablations):
    python -m analysis.plot_metrics \\
        --path  outputs/run_null/         --label "Null model" \\
        --compare outputs/run_entropy/    --label "Entropy selection" \\
        --compare outputs/run_ablation/   --label "Ablation B" \\
        --output comparison.png

Programmatic single-run:
    from analysis.plot_metrics import plot_metrics
    plot_metrics(exp_dir)

Programmatic comparison:
    from analysis.plot_metrics import plot_metrics_compare
    plot_metrics_compare(
        runs=[
            {"label": "Null model",        "path": "outputs/run_null/metrics_summary.json"},
            {"label": "Entropy selection", "path": "outputs/run_entropy/metrics_summary.json"},
        ],
        output_path="outputs/comparison.png",
    )
"""

from __future__ import annotations

import argparse
import json
import os
import warnings

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from utils.gl_utils import load_archive_data

__all__ = ["plot_metrics", "plot_metrics_compare", "export_metrics_json", "load_metrics_json"]

# ---------------------------------------------------------------------------
# NeurIPS-style rcParams
# ---------------------------------------------------------------------------

_NEURIPS_RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica Neue", "Helvetica", "DejaVu Sans"],
    "mathtext.fontset": "stixsans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "axes.titlepad": 8,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "xtick.major.size": 4,
    "ytick.major.size": 4,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "legend.fontsize": 9,
    "legend.frameon": False,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
}

# ---------------------------------------------------------------------------
# Color palettes
# ---------------------------------------------------------------------------

# Single-run: metric-specific colors
_METRIC_COLOR = {
    "h_struct":     "#17C3B2",   # teal-cyan
    "h_couple":     "#D62598",   # magenta
    "c_mod_input":  "#F5C518",   # gold
    "c_mod_output": "#FF6B8A",   # rose
    "c_mod_total":  "#7B5EA7",   # violet
}

# Multi-run: one color per run, used across all metric panels
_RUN_PALETTE = [
    "#17C3B2",   # teal-cyan
    "#D62598",   # magenta
    "#F5C518",   # gold
    "#7B5EA7",   # violet
    "#FF6B35",   # orange
    "#4A90D9",   # blue
]

_SCATTER_ALPHA = 0.85
_LINE_ALPHA    = 0.55
_DOT_SIZE      = 28
_LW            = 1.8

# Panel specs: (json_key, panel_title, y_label, metric_color_key, scale)
_PANEL_SPECS = [
    ("h_struct",    r"Structural Entropy  $H_{\rm struct}$", "entropy (bits)", "h_struct",    1.0),
    ("h_couple",    r"Coupling Entropy  $H_{\rm couple}$",   "entropy (bits)", "h_couple",    1.0),
    ("c_mod_input", "Input Tokens",                          "tokens (k)",     "c_mod_input", 1e-3),
    ("c_mod_output","Output Tokens",                         "tokens (k)",     "c_mod_output",1e-3),
    ("c_mod_total", r"Total Tokens  ($C_{\rm mod}$)",        "tokens (k)",     "c_mod_total", 1e-3),
]


# ---------------------------------------------------------------------------
# Data extraction from archive.jsonl / metadata.json
# ---------------------------------------------------------------------------

def _read_metadata_metrics(exp_dir: str, genid) -> dict | None:
    meta_path = os.path.join(exp_dir, f"gen_{genid}", "metadata.json")
    if not os.path.exists(meta_path):
        return None
    try:
        with open(meta_path) as f:
            return json.load(f).get("metrics")
    except Exception:
        return None


def _extract_from_run_dir(exp_dir: str) -> list[dict]:
    """Return ordered list of per-generation flat metric dicts from a run directory.

    Each dict has the keys defined in _PANEL_SPECS plus 'genid',
    'generation_index', 'file_count', 'repo_total_tokens'.
    Skips the 'initial' entry.
    """
    archive_path = os.path.join(os.path.normpath(exp_dir), "archive.jsonl")
    archive_data = load_archive_data(archive_path, last_only=False)

    seen: set = set()
    rows: list[dict] = []
    for entry in archive_data:
        genid = entry.get("current_genid")
        if genid in seen or genid == "initial":
            continue
        seen.add(genid)
        metrics = entry.get("metrics") or _read_metadata_metrics(exp_dir, genid)
        c_mod = (metrics or {}).get("c_mod") or {}
        rows.append({
            "genid":              genid,
            "generation_index":   len(rows),
            "h_struct":           (metrics or {}).get("h_struct"),
            "h_couple":           (metrics or {}).get("h_couple"),
            "file_count":         (metrics or {}).get("file_count"),
            "repo_total_tokens":  (metrics or {}).get("total_tokens"),
            "c_mod_input":        c_mod.get("input_tokens"),
            "c_mod_output":       c_mod.get("output_tokens"),
            "c_mod_total":        c_mod.get("total_tokens"),
        })
    return rows


# ---------------------------------------------------------------------------
# JSON export / load  (the portable interchange format)
# ---------------------------------------------------------------------------

def export_metrics_json(exp_dir: str, label: str | None = None) -> str:
    """Extract metrics from a run directory and save metrics_summary.json.

    The JSON is the single source of truth for recreating or comparing plots.
    It contains all the data needed to reproduce metrics_plot.png, and can be
    loaded by load_metrics_json() for multi-run comparisons.

    Returns the path to the saved file.

    JSON schema:
    {
      "run_id": "generate_20260325_212836_252984",
      "label": "null model",
      "generations": [
        {
          "genid": 1,
          "generation_index": 0,
          "h_struct": 5.9409,
          "h_couple": 5.6297,
          "file_count": 122,
          "repo_total_tokens": 210783,
          "c_mod_input": null,
          "c_mod_output": null,
          "c_mod_total": null
        },
        ...
      ]
    }
    """
    run_id = os.path.basename(os.path.normpath(exp_dir))
    rows = _extract_from_run_dir(exp_dir)
    payload = {
        "run_id": run_id,
        "label": label or run_id,
        "generations": rows,
    }
    out_path = os.path.join(exp_dir, "metrics_summary.json")
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    return out_path


def load_metrics_json(path: str) -> dict:
    """Load a metrics_summary.json file.  path can be a run dir or the JSON file."""
    # Accept either a run dir (auto-locate) or a direct file path
    if os.path.isdir(path):
        path = os.path.join(path, "metrics_summary.json")
    with open(path) as f:
        return json.load(f)


def _generations_to_arrays(generations: list[dict]) -> tuple[list, dict[str, list]]:
    """Convert generations list to (xs, {metric_key: [values]}) arrays."""
    xs = [g["generation_index"] for g in generations]
    arrays: dict[str, list] = {}
    for key in ("h_struct", "h_couple", "c_mod_input", "c_mod_output", "c_mod_total"):
        arrays[key] = [g.get(key) for g in generations]
    return xs, arrays


# ---------------------------------------------------------------------------
# Core plotting helpers
# ---------------------------------------------------------------------------

def _style_ax(ax, title: str, ylabel: str, xlabel: str = "Generation"):
    ax.set_title(title, pad=8, fontweight="medium")
    ax.set_xlabel(xlabel, labelpad=5)
    ax.set_ylabel(ylabel, labelpad=5)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True, nbins=6))
    ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=5, prune="both"))
    ax.tick_params(which="both", direction="out", length=4)
    ax.set_facecolor("#FAFAFA")


def _set_ylim_with_padding(ax, ys_clean: list[float]) -> None:
    y_min, y_max = min(ys_clean), max(ys_clean)
    y_range = y_max - y_min
    if y_range < 1e-6:
        pad = max(abs(y_min) * 0.05, 0.05)
    else:
        pad = y_range * 0.15
    new_lo = y_min - pad
    new_hi = y_max + pad
    # For multi-run: expand window to include all series.
    # Only expand if axes already has plotted artists (subsequent series);
    # on the first series just set directly to avoid anchoring at matplotlib's
    # default [0, 1] which pushes data to the top of the panel.
    if ax.lines or ax.collections:
        cur_lo, cur_hi = ax.get_ylim()
        ax.set_ylim(min(cur_lo, new_lo), max(cur_hi, new_hi))
    else:
        ax.set_ylim(new_lo, new_hi)


def _plot_series(
    ax,
    xs: list,
    ys_raw: list,
    color: str,
    scale: float = 1.0,
    label: str | None = None,
    annotate_last: bool = True,
) -> bool:
    """Plot one metric series.  Returns True if data was plotted, False if empty."""
    pairs = [(x, y * scale) for x, y in zip(xs, ys_raw) if y is not None]
    if not pairs:
        if len(xs) >= 2:
            ax.set_xlim(xs[0] - 0.5, xs[-1] + 0.5)
        ax.set_ylim(0, 1)
        if not label:  # only show "no data" for single-run (no label)
            ax.text(0.5, 0.5, "no data", transform=ax.transAxes,
                    ha="center", va="center", color="#AAAAAA", fontsize=10)
            ax.set_yticks([])
        return False

    xs_clean, ys_clean = zip(*pairs)
    xs_clean = list(xs_clean)
    ys_clean = list(ys_clean)

    if xs:
        ax.set_xlim(xs[0] - 0.5, xs[-1] + 0.5)

    _set_ylim_with_padding(ax, ys_clean)

    # Connecting line
    ax.plot(xs_clean, ys_clean, color=color, lw=_LW, alpha=_LINE_ALPHA,
            zorder=2, label=label)

    # Scatter dots
    ax.scatter(xs_clean, ys_clean, color=color, s=_DOT_SIZE, alpha=_SCATTER_ALPHA,
               zorder=3, edgecolors="white", linewidths=0.5)

    # Rolling mean (single-run only, when no label crowding needed)
    n = len(ys_clean)
    if n >= 5 and label is None:
        window = max(3, n // 5)
        kernel = np.ones(window) / window
        padded = np.pad(ys_clean, (window // 2, window - window // 2 - 1), mode="edge")
        smoothed = np.convolve(padded, kernel, mode="valid")[:n]
        ax.plot(xs_clean, smoothed, color=color, lw=2.2, alpha=0.9, zorder=4)

    # Final-value annotation (single-run only)
    if annotate_last and label is None:
        ax.annotate(
            f"{ys_clean[-1]:.2f}",
            xy=(xs_clean[-1], ys_clean[-1]),
            xytext=(6, 4), textcoords="offset points",
            fontsize=7.5, color=color, fontweight="semibold",
        )

    return True


def _apply_genid_xticks(axes, genid_labels: list[str], n_gens: int) -> None:
    for ax in axes:
        tick_positions = ax.get_xticks()
        valid = [int(t) for t in tick_positions if 0 <= int(t) < n_gens]
        if valid:
            ax.set_xticks(valid)
            ax.set_xticklabels([genid_labels[t] for t in valid], rotation=0)


def _build_figure_axes():
    """Create the standard 2-row, 5-panel figure.  Returns (fig, axes_dict)."""
    fig = plt.figure(figsize=(13, 5.5))
    gs = fig.add_gridspec(
        2, 6, hspace=0.52, wspace=0.55,
        left=0.07, right=0.97, top=0.92, bottom=0.12,
    )
    return fig, {
        "h_struct":    fig.add_subplot(gs[0, 0:3]),
        "h_couple":    fig.add_subplot(gs[0, 3:6]),
        "c_mod_input": fig.add_subplot(gs[1, 0:2]),
        "c_mod_output":fig.add_subplot(gs[1, 2:4]),
        "c_mod_total": fig.add_subplot(gs[1, 4:6]),
    }


def _save_fig(fig, base_path: str, svg: bool = False) -> None:
    fig.savefig(base_path)
    print(f"Saved: {base_path}")
    if svg:
        svg_path = base_path.replace(".png", ".svg")
        fig.savefig(svg_path)
        print(f"Saved: {svg_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Public API — single run
# ---------------------------------------------------------------------------

def plot_metrics(exp_dir: str, svg: bool = False) -> None:
    """Generate the 5-panel metric figure for a single run and export metrics_summary.json.

    Saves:
      <exp_dir>/metrics_plot.png        — combined 5-panel figure
      <exp_dir>/metrics_summary.json    — portable JSON for compare plots
      <exp_dir>/metrics_<key>.png       — individual panel PNGs
    """
    rows = _extract_from_run_dir(exp_dir)
    n_gens = len(rows)
    if n_gens == 0:
        warnings.warn(f"plot_metrics: no metric data found in {exp_dir}")
        return

    xs, arrays = _generations_to_arrays(rows)
    genid_labels = [str(r["genid"]) for r in rows]

    # Export portable JSON (so future compare plots can use it without re-reading the run)
    json_path = export_metrics_json(exp_dir)
    print(f"Metrics summary: {json_path}")

    with mpl.rc_context(_NEURIPS_RC):
        fig, axes = _build_figure_axes()
        fig.text(0.5, 0.975, "Codebase Entropy Trajectory",
                 ha="center", va="top", fontsize=12, fontweight="semibold", color="#333333")

        for key, title, ylabel, color_key, scale in _PANEL_SPECS:
            ax = axes[key]
            _style_ax(ax, title, ylabel)
            _plot_series(ax, xs, arrays[key], color=_METRIC_COLOR[color_key], scale=scale)

        _apply_genid_xticks(list(axes.values()), genid_labels, n_gens)
        _save_fig(fig, os.path.join(exp_dir, "metrics_plot.png"), svg=svg)

    # Individual per-metric plots
    _save_individual_plots(exp_dir, xs, arrays, genid_labels, svg=svg)


def _save_individual_plots(
    exp_dir: str, xs: list, arrays: dict, genid_labels: list[str], svg: bool = False,
) -> None:
    token_keys = {"c_mod_input", "c_mod_output", "c_mod_total"}
    with mpl.rc_context(_NEURIPS_RC):
        for key, title, ylabel, color_key, scale in _PANEL_SPECS:
            fig, ax = plt.subplots(1, 1, figsize=(4.2, 3.1))
            _style_ax(ax, title, ylabel)
            _plot_series(ax, xs, arrays[key], color=_METRIC_COLOR[color_key], scale=scale)
            n = len(genid_labels)
            valid = [int(t) for t in ax.get_xticks() if 0 <= int(t) < n]
            if valid:
                ax.set_xticks(valid)
                ax.set_xticklabels([genid_labels[t] for t in valid])
            _save_fig(fig, os.path.join(exp_dir, f"metrics_{key}.png"), svg=svg)


# ---------------------------------------------------------------------------
# Public API — multi-run comparison
# ---------------------------------------------------------------------------

def plot_metrics_compare(
    runs: list[dict],
    output_path: str,
    svg: bool = False,
) -> None:
    """Overlay metrics from multiple runs on the same 5-panel figure.

    Args:
        runs: list of dicts, each with:
              - "path":  path to metrics_summary.json OR a run directory
              - "label": legend label for this run
              - "color": (optional) override color hex string
        output_path: path for the output PNG (e.g. "outputs/comparison.png")
        svg:         also save an SVG version

    Example:
        plot_metrics_compare(
            runs=[
                {"label": "Null model",        "path": "outputs/null/metrics_summary.json"},
                {"label": "Entropy selection", "path": "outputs/entropy_sel/metrics_summary.json"},
            ],
            output_path="outputs/comparison.png",
        )
    """
    if not runs:
        raise ValueError("runs must be a non-empty list")

    # Load all run data
    loaded: list[tuple[str, str, list, list, dict]] = []  # (label, color, xs, genid_labels, arrays)
    for i, run in enumerate(runs):
        path = run["path"]
        label = run.get("label", f"run_{i}")
        color = run.get("color", _RUN_PALETTE[i % len(_RUN_PALETTE)])

        # Accept run dir or metrics_summary.json path
        if os.path.isdir(path):
            summary_json = os.path.join(path, "metrics_summary.json")
            if not os.path.exists(summary_json):
                # Auto-export if missing
                export_metrics_json(path)
            data = load_metrics_json(summary_json)
        else:
            data = load_metrics_json(path)

        rows = data["generations"]
        xs, arrays = _generations_to_arrays(rows)
        genid_labels = [str(r["genid"]) for r in rows]
        loaded.append((label, color, xs, genid_labels, arrays))

    with mpl.rc_context(_NEURIPS_RC):
        fig, axes = _build_figure_axes()
        fig.text(0.5, 0.975, "Codebase Metric Comparison",
                 ha="center", va="top", fontsize=12, fontweight="semibold", color="#333333")

        for key, title, ylabel, _color_key, scale in _PANEL_SPECS:
            ax = axes[key]
            _style_ax(ax, title, ylabel)
            any_data = False
            for label, color, xs, genid_labels, arrays in loaded:
                plotted = _plot_series(
                    ax, xs, arrays[key], color=color,
                    scale=scale, label=label, annotate_last=False,
                )
                any_data = any_data or plotted
            if any_data:
                ax.legend(loc="best", fontsize=8)

        # Use the longest run's genid labels for x-ticks
        longest = max(loaded, key=lambda t: len(t[2]))
        _apply_genid_xticks(list(axes.values()), longest[3], len(longest[2]))

        _save_fig(fig, output_path, svg=svg)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Plot entropy and token usage trajectories.  "
            "Pass --compare to overlay multiple runs."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single run
  python -m analysis.plot_metrics --path outputs/run_null/

  # Compare two runs
  python -m analysis.plot_metrics \\
      --path    outputs/run_null/     --label "Null model" \\
      --compare outputs/run_entropy/  --label "Entropy selection" \\
      --output  outputs/comparison.png
""",
    )
    parser.add_argument(
        "--path", type=str, required=True,
        help="Primary run directory (must contain archive.jsonl).",
    )
    parser.add_argument(
        "--compare", type=str, action="append", default=[],
        metavar="RUN_DIR",
        help="Additional run directory to overlay (repeatable).",
    )
    parser.add_argument(
        "--label", type=str, action="append", default=[],
        help="Legend label for each run, in order: primary first, then --compare runs.",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output path for comparison plot (default: <primary_dir>/metrics_comparison.png).",
    )
    parser.add_argument("--svg", action="store_true", help="Also save SVG versions.")
    args = parser.parse_args()

    if not args.compare:
        # Single-run mode
        plot_metrics(args.path, svg=args.svg)
    else:
        # Multi-run comparison mode
        all_paths = [args.path] + args.compare
        labels = args.label or []
        runs = []
        for i, p in enumerate(all_paths):
            runs.append({
                "path":  p,
                "label": labels[i] if i < len(labels) else os.path.basename(p.rstrip("/")),
            })
        out = args.output or os.path.join(args.path, "metrics_comparison.png")
        plot_metrics_compare(runs, output_path=out, svg=args.svg)
