"""NeurIPS-quality metric trajectory plots for entropy and token usage.

Reads per-generation metrics from archive.jsonl and produces publication-quality
plots of:
  - Structural Entropy (H_struct)       — cyan
  - Coupling Entropy   (H_couple)       — magenta
  - Input tokens   (C_mod component)   — gold
  - Output tokens  (C_mod component)   — rose
  - Total tokens   (C_mod / C_mod)     — cyan (lighter tint)

Usage (standalone):
    python analysis/plot_metrics.py --path outputs/generate_20260325_212836_252984/

Programmatic:
    from analysis.plot_metrics import plot_metrics
    plot_metrics(exp_dir)
"""

from __future__ import annotations

import argparse
import os
import pathlib
import warnings

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from utils.gl_utils import load_archive_data

__all__ = ["plot_metrics"]

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
# Color palette — cyan / magenta / gold / rose
# ---------------------------------------------------------------------------

_C = {
    "h_struct":     "#17C3B2",   # teal-cyan
    "h_couple":     "#D62598",   # magenta
    "input_tokens": "#F5C518",   # gold
    "output_tokens":"#FF6B8A",   # rose
    "total_tokens": "#7B5EA7",   # violet (distinct from the above)
    "scatter_alpha": 0.85,
    "line_alpha": 0.55,
    "dot_size": 28,
    "lw": 1.8,
}

# Panel definitions: (key_path, label, y_axis_label, color_key, scale)
# key_path is a tuple of keys to drill into the metrics dict
_PANELS = [
    (("h_struct",),          r"Structural Entropy $H_{\rm struct}$",  "bits",         "h_struct",     1.0),
    (("h_couple",),          r"Coupling Entropy $H_{\rm couple}$",    "bits",         "h_couple",     1.0),
    (("c_mod", "input_tokens"),  "Input Tokens",                      "tokens (k)",   "input_tokens", 1e-3),
    (("c_mod", "output_tokens"), "Output Tokens",                     "tokens (k)",   "output_tokens",1e-3),
    (("c_mod", "total_tokens"),  "Total Tokens (C_mod)",              "tokens (k)",   "total_tokens", 1e-3),
]


# ---------------------------------------------------------------------------
# Data extraction
# ---------------------------------------------------------------------------

def _read_metadata_metrics(exp_dir: str, genid) -> dict | None:
    """Read metrics from a generation's metadata.json, or None if absent."""
    import json as _json
    meta_path = os.path.join(exp_dir, f"gen_{genid}", "metadata.json")
    if not os.path.exists(meta_path):
        return None
    try:
        with open(meta_path) as f:
            return _json.load(f).get("metrics")
    except Exception:
        return None


def _extract_metrics(exp_dir: str) -> dict[str, list]:
    """Return {metric_key: [value_at_gen_0, value_at_gen_1, ...]} in archive order.

    Skips the 'initial' entry.  Metrics are sourced from archive.jsonl when
    present (new runs), and fall back to each generation's metadata.json
    (backfilled runs).  Generations with no metrics anywhere are still included
    with None values so the x-axis stays consistent.
    """
    archive_path = os.path.join(os.path.normpath(exp_dir), "archive.jsonl")
    archive_data = load_archive_data(archive_path, last_only=False)

    seen: set = set()
    ordered: list[tuple] = []
    for entry in archive_data:
        genid = entry.get("current_genid")
        if genid in seen or genid == "initial":
            continue
        seen.add(genid)
        # Prefer archive.jsonl metrics; fall back to metadata.json
        metrics = entry.get("metrics") or _read_metadata_metrics(exp_dir, genid)
        ordered.append((genid, metrics))

    result: dict[str, list] = {
        "genids": [],
        "h_struct": [],
        "h_couple": [],
        "input_tokens": [],
        "output_tokens": [],
        "total_tokens": [],
    }

    for genid, metrics in ordered:
        result["genids"].append(genid)
        if metrics is None:
            result["h_struct"].append(None)
            result["h_couple"].append(None)
            result["input_tokens"].append(None)
            result["output_tokens"].append(None)
            result["total_tokens"].append(None)
        else:
            result["h_struct"].append(metrics.get("h_struct"))
            result["h_couple"].append(metrics.get("h_couple"))
            c_mod = metrics.get("c_mod") or {}
            result["input_tokens"].append(c_mod.get("input_tokens"))
            result["output_tokens"].append(c_mod.get("output_tokens"))
            result["total_tokens"].append(c_mod.get("total_tokens"))

    return result


# ---------------------------------------------------------------------------
# Core plotting helpers
# ---------------------------------------------------------------------------

def _style_ax(ax, title: str, ylabel: str, xlabel: str = "Generation"):
    """Apply NeurIPS spine / tick / label styling to an axis."""
    ax.set_title(title, pad=8, fontweight="medium")
    ax.set_xlabel(xlabel, labelpad=5)
    ax.set_ylabel(ylabel, labelpad=5)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True, nbins=6))
    ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=5, prune="both"))
    ax.tick_params(which="both", direction="out", length=4)
    # Light background to separate from page
    ax.set_facecolor("#FAFAFA")


def _plot_series(
    ax,
    xs: list,
    ys_raw: list,
    color: str,
    scale: float = 1.0,
    label: str | None = None,
):
    """Plot a metric series with dots + connecting line + optional rolling mean."""
    # Filter None values
    pairs = [(x, y * scale) for x, y in zip(xs, ys_raw) if y is not None]
    if not pairs:
        # Still set sensible x-axis limits so the panel looks consistent
        if len(xs) >= 2:
            ax.set_xlim(xs[0] - 0.5, xs[-1] + 0.5)
        ax.set_ylim(0, 1)
        ax.text(
            0.5, 0.5, "no data",
            transform=ax.transAxes,
            ha="center", va="center",
            color="#AAAAAA", fontsize=10,
        )
        ax.set_yticks([])
        return

    xs_clean, ys_clean = zip(*pairs)
    xs_clean = list(xs_clean)
    ys_clean = list(ys_clean)

    # Ensure x-axis covers the full generation range even if some are missing
    if xs:
        ax.set_xlim(xs[0] - 0.5, xs[-1] + 0.5)

    # Y-padding: near-flat lines get a minimum visual range so they don't look
    # like noise amplified to fill the axis
    y_min, y_max = min(ys_clean), max(ys_clean)
    y_range = y_max - y_min
    if y_range < 1e-6:
        pad = max(abs(y_min) * 0.05, 0.05)
        ax.set_ylim(y_min - pad, y_max + pad)
    else:
        pad = y_range * 0.15
        ax.set_ylim(y_min - pad, y_max + pad)

    # Connecting line (thin, semi-transparent)
    ax.plot(
        xs_clean, ys_clean,
        color=color, lw=_C["lw"],
        alpha=_C["line_alpha"],
        zorder=2,
        label=label,
    )

    # Individual generation scatter dots
    ax.scatter(
        xs_clean, ys_clean,
        color=color, s=_C["dot_size"],
        alpha=_C["scatter_alpha"],
        zorder=3,
        edgecolors="white", linewidths=0.5,
    )

    # Rolling mean overlay when there are enough points
    n = len(ys_clean)
    window = max(3, n // 5)
    if n >= 5:
        kernel = np.ones(window) / window
        padded = np.pad(ys_clean, (window // 2, window - window // 2 - 1), mode="edge")
        smoothed = np.convolve(padded, kernel, mode="valid")[:n]
        ax.plot(
            xs_clean, smoothed,
            color=color, lw=2.2,
            alpha=0.9,
            zorder=4,
            ls="-",
            label=f"rolling mean (w={window})",
        )

    # Annotate final value
    last_x, last_y = xs_clean[-1], ys_clean[-1]
    ax.annotate(
        f"{last_y:.2f}",
        xy=(last_x, last_y),
        xytext=(6, 4),
        textcoords="offset points",
        fontsize=7.5,
        color=color,
        fontweight="semibold",
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def plot_metrics(exp_dir: str, svg: bool = False) -> None:
    """Generate a 5-panel metric trajectory figure and save to exp_dir.

    Saves:
      metrics_plot.png  (and metrics_plot.svg if svg=True)

    Args:
        exp_dir: Path to the run output directory containing archive.jsonl.
        svg:     Also save an SVG version for vector editing.
    """
    data = _extract_metrics(exp_dir)
    n_gens = len(data["genids"])
    if n_gens == 0:
        warnings.warn(f"plot_metrics: no metric data found in {exp_dir}")
        return

    # Use generation index as x-axis (0, 1, 2, ...) for clean labelling
    xs = list(range(n_gens))

    with mpl.rc_context(_NEURIPS_RC):
        # --- figure layout: 2-row, 3-col; entropy on top (2 panels centered),
        #     tokens on bottom (3 panels)
        fig = plt.figure(figsize=(13, 5.5))
        gs = fig.add_gridspec(
            2, 6,
            hspace=0.52,
            wspace=0.55,
            left=0.07, right=0.97, top=0.92, bottom=0.12,
        )

        # Top row: 2 entropy panels (each spanning 3 cols of 6)
        ax_h_struct = fig.add_subplot(gs[0, 0:3])
        ax_h_couple = fig.add_subplot(gs[0, 3:6])

        # Bottom row: 3 token panels (each spanning 2 cols of 6)
        ax_input  = fig.add_subplot(gs[1, 0:2])
        ax_output = fig.add_subplot(gs[1, 2:4])
        ax_total  = fig.add_subplot(gs[1, 4:6])

        # ── top-row section label ──────────────────────────────────────────
        fig.text(
            0.5, 0.975, "Codebase Entropy Trajectory",
            ha="center", va="top", fontsize=12, fontweight="semibold",
            color="#333333",
        )

        # ── H_struct ──────────────────────────────────────────────────────
        _style_ax(ax_h_struct, r"Structural Entropy  $H_{\rm struct}$", "entropy (bits)")
        _plot_series(ax_h_struct, xs, data["h_struct"], color=_C["h_struct"])

        # ── H_couple ──────────────────────────────────────────────────────
        _style_ax(ax_h_couple, r"Coupling Entropy  $H_{\rm couple}$", "entropy (bits)")
        _plot_series(ax_h_couple, xs, data["h_couple"], color=_C["h_couple"])

        # ── tokens ────────────────────────────────────────────────────────
        _style_ax(ax_input,  "Input Tokens", "tokens (k)")
        _plot_series(ax_input, xs, data["input_tokens"], color=_C["input_tokens"], scale=1e-3)

        _style_ax(ax_output, "Output Tokens", "tokens (k)")
        _plot_series(ax_output, xs, data["output_tokens"], color=_C["output_tokens"], scale=1e-3)

        _style_ax(ax_total, r"Total Tokens  ($C_{\rm mod}$)", "tokens (k)")
        _plot_series(ax_total, xs, data["total_tokens"], color=_C["total_tokens"], scale=1e-3)

        # ── x-tick labels: map index → genid ──────────────────────────────
        genid_labels = [str(g) for g in data["genids"]]
        for ax in (ax_h_struct, ax_h_couple, ax_input, ax_output, ax_total):
            tick_positions = ax.get_xticks()
            valid_ticks = [int(t) for t in tick_positions if 0 <= int(t) < n_gens]
            if valid_ticks:
                ax.set_xticks(valid_ticks)
                ax.set_xticklabels([genid_labels[t] for t in valid_ticks], rotation=0)

        # ── save ──────────────────────────────────────────────────────────
        out_png = os.path.join(exp_dir, "metrics_plot.png")
        fig.savefig(out_png)
        print(f"Metrics plot saved: {out_png}")

        if svg:
            out_svg = os.path.join(exp_dir, "metrics_plot.svg")
            fig.savefig(out_svg)
            print(f"Metrics plot saved: {out_svg}")

        plt.close(fig)

    # Also save individual plots for each metric panel
    _save_individual_plots(exp_dir, xs, data, genid_labels, svg=svg)


def _save_individual_plots(
    exp_dir: str,
    xs: list,
    data: dict,
    genid_labels: list[str],
    svg: bool = False,
) -> None:
    """Save a separate high-quality PNG per metric for use in papers / slides."""
    metric_specs = [
        ("h_struct",      r"Structural Entropy  $H_{\rm struct}$",  "entropy (bits)", _C["h_struct"]),
        ("h_couple",      r"Coupling Entropy  $H_{\rm couple}$",    "entropy (bits)", _C["h_couple"]),
        ("input_tokens",  "Input Tokens",                            "tokens (k)",     _C["input_tokens"]),
        ("output_tokens", "Output Tokens",                           "tokens (k)",     _C["output_tokens"]),
        ("total_tokens",  r"Total Tokens  ($C_{\rm mod}$)",         "tokens (k)",     _C["total_tokens"]),
    ]
    token_keys = {"input_tokens", "output_tokens", "total_tokens"}

    with mpl.rc_context(_NEURIPS_RC):
        for key, title, ylabel, color in metric_specs:
            fig, ax = plt.subplots(1, 1, figsize=(4.2, 3.1))
            scale = 1e-3 if key in token_keys else 1.0
            _plot_series(ax, xs, data[key], color=color, scale=scale)
            _style_ax(ax, title, ylabel)

            # x-tick labels → genids
            n = len(data["genids"])
            tick_positions = ax.get_xticks()
            valid_ticks = [int(t) for t in tick_positions if 0 <= int(t) < n]
            if valid_ticks:
                ax.set_xticks(valid_ticks)
                ax.set_xticklabels([genid_labels[t] for t in valid_ticks])

            out_png = os.path.join(exp_dir, f"metrics_{key}.png")
            fig.savefig(out_png)
            if svg:
                fig.savefig(os.path.join(exp_dir, f"metrics_{key}.svg"))
            plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot entropy and token usage trajectory from archive.jsonl."
    )
    parser.add_argument(
        "--path",
        type=str,
        required=True,
        help="Path to the run output directory (must contain archive.jsonl).",
    )
    parser.add_argument(
        "--svg",
        action="store_true",
        help="Also save SVG versions of the plots.",
    )
    args = parser.parse_args()
    plot_metrics(args.path, svg=args.svg)
