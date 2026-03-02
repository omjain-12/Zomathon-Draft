"""
plotting.py — Publication-quality visualisation utilities.

All functions follow the same conventions:
  * Receive a DataFrame and keyword options.
  * Save a PNG side-effect when ``save_path`` is provided.
  * Return the matplotlib Figure object.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .config import PALETTE

logger = logging.getLogger(__name__)

# Default figure output directory (relative to the notebook)
_DEFAULT_OUTPUT_DIR = Path("figures")


def _ensure_dir(path: Optional[Path]) -> None:
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)


def plot_error_distributions(
    df: pd.DataFrame,
    sample_n: int = 10_000,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """3-panel figure: KPT distribution, absolute error CDF, rider-wait box.

    Parameters
    ----------
    df:
        Fused + simulated order DataFrame.
    sample_n:
        Max rows to sample for scatter/histogram plots.
    save_path:
        If provided, save figure as PNG.

    Returns
    -------
    matplotlib.figure.Figure
    """
    sample = df.sample(min(sample_n, len(df)), random_state=42)
    baseline_err = (sample["reported_FOR_delta_min"] - sample["true_kpt_min"]).abs()
    proposed_err = (sample["predicted_kpt_min"] - sample["true_kpt_min"]).abs()

    fig = plt.figure(figsize=(16, 5))
    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)

    # Panel 1 — KPT distribution
    ax1 = fig.add_subplot(gs[0])
    ax1.hist(sample["true_kpt_min"], bins=50, color=PALETTE[0], alpha=0.85, edgecolor="white")
    ax1.set(title="True KPT Distribution", xlabel="KPT (min)", ylabel="Count")
    ax1.axvline(sample["true_kpt_min"].mean(), color=PALETTE[1], linestyle="--", label="Mean")
    ax1.legend(frameon=False)

    # Panel 2 — Absolute error CDF
    ax2 = fig.add_subplot(gs[1])
    for errors, label, color in [
        (baseline_err, "Baseline (FOR)", PALETTE[1]),
        (proposed_err, "Proposed (Fusion)", PALETTE[0]),
    ]:
        sorted_e = np.sort(errors)
        cdf = np.arange(1, len(sorted_e) + 1) / len(sorted_e)
        ax2.plot(sorted_e, cdf, color=color, lw=2, label=label)
    ax2.axvline(2.0, color="gray", linestyle=":", alpha=0.7, label="2-min threshold")
    ax2.set(title="Absolute Error CDF", xlabel="|Error| (min)", ylabel="Cumulative fraction")
    ax2.legend(frameon=False)

    # Panel 3 — Rider wait box
    ax3 = fig.add_subplot(gs[2])
    wait_data = pd.DataFrame(
        {
            "Baseline": simulate_baseline_wait(sample),
            "Proposed": sample["rider_wait_min"],
        }
    )
    wait_data.plot.box(ax=ax3, color={"medians": PALETTE[1]}, patch_artist=True)
    ax3.set(title="Rider Wait Distribution", ylabel="Rider wait (min)")

    fig.suptitle("KPT Estimation — Evaluation Overview", fontsize=14, fontweight="bold")

    if save_path:
        _ensure_dir(save_path)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Saved figure: %s", save_path)
    return fig


def simulate_baseline_wait(sample: pd.DataFrame) -> pd.Series:
    """Lightweight inline baseline-wait estimate for plot Panel 3."""
    from .simulation import simulate_dispatch

    base = simulate_dispatch(sample, "reported_FOR_delta_min")
    return base["rider_wait_min"]


def plot_wait_time_comparison(
    segment_results: Dict[str, pd.Series],
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Horizontal bar chart comparing rider wait improvment across segments.

    Parameters
    ----------
    segment_results:
        Mapping of segment name → Series from
        :func:`~evaluation.segment_experiment`.
    save_path:
        If provided, save figure as PNG.

    Returns
    -------
    matplotlib.figure.Figure
    """
    records = {k: v for k, v in segment_results.items() if v is not None}
    if not records:
        logger.warning("No segment results to plot.")
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No segments available", ha="center", va="center")
        return fig

    seg_df = pd.DataFrame(records).T
    labels = seg_df.index.tolist()
    baseline = seg_df["avg_wait_baseline"].values
    proposed = seg_df["avg_wait_proposed"].values
    x = np.arange(len(labels))
    width = 0.38

    fig, ax = plt.subplots(figsize=(10, max(4, len(labels) * 0.7)))
    bars_b = ax.barh(x + width / 2, baseline, width, label="Baseline", color=PALETTE[1], alpha=0.85)
    bars_p = ax.barh(x - width / 2, proposed, width, label="Proposed", color=PALETTE[0], alpha=0.85)

    for i, (b, p) in enumerate(zip(baseline, proposed)):
        pct = (b - p) / b * 100 if b > 0 else 0
        ax.text(max(b, p) + 0.05, i, f"−{pct:.1f}%", va="center", fontsize=9, color=PALETTE[2])

    ax.set(yticks=x, yticklabels=labels, xlabel="Avg rider wait (min)",
           title="Rider Wait Reduction by Segment")
    ax.legend(frameon=False)
    ax.invert_yaxis()
    plt.tight_layout()

    if save_path:
        _ensure_dir(save_path)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Saved figure: %s", save_path)
    return fig


def plot_ablation(
    ablation: pd.DataFrame,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """2-panel bar chart for ablation: MAE and avg rider wait vs config.

    Parameters
    ----------
    ablation:
        Output of :func:`~evaluation.run_ablation_study`.
    save_path:
        If provided, save figure as PNG.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, metric, title, ylabel in [
        (axes[0], "MAE", "MAE by Configuration", "MAE (min)"),
        (axes[1], "avg_rider_wait_min", "Avg Rider Wait by Configuration", "Avg wait (min)"),
    ]:
        if metric not in ablation.columns:
            ax.text(0.5, 0.5, f"'{metric}' not available", ha="center", va="center")
            ax.set_title(title)
            continue
        vals = ablation[metric]
        colors = [PALETTE[0] if i == len(vals) - 1 else PALETTE[4] for i in range(len(vals))]
        bars = ax.bar(range(len(vals)), vals, color=colors, edgecolor="white", alpha=0.9)
        ax.set(xticks=range(len(vals)), title=title, ylabel=ylabel)
        ax.set_xticklabels(vals.index, rotation=25, ha="right", fontsize=9)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + vals.max() * 0.01,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=8)

    fig.suptitle("Ablation Study — Signal Configuration Comparison", fontsize=13, fontweight="bold")
    plt.tight_layout()

    if save_path:
        _ensure_dir(save_path)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Saved figure: %s", save_path)
    return fig
