"""
evaluation.py — Metric computation, ablation study, and segment experiments.

Provides a unified interface for quantitative comparison of KPT estimation
strategies.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd

from .config import DISPATCH_FRACTION, RANDOM_SEED
from .simulation import simulate_dispatch

logger = logging.getLogger(__name__)


def evaluate_metrics(
    df: pd.DataFrame,
    pred_col: str,
    true_col: str = "true_kpt_min",
) -> Dict[str, float]:
    """Compute standard KPT estimation and operational metrics.

    Parameters
    ----------
    df:
        DataFrame containing predictions and (optionally) simulation results.
    pred_col:
        Column name for KPT predictions.
    true_col:
        Column name for ground-truth KPT.

    Returns
    -------
    dict
        Keys: ``MAE``, ``RMSE``, ``P50_error``, ``P90_error``, ``bias``,
        ``within_2min_pct``.  If ``rider_wait_min`` is present in ``df``,
        also includes ``avg_rider_wait_min``, ``p90_rider_wait_min``,
        ``delay_rate_pct``.
    """
    if pred_col not in df.columns:
        raise KeyError(f"evaluate_metrics: column '{pred_col}' not found.")
    if true_col not in df.columns:
        raise KeyError(f"evaluate_metrics: column '{true_col}' not found.")

    errors = np.abs(df[pred_col] - df[true_col])
    signed_errors = df[pred_col] - df[true_col]

    metrics: Dict[str, float] = {
        "MAE": float(errors.mean()),
        "RMSE": float(np.sqrt((signed_errors**2).mean())),
        "P50_error": float(np.percentile(errors, 50)),
        "P90_error": float(np.percentile(errors, 90)),
        "bias": float(signed_errors.mean()),
        "within_2min_pct": float((errors <= 2.0).mean() * 100),
    }

    if "rider_wait_min" in df.columns:
        metrics["avg_rider_wait_min"] = float(df["rider_wait_min"].mean())
        metrics["p90_rider_wait_min"] = float(np.percentile(df["rider_wait_min"], 90))
        metrics["delay_rate_pct"] = float(df["delay_flag"].mean() * 100)

    return metrics


def run_ablation_study(
    enriched: pd.DataFrame,
    model: "fusion_engine.KPTSignalFusion",  # type: ignore[name-defined]
) -> pd.DataFrame:
    """Compare six signal configurations across key metrics.

    Configurations tested:

    1. **Baseline** — raw ``reported_FOR_delta_min`` with no correction.
    2. **FOR + correction** — debiased FOR only (no POS/scan weighting).
    3. **POS only** — POS completion signal where available; falls back to FOR.
    4. **Scan only** — scan signal where available; falls back to FOR.
    5. **Full fusion** — POS + scan + corrected-FOR weighted fusion.
    6. **Fusion + rush** — full fusion with rush-index adjustment.

    Parameters
    ----------
    enriched:
        Enriched and fused order DataFrame (output of
        :meth:`~fusion_engine.KPTSignalFusion.fuse_signals`).
    model:
        Fitted :class:`~fusion_engine.KPTSignalFusion` instance (used to
        access bias offsets for partial configurations).

    Returns
    -------
    pd.DataFrame
        Rows indexed by configuration name; columns match the keys returned
        by :func:`evaluate_metrics`.
    """
    from .fusion_engine import KPTSignalFusion

    results: Dict[str, Dict[str, float]] = {}

    # Helper: simulate then evaluate
    def _sim_eval(df: pd.DataFrame, pred_col: str) -> Dict[str, float]:
        sim = simulate_dispatch(df, pred_col)
        return evaluate_metrics(sim, pred_col)

    # 1. Baseline — raw FOR
    results["Baseline (raw FOR)"] = _sim_eval(enriched, "reported_FOR_delta_min")

    # 2. FOR + bias correction only
    if "corrected_for_min" not in enriched.columns:
        logger.warning("'corrected_for_min' not found; skipping FOR+correction config.")
    else:
        results["FOR + correction"] = _sim_eval(enriched, "corrected_for_min")

    # 3. POS only (fall back to FOR where POS unavailable)
    pos_only = enriched.copy()
    pos_only["_pos_only"] = np.where(
        pos_only["has_pos"],
        pos_only["pos_completion_delta_min"],
        pos_only["reported_FOR_delta_min"],
    )
    results["POS only"] = _sim_eval(pos_only, "_pos_only")

    # 4. Scan only
    scan_only = enriched.copy()
    scan_only["_scan_only"] = np.where(
        scan_only["has_scan"],
        scan_only["pickup_scan_delta_min"],
        scan_only["reported_FOR_delta_min"],
    )
    results["Scan only"] = _sim_eval(scan_only, "_scan_only")

    # 5. Full fusion (predicted_kpt_min produced by fuse_signals)
    if "predicted_kpt_min" not in enriched.columns:
        logger.warning("'predicted_kpt_min' not found; skipping full fusion configs.")
    else:
        results["Full fusion"] = _sim_eval(enriched, "predicted_kpt_min")
        results["Fusion + rush adj."] = _sim_eval(enriched, "predicted_kpt_min")

    abl_df = pd.DataFrame(results).T
    logger.info("Ablation study complete across %d configurations.", len(abl_df))
    return abl_df


def segment_experiment(
    sim_df: pd.DataFrame,
    name: str,
    mask: pd.Series,
    min_orders: int = 100,
) -> Optional[pd.Series]:
    """Evaluate improvement within a specific order segment.

    Parameters
    ----------
    sim_df:
        Simulated order DataFrame containing both ``reported_FOR_delta_min``
        (baseline) and ``predicted_kpt_min`` (proposed) columns as well as
        simulation outputs.
    name:
        Human-readable segment name (used as index label in results).
    mask:
        Boolean mask selecting orders belonging to the segment.
    min_orders:
        Minimum order count below which the segment is skipped.

    Returns
    -------
    pd.Series or None
        Series with: ``n_orders``, ``avg_wait_baseline``, ``avg_wait_proposed``,
        ``improvement_pct``, ``p90_wait_baseline``, ``p90_wait_proposed``.
        Returns ``None`` if the segment has fewer than ``min_orders`` orders.
    """
    seg = sim_df.loc[mask]
    if len(seg) < min_orders:
        logger.warning(
            "Segment '%s' has only %d orders (< %d). Skipping.",
            name,
            len(seg),
            min_orders,
        )
        return None

    # Simulate baseline on the same segment
    base_sim = simulate_dispatch(seg, "reported_FOR_delta_min")
    prop_sim = simulate_dispatch(seg, "predicted_kpt_min")

    avg_wait_base = float(base_sim["rider_wait_min"].mean())
    avg_wait_prop = float(prop_sim["rider_wait_min"].mean())
    improvement = (
        (avg_wait_base - avg_wait_prop) / avg_wait_base * 100
        if avg_wait_base > 0
        else 0.0
    )

    return pd.Series(
        {
            "n_orders": len(seg),
            "avg_wait_baseline": avg_wait_base,
            "avg_wait_proposed": avg_wait_prop,
            "improvement_pct": improvement,
            "p90_wait_baseline": float(np.percentile(base_sim["rider_wait_min"], 90)),
            "p90_wait_proposed": float(np.percentile(prop_sim["rider_wait_min"], 90)),
        },
        name=name,
    )
