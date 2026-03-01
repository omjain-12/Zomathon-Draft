"""
simulation.py — Dispatch simulation and rider-wait modelling.

Models the downstream operational impact of KPT accuracy by simulating
rider dispatch timing and deriving rider-wait and food-cold-wait metrics.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .config import DISPATCH_FRACTION, RANDOM_SEED

logger = logging.getLogger(__name__)

# Rider-wait threshold (minutes) above which an order is flagged as delayed
_DELAY_THRESHOLD_MIN: float = 3.0


def simulate_dispatch(
    df: pd.DataFrame,
    pred_col: str,
    dispatch_fraction: float = DISPATCH_FRACTION,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Simulate rider dispatch and compute wait-time metrics.

    The rider is dispatched when the elapsed time since order placement
    reaches ``dispatch_fraction × predicted_kpt``.  This mirrors a common
    platform dispatch heuristic.

    Parameters
    ----------
    df:
        DataFrame with fused KPT predictions (output of
        :meth:`~fusion_engine.KPTSignalFusion.fuse_signals` or the raw
        order table when evaluating the baseline).
    pred_col:
        Column name containing KPT predictions to dispatch against.
    dispatch_fraction:
        Fraction of predicted KPT at which the rider is dispatched (0–1).
    seed:
        Reserved for future stochastic extensions.

    Returns
    -------
    pd.DataFrame
        Input augmented with:

        * ``dispatch_time_min``  — simulated dispatch time (from order placement).
        * ``sim_arrival_min``    — simulated rider arrival at merchant.
        * ``rider_wait_min``     — time rider waits for food (max 0).
        * ``food_cold_wait_min`` — time food waits for rider (max 0).
        * ``eta_error_min``      — prediction error: true_kpt − predicted_kpt.
        * ``delay_flag``         — True when rider_wait > _DELAY_THRESHOLD_MIN.
    """
    if pred_col not in df.columns:
        raise KeyError(f"simulate_dispatch: column '{pred_col}' not found.")

    result = df.copy()

    result["dispatch_time_min"] = result[pred_col] * dispatch_fraction
    result["sim_arrival_min"] = (
        result["dispatch_time_min"] + result["rider_travel_time_min"]
    )
    result["rider_wait_min"] = (
        result["true_kpt_min"] - result["sim_arrival_min"]
    ).clip(lower=0.0)
    result["food_cold_wait_min"] = (
        result["sim_arrival_min"] - result["true_kpt_min"]
    ).clip(lower=0.0)
    result["eta_error_min"] = result["true_kpt_min"] - result[pred_col]
    result["delay_flag"] = result["rider_wait_min"] > _DELAY_THRESHOLD_MIN

    logger.info(
        "Dispatch simulation on '%s': avg rider wait=%.3f min, delay rate=%.1f%%.",
        pred_col,
        result["rider_wait_min"].mean(),
        result["delay_flag"].mean() * 100,
    )
    return result
