"""
bias_detection.py — FOR bias detection and merchant-level offset calibration.

The Food-Order-Ready (FOR) signal is human-reported and prone to reactive
marking: a merchant presses the button only after the rider has arrived,
not when the food is genuinely ready.  This module identifies such events
and estimates a per-merchant correction offset.
"""

from __future__ import annotations

import logging
from typing import Final

import numpy as np
import pandas as pd

from .config import BIAS_THRESHOLD_SEC, ROLLING_WINDOW

logger = logging.getLogger(__name__)

# Minimum calibration orders before falling back to global median
_MIN_CALIBRATION_ORDERS: Final[int] = 10

# Required input columns for bias detection
_REQUIRED_COLS: Final[tuple[str, ...]] = (
    "reported_FOR_delta_min",
    "rider_arrival_delta_min",
)


def detect_for_bias(
    orders: pd.DataFrame,
    threshold_sec: int = BIAS_THRESHOLD_SEC,
) -> pd.DataFrame:
    """Flag orders where the FOR signal is likely rider-triggered (reactive).

    A FOR event is classified as biased when the absolute gap between
    ``reported_FOR_delta_min`` and ``rider_arrival_delta_min`` is smaller
    than ``threshold_sec / 60`` minutes.  At 90 s this threshold comfortably
    catches reactive marks while avoiding false positives on fast kitchens.

    Parameters
    ----------
    orders:
        Raw order DataFrame (output of :func:`~data_generation.generate_synthetic_orders`).
    threshold_sec:
        Maximum |FOR − rider_arrival| gap (seconds) still considered reactive.

    Returns
    -------
    pd.DataFrame
        Input DataFrame augmented with two new columns:

        * ``for_rider_gap_min``   — signed gap (FOR minus rider arrival).
        * ``is_for_biased_flag``  — boolean; True means reactive FOR.

    Raises
    ------
    KeyError
        If any required column is absent from ``orders``.
    """
    missing = [c for c in _REQUIRED_COLS if c not in orders.columns]
    if missing:
        raise KeyError(f"detect_for_bias: missing columns {missing}")

    threshold_min: float = threshold_sec / 60.0
    result = orders.copy()
    result["for_rider_gap_min"] = (
        result["reported_FOR_delta_min"] - result["rider_arrival_delta_min"]
    )
    result["is_for_biased_flag"] = (
        result["for_rider_gap_min"].abs() < threshold_min
    )

    bias_rate = result["is_for_biased_flag"].mean()
    logger.info(
        "FOR bias detection complete.  Flagged %.1f%% of %d orders (threshold=%ds).",
        bias_rate * 100,
        len(result),
        threshold_sec,
    )
    return result


def compute_merchant_bias_offsets(
    enriched: pd.DataFrame,
) -> pd.DataFrame:
    """Estimate a per-merchant FOR correction offset using a proxy ground truth.

    The proxy ground truth is chosen as:

    * POS completion delta (``pos_completion_delta_min``) when available — the
      tightest machine-generated readiness signal.
    * Pickup scan delta (``pickup_scan_delta_min``) otherwise.

    Merchants with fewer than ``_MIN_CALIBRATION_ORDERS`` qualifying orders
    receive the global median offset as a fall-back to prevent over-fitting
    on sparse data.

    Parameters
    ----------
    enriched:
        Enriched order DataFrame (output of
        :func:`~signal_enrichment.enrich_with_signals`).

    Returns
    -------
    pd.DataFrame
        One row per merchant with columns:

        * ``merchant_id``        — merchant identifier.
        * ``bias_offset_min``    — median (FOR − proxy) in minutes.
        * ``calibration_orders`` — number of orders used for calibration.
    """
    df = enriched.copy()

    # Choose the best available proxy per row
    df["_proxy"] = np.where(
        df["pos_completion_delta_min"].notna(),
        df["pos_completion_delta_min"],
        df["pickup_scan_delta_min"],
    )
    df["_offset"] = df["reported_FOR_delta_min"] - df["_proxy"]

    # Drop rows without any proxy signal
    valid = df.dropna(subset=["_proxy", "_offset"])

    stats = (
        valid.groupby("merchant_id")["_offset"]
        .agg(bias_offset_min="median", calibration_orders="count")
        .reset_index()
    )

    global_median: float = float(valid["_offset"].median())
    low_data_mask = stats["calibration_orders"] < _MIN_CALIBRATION_ORDERS
    stats.loc[low_data_mask, "bias_offset_min"] = global_median

    logger.info(
        "Merchant bias offsets computed. %d merchants calibrated; "
        "%d fell back to global median (%.3f min).",
        (~low_data_mask).sum(),
        low_data_mask.sum(),
        global_median,
    )
    return stats[["merchant_id", "bias_offset_min", "calibration_orders"]]
