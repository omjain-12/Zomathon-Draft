"""
signal_enrichment.py — Feature enrichment layer for KPT estimation.

Adds composite features on top of the raw order signals:
  * binary availability flags for POS and scan
  * rush index (normalised concurrent-order pressure)
  * rolling merchant bias score (fraction of recent orders flagged biased)
  * signal confidence score (weighted availability, penalised for rush)
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .bias_detection import detect_for_bias
from .config import (
    BIAS_THRESHOLD_SEC,
    CONFIDENCE_FLOOR,
    ROLLING_WINDOW,
    W_FOR,
    W_POS,
    W_SCAN,
)

logger = logging.getLogger(__name__)


def enrich_with_signals(
    orders: pd.DataFrame,
    threshold_sec: int = BIAS_THRESHOLD_SEC,
    rolling_window: int = ROLLING_WINDOW,
) -> pd.DataFrame:
    """Augment raw orders with enriched signal and confidence features.

    This function is the primary feature-engineering step.  It first runs
    :func:`~bias_detection.detect_for_bias`, then derives several higher-level
    features used by :class:`~fusion_engine.KPTSignalFusion`.

    Parameters
    ----------
    orders:
        Raw order DataFrame from :func:`~data_generation.generate_synthetic_orders`.
    threshold_sec:
        Forwarded to :func:`~bias_detection.detect_for_bias`.
    rolling_window:
        Number of historical orders per merchant used for the rolling
        bias-score estimate.

    Returns
    -------
    pd.DataFrame
        Input enriched with:

        * ``for_rider_gap_min``     — |FOR − rider_arrival| (from bias detection).
        * ``is_for_biased_flag``    — boolean reactive-FOR flag.
        * ``has_pos``               — POS signal available for this order.
        * ``has_scan``              — scan signal available for this order.
        * ``rush_index``            — normalised [0, 1] concurrent-order pressure.
        * ``merchant_bias_score``   — rolling fraction of biased FORs per merchant.
        * ``signal_confidence``     — composite quality score for this order's signals.
    """
    enriched = detect_for_bias(orders, threshold_sec=threshold_sec)

    # ── Signal availability flags ────────────────────────────────────────────
    enriched["has_pos"] = enriched["pos_completion_delta_min"].notna()
    enriched["has_scan"] = enriched["pickup_scan_delta_min"].notna()

    # ── Rush index ───────────────────────────────────────────────────────────
    # Normalise total concurrent order pressure to [0, 1]
    total_concurrent = (
        enriched["platform_orders_concurrent"] + enriched["external_orders_count"]
    )
    max_concurrent = total_concurrent.max()
    enriched["rush_index"] = (
        total_concurrent / max_concurrent if max_concurrent > 0 else 0.0
    )

    # ── Rolling merchant bias score ──────────────────────────────────────────
    # Sort by order_time_min so the rolling window is chronologically meaningful
    enriched = enriched.sort_values(["merchant_id", "order_time_min"]).reset_index(
        drop=True
    )
    enriched["merchant_bias_score"] = (
        enriched.groupby("merchant_id")["is_for_biased_flag"]
        .transform(
            lambda x: x.rolling(window=rolling_window, min_periods=1).mean()
        )
    )

    # ── Signal confidence score ──────────────────────────────────────────────
    # Base confidence: weighted signal availability
    base_confidence = (
        enriched["has_pos"].astype(float) * W_POS
        + enriched["has_scan"].astype(float) * W_SCAN
        + (1 - enriched["merchant_bias_score"]) * W_FOR
    )
    # Rush penalty: high concurrent pressure degrades confidence by up to 0.15
    rush_penalty = (enriched["rush_index"] / enriched["rush_index"].max().clip(lower=1e-9)) * 0.15
    enriched["signal_confidence"] = (base_confidence - rush_penalty).clip(
        lower=0.0, upper=1.0
    )

    n_low_conf = (enriched["signal_confidence"] < CONFIDENCE_FLOOR).sum()
    logger.info(
        "Signal enrichment complete. Rush-hour orders: %d. "
        "Low-confidence orders (< %.2f): %d (%.1f%%).",
        enriched["is_rush_hour"].sum(),
        CONFIDENCE_FLOOR,
        n_low_conf,
        n_low_conf / len(enriched) * 100,
    )
    return enriched
