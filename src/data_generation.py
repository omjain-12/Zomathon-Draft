"""
data_generation.py — Synthetic merchant and order generation.

Produces realistic but entirely synthetic data that mimics the statistical
properties of a food-delivery platform without exposing any proprietary
records.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from .config import (
    N_MERCHANTS,
    N_ORDERS,
    RANDOM_SEED,
    SIM_DAYS,
)

logger = logging.getLogger(__name__)

# ── Merchant-size distributions ───────────────────────────────────────────────
_SIZE_LABELS: list[str] = ["small", "medium", "large"]
_SIZE_PROBS: list[float] = [0.50, 0.35, 0.15]

# Base prep speed (mean, std) in minutes
_SPEED_PARAMS: dict[str, tuple[float, float]] = {
    "small": (12.0, 4.0),
    "medium": (18.0, 5.0),
    "large": (22.0, 6.0),
}

# FOR bias probability (Beta distribution shape: alpha=p*5, beta=(1-p)*5)
_BIAS_PROBS: dict[str, float] = {
    "small": 0.45,
    "medium": 0.28,
    "large": 0.15,
}

# POS terminal availability rate
_POS_RATES: dict[str, float] = {
    "small": 0.12,
    "medium": 0.40,
    "large": 0.75,
}

# Pickup scan availability rate
_SCAN_RATES: dict[str, float] = {
    "small": 0.30,
    "medium": 0.55,
    "large": 0.80,
}

# Rush-hour definitions (hour-of-day, 0-based)
_RUSH_HOURS: set[int] = set(range(12, 15)) | set(range(19, 23))  # 12–14h, 19–22h


def generate_synthetic_merchants(
    n: int = N_MERCHANTS,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Generate a table of synthetic merchants with infrastructure metadata.

    Parameters
    ----------
    n:
        Number of merchants to generate.
    seed:
        NumPy random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        One row per merchant with columns:
        ``merchant_id``, ``merchant_size``, ``base_prep_speed_min``,
        ``bias_probability``, ``pos_available``, ``scan_available``.
    """
    rng = np.random.default_rng(seed)

    sizes: list[str] = list(
        rng.choice(_SIZE_LABELS, size=n, p=_SIZE_PROBS)
    )

    base_speeds: list[float] = [
        float(
            np.clip(
                rng.normal(*_SPEED_PARAMS[s]),
                _SPEED_PARAMS[s][0] - 2 * _SPEED_PARAMS[s][1],
                _SPEED_PARAMS[s][0] + 2 * _SPEED_PARAMS[s][1],
            )
        )
        for s in sizes
    ]

    bias_probs: list[float] = [
        float(
            np.clip(
                rng.beta(
                    _BIAS_PROBS[s] * 5,
                    (1 - _BIAS_PROBS[s]) * 5,
                ),
                0.0,
                1.0,
            )
        )
        for s in sizes
    ]

    pos_available: list[bool] = [
        bool(rng.random() < _POS_RATES[s]) for s in sizes
    ]
    scan_available: list[bool] = [
        bool(rng.random() < _SCAN_RATES[s]) for s in sizes
    ]

    merchants = pd.DataFrame(
        {
            "merchant_id": range(n),
            "merchant_size": sizes,
            "base_prep_speed_min": base_speeds,
            "bias_probability": bias_probs,
            "pos_available": pos_available,
            "scan_available": scan_available,
        }
    )

    logger.info("Generated %d synthetic merchants.", n)
    return merchants


def generate_synthetic_orders(
    merchants: pd.DataFrame,
    n_orders: int = N_ORDERS,
    sim_days: int = SIM_DAYS,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Generate a table of synthetic orders with all raw signal columns.

    The simulation injects realistic FOR-bias behaviour: a fraction of
    merchants (governed by their ``bias_probability``) reactively mark
    food-order-ready *after* the rider arrives, not when the food is
    actually ready.

    Parameters
    ----------
    merchants:
        Output of :func:`generate_synthetic_merchants`.
    n_orders:
        Total orders to generate.
    sim_days:
        Simulated time horizon in days (used to set ``order_time_min``).
    seed:
        NumPy random seed.

    Returns
    -------
    pd.DataFrame
        One row per order with columns:
        ``order_id``, ``merchant_id``, ``dish_complexity``,
        ``order_time_min``, ``hour_of_day``, ``is_rush_hour``,
        ``platform_orders_concurrent``, ``external_orders_count``,
        ``rush_multiplier``, ``true_kpt_min``,
        ``reported_FOR_delta_min``, ``rider_assigned_delta_min``,
        ``rider_travel_time_min``, ``rider_arrival_delta_min``,
        ``pickup_scan_delta_min``, ``pos_completion_delta_min``,
        ``is_merchant_biased``.
    """
    rng = np.random.default_rng(seed)

    # ── Sample merchant attributes ────────────────────────────────────────────
    merchant_ids: np.ndarray = rng.integers(0, len(merchants), n_orders)
    m = merchants.set_index("merchant_id")

    sizes = m.loc[merchant_ids, "merchant_size"].values
    base_speeds = m.loc[merchant_ids, "base_prep_speed_min"].values.astype(float)
    bias_probs = m.loc[merchant_ids, "bias_probability"].values.astype(float)
    pos_avail = m.loc[merchant_ids, "pos_available"].values.astype(bool)
    scan_avail = m.loc[merchant_ids, "scan_available"].values.astype(bool)

    # ── Temporal features ─────────────────────────────────────────────────────
    order_times_min: np.ndarray = rng.uniform(0, sim_days * 24 * 60, n_orders)
    hours: np.ndarray = (order_times_min // 60 % 24).astype(int)
    is_rush: np.ndarray = np.isin(hours, list(_RUSH_HOURS))

    # ── Concurrency / external load ───────────────────────────────────────────
    platform_concurrent: np.ndarray = np.where(
        is_rush,
        rng.integers(8, 18, n_orders),
        rng.integers(2, 8, n_orders),
    )
    external_orders: np.ndarray = np.where(
        is_rush,
        rng.poisson(12, n_orders),
        rng.poisson(4, n_orders),
    )

    # ── True KPT ─────────────────────────────────────────────────────────────
    dish_complexity: np.ndarray = rng.uniform(0.7, 1.5, n_orders)
    rush_mult: np.ndarray = np.where(is_rush, rng.uniform(1.1, 1.4, n_orders), 1.0)
    true_kpt: np.ndarray = np.clip(
        base_speeds * dish_complexity * rush_mult + rng.normal(0, 1.5, n_orders),
        2.0,
        90.0,
    )

    # ── Rider signal ─────────────────────────────────────────────────────────
    rider_assigned: np.ndarray = rng.uniform(0.5, 3.0, n_orders)
    rider_travel: np.ndarray = np.clip(
        rng.normal(7.0, 2.5, n_orders), 2.0, 20.0
    )
    rider_arrival: np.ndarray = rider_assigned + rider_travel

    # ── POS and scan signals ──────────────────────────────────────────────────
    # POS: small noise around true_kpt; NaN when unavailable
    pos_delta: np.ndarray = np.where(
        pos_avail,
        true_kpt + np.clip(rng.normal(0.3, 0.8, n_orders), -1.0, 5.0),
        np.nan,
    )

    # Scan: lognormal lag above true_kpt; NaN when unavailable
    scan_lag: np.ndarray = rng.lognormal(np.log(1.5), 0.4, n_orders)
    scan_delta: np.ndarray = np.where(scan_avail, true_kpt + scan_lag, np.nan)

    # ── Biased vs honest FOR ─────────────────────────────────────────────────
    is_biased: np.ndarray = rng.random(n_orders) < bias_probs
    honest_for: np.ndarray = true_kpt + rng.normal(0.5, 2.0, n_orders)
    biased_for: np.ndarray = rider_arrival + rng.normal(0.5, 0.8, n_orders)
    for_delta: np.ndarray = np.where(is_biased, biased_for, honest_for)

    orders = pd.DataFrame(
        {
            "order_id": range(n_orders),
            "merchant_id": merchant_ids,
            "dish_complexity": dish_complexity,
            "order_time_min": order_times_min,
            "hour_of_day": hours,
            "is_rush_hour": is_rush,
            "platform_orders_concurrent": platform_concurrent,
            "external_orders_count": external_orders,
            "rush_multiplier": rush_mult,
            "true_kpt_min": true_kpt,
            "reported_FOR_delta_min": for_delta,
            "rider_assigned_delta_min": rider_assigned,
            "rider_travel_time_min": rider_travel,
            "rider_arrival_delta_min": rider_arrival,
            "pickup_scan_delta_min": scan_delta,
            "pos_completion_delta_min": pos_delta,
            "is_merchant_biased": is_biased,
        }
    )

    logger.info(
        "Generated %d synthetic orders. Biased FOR rate: %.1f%%.",
        n_orders,
        is_biased.mean() * 100,
    )
    return orders
