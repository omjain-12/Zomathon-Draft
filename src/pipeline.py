"""
pipeline.py — End-to-end KPT Signal Fusion pipeline orchestrator.

:func:`run_full_pipeline` encapsulates the six-step workflow:

  1. Generate synthetic merchants.
  2. Generate synthetic orders.
  3. Validate the dataset.
  4. Enrich with signal features.
  5. Fit and apply the :class:`~fusion_engine.KPTSignalFusion` model.
  6. Simulate dispatch and evaluate metrics.
"""

from __future__ import annotations

import logging
import warnings
from typing import Any, Dict

import pandas as pd

from .config import N_MERCHANTS, N_ORDERS, RANDOM_SEED
from .data_generation import generate_synthetic_merchants, generate_synthetic_orders
from .evaluation import evaluate_metrics, run_ablation_study
from .fusion_engine import KPTSignalFusion
from .signal_enrichment import enrich_with_signals
from .simulation import simulate_dispatch

logger = logging.getLogger(__name__)


def validate_dataset(orders: pd.DataFrame, merchants: pd.DataFrame) -> None:
    """Perform sanity checks on the generated dataset.

    Raises
    ------
    AssertionError
        If any fundamental invariant is violated.
    """
    assert orders["true_kpt_min"].between(2, 90).all(), (
        "true_kpt_min contains out-of-range values."
    )
    assert orders["merchant_id"].isin(merchants["merchant_id"]).all(), (
        "Orders reference unknown merchant_ids."
    )
    assert not orders["order_id"].duplicated().any(), (
        "Duplicate order_ids detected."
    )
    assert not orders["true_kpt_min"].isna().any(), (
        "true_kpt_min contains NaN values."
    )
    logger.info("Dataset validation passed (%d orders, %d merchants).", len(orders), len(merchants))


def run_full_pipeline(
    n_orders: int = N_ORDERS,
    n_merchants: int = N_MERCHANTS,
    seed: int = RANDOM_SEED,
) -> Dict[str, Any]:
    """Execute the complete KPT Signal Fusion pipeline.

    Parameters
    ----------
    n_orders:
        Number of synthetic orders to generate.
    n_merchants:
        Number of synthetic merchants to generate.
    seed:
        Global random seed for reproducibility.

    Returns
    -------
    dict
        Keys:

        * ``merchants``     — merchant metadata DataFrame.
        * ``orders``        — raw order DataFrame.
        * ``enriched``      — enriched + fused order DataFrame.
        * ``simulated``     — simulated order DataFrame.
        * ``fusion_model``  — fitted :class:`~fusion_engine.KPTSignalFusion` instance.
        * ``base_metrics``  — baseline (raw FOR) evaluation metrics.
        * ``prop_metrics``  — proposed (fused) evaluation metrics.
        * ``ablation``      — ablation study results DataFrame.
    """
    logger.info(
        "Pipeline start — n_orders=%d, n_merchants=%d, seed=%d.",
        n_orders, n_merchants, seed,
    )

    # ── Step 1: Merchant generation ───────────────────────────────────────────
    merchants = generate_synthetic_merchants(n=n_merchants, seed=seed)

    # ── Step 2: Order generation ──────────────────────────────────────────────
    orders = generate_synthetic_orders(
        merchants=merchants, n_orders=n_orders, seed=seed
    )

    # ── Step 3: Validation ────────────────────────────────────────────────────
    validate_dataset(orders, merchants)

    # ── Step 4: Signal enrichment ─────────────────────────────────────────────
    enriched = enrich_with_signals(orders)

    # ── Step 5: Fit model + fuse signals ─────────────────────────────────────
    model = KPTSignalFusion()
    model.compute_bias_offsets(enriched)
    estimated = model.fuse_signals(enriched)

    # ── Step 6: Simulate dispatch + evaluate ──────────────────────────────────
    simulated = simulate_dispatch(estimated, pred_col="predicted_kpt_min")

    base_sim = simulate_dispatch(estimated, pred_col="reported_FOR_delta_min")
    base_metrics = evaluate_metrics(base_sim, "reported_FOR_delta_min")
    prop_metrics = evaluate_metrics(simulated, "predicted_kpt_min")

    ablation = run_ablation_study(estimated, model)

    logger.info(
        "Pipeline complete. MAE %.3f → %.3f min (Δ%.1f%%).",
        base_metrics["MAE"],
        prop_metrics["MAE"],
        (base_metrics["MAE"] - prop_metrics["MAE"]) / base_metrics["MAE"] * 100,
    )

    return {
        "merchants": merchants,
        "orders": orders,
        "enriched": enriched,
        "simulated": simulated,
        "fusion_model": model,
        "base_metrics": base_metrics,
        "prop_metrics": prop_metrics,
        "ablation": ablation,
    }


def print_results_summary(pipeline: Dict[str, Any]) -> None:
    """Print a human-readable summary of pipeline results to stdout.

    Parameters
    ----------
    pipeline:
        Return value of :func:`run_full_pipeline`.
    """
    bm = pipeline["base_metrics"]
    pm = pipeline["prop_metrics"]

    def _delta(b: float, p: float) -> str:
        return f"{(b - p) / b * 100:+.1f}%"

    print("\n" + "=" * 60)
    print("  KPT Signal Fusion — Results Summary")
    print("=" * 60)
    print(f"{'Metric':<30} {'Baseline':>10} {'Proposed':>10} {'Δ':>10}")
    print("-" * 60)

    rows = [
        ("MAE (min)", bm["MAE"], pm["MAE"]),
        ("RMSE (min)", bm["RMSE"], pm["RMSE"]),
        ("P90 error (min)", bm["P90_error"], pm["P90_error"]),
        ("Within 2 min (%)", bm["within_2min_pct"], pm["within_2min_pct"]),
    ]
    if "avg_rider_wait_min" in bm:
        rows.append(("Avg rider wait (min)", bm["avg_rider_wait_min"], pm["avg_rider_wait_min"]))

    for label, b, p in rows:
        print(f"{label:<30} {b:>10.3f} {p:>10.3f} {_delta(b, p):>10}")

    print("=" * 60)
