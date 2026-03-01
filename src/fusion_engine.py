"""
fusion_engine.py — Multi-signal KPT fusion model with dynamic weighting.

:class:`KPTSignalFusion` implements the core algorithm:

1. **Debias FOR** — biased (rider-triggered) FORs are corrected by subtracting
   the travel-buffer heuristic; honest FORs are corrected using the calibrated
   per-merchant offset from :func:`~bias_detection.compute_merchant_bias_offsets`.
2. **Adaptive weight normalisation** — absent signals have their weights
   redistributed to present signals, preserving the original relative ratios.
3. **Weighted fusion** — POS, scan, and corrected-FOR are combined into a
   single ``predicted_kpt_min`` estimate.
4. **Rush adjustment** — FOR-only orders (no POS, no scan) during high-rush
   conditions get a small additive adjustment proportional to the rush index.
5. **Confidence gate** — orders with ``signal_confidence < CONFIDENCE_FLOOR``
   fall back to the raw FOR value to avoid over-correcting noisy inputs.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from .bias_detection import compute_merchant_bias_offsets
from .config import (
    CONFIDENCE_FLOOR,
    RANDOM_SEED,
    TRAVEL_BUFFER_MIN,
    W_FOR,
    W_POS,
    W_SCAN,
)

logger = logging.getLogger(__name__)

# Uncertainty half-width constants (minutes) — symmetrical prediction interval
_UNCERTAINTY_BASE: float = 1.5
_UNCERTAINTY_RUSH_SCALE: float = 0.5


class KPTSignalFusion:
    """Multi-signal Kitchen Preparation Time fusion model.

    Parameters
    ----------
    w_pos:
        Base weight for POS completion signal.
    w_scan:
        Base weight for pickup scan signal.
    w_for:
        Base weight for FOR (food-order-ready) signal.
    travel_buffer_min:
        Minutes subtracted from rider-triggered FORs to estimate true
        readiness (rider travel is deducted to recover dispatch offset).
    confidence_floor:
        Orders with ``signal_confidence`` below this fall back to raw FOR.
    seed:
        Reserved for any future stochastic components.

    Attributes
    ----------
    bias_offsets_ : pd.DataFrame or None
        Per-merchant bias offsets; populated by :meth:`compute_bias_offsets`.
    """

    def __init__(
        self,
        w_pos: float = W_POS,
        w_scan: float = W_SCAN,
        w_for: float = W_FOR,
        travel_buffer_min: float = TRAVEL_BUFFER_MIN,
        confidence_floor: float = CONFIDENCE_FLOOR,
        seed: int = RANDOM_SEED,
    ) -> None:
        if abs(w_pos + w_scan + w_for - 1.0) > 1e-6:
            raise ValueError("Fusion weights (w_pos + w_scan + w_for) must sum to 1.0.")
        self.w_pos = w_pos
        self.w_scan = w_scan
        self.w_for = w_for
        self.travel_buffer_min = travel_buffer_min
        self.confidence_floor = confidence_floor
        self.seed = seed
        self.bias_offsets_: Optional[pd.DataFrame] = None

    # ── Fit-style step ────────────────────────────────────────────────────────

    def compute_bias_offsets(self, enriched: pd.DataFrame) -> "KPTSignalFusion":
        """Calibrate per-merchant FOR bias offsets from historical orders.

        Parameters
        ----------
        enriched:
            Enriched order DataFrame (output of
            :func:`~signal_enrichment.enrich_with_signals`).

        Returns
        -------
        KPTSignalFusion
            Returns ``self`` to support method chaining.
        """
        self.bias_offsets_ = compute_merchant_bias_offsets(enriched)
        logger.info(
            "Bias offsets calibrated for %d merchants.",
            len(self.bias_offsets_),
        )
        return self

    # ── Inference step ────────────────────────────────────────────────────────

    def fuse_signals(self, enriched: pd.DataFrame) -> pd.DataFrame:
        """Apply the 5-step fusion algorithm and return predictions.

        Parameters
        ----------
        enriched:
            Enriched order DataFrame.  Must contain the columns produced by
            :func:`~signal_enrichment.enrich_with_signals`.

        Returns
        -------
        pd.DataFrame
            Input augmented with:

            * ``corrected_for_min``       — debiased FOR value.
            * ``w_pos_eff``               — effective normalised POS weight.
            * ``w_scan_eff``              — effective normalised scan weight.
            * ``w_for_eff``               — effective normalised FOR weight.
            * ``predicted_kpt_min``       — fused KPT prediction (minutes).
            * ``predicted_lower_bound``   — lower uncertainty bound.
            * ``predicted_upper_bound``   — upper uncertainty bound.
            * ``signal_confidence_score`` — copy of ``signal_confidence``.

        Raises
        ------
        RuntimeError
            If :meth:`compute_bias_offsets` has not been called first.
        """
        if self.bias_offsets_ is None:
            raise RuntimeError(
                "Call compute_bias_offsets() before fuse_signals()."
            )

        df = enriched.copy()

        # Merge per-merchant offsets
        df = df.merge(
            self.bias_offsets_[["merchant_id", "bias_offset_min"]],
            on="merchant_id",
            how="left",
        )
        global_offset = float(self.bias_offsets_["bias_offset_min"].median())
        df["bias_offset_min"] = df["bias_offset_min"].fillna(global_offset)

        # ── Step 1: Debias FOR ────────────────────────────────────────────────
        biased_correction = (
            df["rider_arrival_delta_min"] - self.travel_buffer_min
        )
        honest_correction = df["reported_FOR_delta_min"] - df["bias_offset_min"]
        df["corrected_for_min"] = np.where(
            df["is_for_biased_flag"], biased_correction, honest_correction
        )

        # ── Step 2: Adaptive weight normalisation ─────────────────────────────
        pos_w = np.where(df["has_pos"], self.w_pos, 0.0)
        scan_w = np.where(df["has_scan"], self.w_scan, 0.0)
        for_w = np.full(len(df), self.w_for)
        total_w = pos_w + scan_w + for_w
        # Avoid division by zero (at least FOR is always present)
        total_w = np.where(total_w == 0, 1.0, total_w)
        df["w_pos_eff"] = pos_w / total_w
        df["w_scan_eff"] = scan_w / total_w
        df["w_for_eff"] = for_w / total_w

        # ── Step 3: Weighted fusion ───────────────────────────────────────────
        pos_val = df["pos_completion_delta_min"].fillna(0.0)
        scan_val = df["pickup_scan_delta_min"].fillna(0.0)
        fused = (
            df["w_pos_eff"] * pos_val
            + df["w_scan_eff"] * scan_val
            + df["w_for_eff"] * df["corrected_for_min"]
        )

        # ── Step 4: Rush adjustment for FOR-only orders ───────────────────────
        for_only_mask = (~df["has_pos"]) & (~df["has_scan"])
        rush_adj = np.where(for_only_mask, df["rush_index"] * 0.08, 0.0)
        fused = fused + rush_adj

        # ── Step 5: Confidence gate ───────────────────────────────────────────
        low_conf_mask = df["signal_confidence"] < self.confidence_floor
        df["predicted_kpt_min"] = np.where(
            low_conf_mask, df["reported_FOR_delta_min"], fused
        )

        # ── Uncertainty bounds ────────────────────────────────────────────────
        half_width = (
            _UNCERTAINTY_BASE
            + df["rush_index"] * _UNCERTAINTY_RUSH_SCALE
        )
        df["predicted_lower_bound"] = (df["predicted_kpt_min"] - half_width).clip(lower=0)
        df["predicted_upper_bound"] = df["predicted_kpt_min"] + half_width
        df["signal_confidence_score"] = df["signal_confidence"]

        logger.info(
            "Signal fusion complete. Mean predicted KPT: %.2f min "
            "(low-confidence fallbacks: %d orders).",
            df["predicted_kpt_min"].mean(),
            int(low_conf_mask.sum()),
        )
        return df
