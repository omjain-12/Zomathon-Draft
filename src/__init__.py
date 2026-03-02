"""
KPT Signal Fusion — modular ML engineering prototype.

Modules
-------
config            : global constants and hyper-parameters
data_generation   : synthetic merchant & order generation
bias_detection    : FOR bias detection + merchant-level offset calibration
signal_enrichment : feature enrichment (rush index, confidence, merchant bias score)
fusion_engine     : KPTSignalFusion multi-signal weighting model
simulation        : dispatch simulation and rider-wait modelling
evaluation        : metric computation, ablation study, segment experiments
pipeline          : end-to-end orchestration
"""

from .config import (
    RANDOM_SEED,
    N_MERCHANTS,
    N_ORDERS,
    SIM_DAYS,
    BIAS_THRESHOLD_SEC,
    ROLLING_WINDOW,
    W_POS,
    W_SCAN,
    W_FOR,
    DISPATCH_FRACTION,
    TRAVEL_BUFFER_MIN,
    CONFIDENCE_FLOOR,
    PALETTE,
)
from .pipeline import run_full_pipeline

__all__ = [
    "RANDOM_SEED",
    "N_MERCHANTS",
    "N_ORDERS",
    "SIM_DAYS",
    "BIAS_THRESHOLD_SEC",
    "ROLLING_WINDOW",
    "W_POS",
    "W_SCAN",
    "W_FOR",
    "DISPATCH_FRACTION",
    "TRAVEL_BUFFER_MIN",
    "CONFIDENCE_FLOOR",
    "PALETTE",
    "run_full_pipeline",
]
