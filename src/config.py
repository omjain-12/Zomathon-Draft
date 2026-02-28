"""
config.py — Global constants and hyper-parameters for KPT Signal Fusion.

All magic numbers live here.  Import from this module across the entire
codebase so that a single edit propagates everywhere.
"""

# ── Reproducibility ───────────────────────────────────────────────────────────
RANDOM_SEED: int = 42

# ── Dataset scale ─────────────────────────────────────────────────────────────
N_MERCHANTS: int = 300
N_ORDERS: int = 75_000
SIM_DAYS: int = 90

# ── FOR bias detection ────────────────────────────────────────────────────────
# Orders where |reported_FOR - rider_arrival| < BIAS_THRESHOLD_SEC / 60 minutes
# are flagged as reactively marked (rider-triggered, not kitchen-triggered).
BIAS_THRESHOLD_SEC: int = 90

# ── Merchant bias scoring ─────────────────────────────────────────────────────
# Rolling lookback window (in orders) for computing per-merchant running bias score.
ROLLING_WINDOW: int = 50

# ── Signal fusion weights (must sum to 1.0) ───────────────────────────────────
# POS (machine-generated, tightest readiness proxy)
W_POS: float = 0.50
# Pickup scan (physical bag scan, moderate dispatch lag)
W_SCAN: float = 0.35
# FOR — food-order-ready (human-reported, noisiest signal)
W_FOR: float = 0.15

assert abs(W_POS + W_SCAN + W_FOR - 1.0) < 1e-9, "Fusion weights must sum to 1.0"

# ── Dispatch simulation ───────────────────────────────────────────────────────
# Rider is dispatched when the predicted KPT reaches this fraction.
DISPATCH_FRACTION: float = 0.55
# Deducted from biased FOR (rider-triggered) to recover actual food readiness.
TRAVEL_BUFFER_MIN: float = 5.0

# ── Confidence gate ───────────────────────────────────────────────────────────
# If signal_confidence < CONFIDENCE_FLOOR, skip correction and fall back to raw FOR.
CONFIDENCE_FLOOR: float = 0.15

# ── Plotting palette ──────────────────────────────────────────────────────────
# Zomato brand-adjacent palette: blue, orange, green, yellow, purple.
PALETTE: list[str] = ["#1a73e8", "#f4511e", "#34a853", "#fbbc04", "#9c27b0"]
