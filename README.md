# KPT Signal Fusion

> **Hackathon prototype** — multi-signal Kitchen Preparation Time (KPT) estimation for food-delivery dispatch optimisation.

---

## Problem statement

Platform KPT predictions rely on the Food-Order-Ready (FOR) signal — a
merchant-pressed button that is frequently **reactive** (triggered by rider
arrival, not actual food readiness).  This inflates estimated KPT by 1–3
minutes and forces riders to wait at merchant locations, degrading both
experience and throughput.

## Solution overview

`KPTSignalFusion` triangulates three orthogonal signals with dynamic confidence-weighted fusion:

| Signal | Source | Availability | Noise |
|---|---|---|---|
| **POS completion** | Machine-generated at terminal | ~40% of orders | Very low |
| **Pickup scan** | Physical bag scan at dispatch | ~55% of orders | Low–medium |
| **FOR (debiased)** | Human-reported button press | 100% | High (reactive bias) |

The model:
1. **Detects** rider-triggered FOR events via a 90-second gap heuristic.
2. **Calibrates** a per-merchant median bias offset from historical orders.
3. **Fuses** available signals with adaptive normalised weights.
4. **Applies** a confidence gate — low-confidence orders fall back to raw FOR.

## Results (75 k synthetic orders, 300 merchants, 90-day horizon)

| Metric | Baseline | Proposed | Δ |
|---|---|---|---|
| KPT MAE (min) | 3.241 | 1.849 | **−42.9%** |
| RMSE (min) | 5.451 | 3.251 | −40.4% |
| P90 absolute error (min) | 7.821 | 3.710 | −52.6% |
| Within-2-min accuracy | 53.3% | 74.3% | +39.3% |
| Avg rider wait (min) | 6.757 | 6.227 | −7.8% |

---

## Repository layout

```
kpt-signal-fusion/
├── src/
│   ├── __init__.py          # package exports
│   ├── config.py            # all constants and hyper-parameters
│   ├── data_generation.py   # synthetic merchant & order generation
│   ├── bias_detection.py    # FOR bias flagging + merchant offset calibration
│   ├── signal_enrichment.py # rush index, confidence score, rolling bias score
│   ├── fusion_engine.py     # KPTSignalFusion class (5-step algorithm)
│   ├── simulation.py        # dispatch simulation & rider-wait modelling
│   ├── evaluation.py        # metrics, ablation study, segment experiments
│   ├── pipeline.py          # end-to-end orchestrator
│   └── plotting.py          # publication-quality visualisation utilities
├── notebooks/
│   └── kpt_final_submission.ipynb   # clean presentation notebook
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Quick start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the full pipeline from a Python script or REPL
python - <<'EOF'
import sys; sys.path.insert(0, ".")
from src.pipeline import run_full_pipeline, print_results_summary
pipeline = run_full_pipeline()
print_results_summary(pipeline)
EOF
```

Or open `notebooks/kpt_final_submission.ipynb` in JupyterLab / VS Code and run all cells.

---

## Design decisions

### Why median for bias offsets?
Merchant-level FOR offsets have fat-tailed distributions (some outlier orders
with very large gaps).  The median is more robust than the mean and avoids
over-correction from rare extreme events.

### Why a confidence gate (floor)?
FOR-only orders during low-traffic periods have very sparse signal.
Attempting correction on them increases variance without reducing bias.
The 0.15 floor was set empirically to cover ≈5% of orders.

### Why 90 seconds as the bias threshold?
Analysis of the simulated rider-arrival distribution shows that genuine
food-ready events within 90 s of rider arrival have < 3% probability.
Raising the threshold to 120 s increases false-positive rate by 12% with
only a 2% recall gain.

---

## Scalability notes

| Concern | Current approach | Production path |
|---|---|---|
| Bias offsets | Computed batch in-memory | Incremental Flink/Spark aggregation |
| Signal join | Pandas merge | Kafka topic join with 5-min late-data window |
| Confidence score | Rule-based formula | Gradient-boosted classifier trained on labelled FOR audits |
| Dispatch integration | Simulated | gRPC microservice with <5 ms p99 latency target |
