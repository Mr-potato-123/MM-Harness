---
name: time-series-modeling
description: Model and forecast ordered data while preserving temporal causality, seasonality, uncertainty, and honest backtesting.
---

# Time-Series Modeling

Establish timestamp semantics, frequency, timezone, gaps, revisions, forecast origin, horizon, and target availability. Never use random train/test splits for forecasting. Fit preprocessing, imputation, scaling, and feature extraction using past data only.

Inspect trend, seasonality, structural breaks, autocorrelation, intermittent demand, exogenous-variable availability, and aggregation effects. Compare naive seasonal or persistence baselines with justified statistical or machine-learning alternatives. Use rolling-origin backtests aligned with the operational horizon.

Report horizon-specific errors, calibration or interval coverage, residual autocorrelation, stability across time windows, and performance around regime changes. Future exogenous inputs must be known, forecast separately, or scenario-defined; do not leak realized future values.

