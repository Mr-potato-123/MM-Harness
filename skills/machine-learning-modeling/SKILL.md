---
name: machine-learning-modeling
description: Build predictive models with leakage-safe evaluation, appropriate baselines, calibration, robustness, and reproducible model selection.
---

# Machine-Learning Modeling

Define prediction unit, target availability, decision threshold, deployment distribution, and error costs. Split data by the real generalization boundary—time, entity, site, group, or random unit—before learned preprocessing. Keep a simple baseline and one interpretable model.

Tune inside training folds only. Match metrics to class balance and decision costs; include calibration when probabilities drive action. Compare uncertainty across folds or repeated splits, inspect subgroup and out-of-distribution behavior, and separate model-selection data from final evaluation.

Record features, preprocessing, seed, versions, hyperparameter space, selected configuration, and data digests. Feature importance is not causality. Reject gains that vanish under leakage checks, stronger baselines, or honest splits.

