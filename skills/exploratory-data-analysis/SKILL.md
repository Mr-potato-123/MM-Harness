---
name: exploratory-data-analysis
description: Explore registered data to expose structure, quality problems, plausible relationships, and modeling risks without converting exploration into conclusions.
---

# Exploratory Data Analysis

First distinguish a sampled dataset from a short list of deterministic physical
constants. For a real dataset, start from schema and provenance and examine
distributions, missingness patterns, duplicates, impossible values, units,
censoring, temporal ordering, group imbalance, correlations, and target leakage.
For fixed physical inputs, replace generic histograms/outlier cleaning with
geometry, units, magnitude, conservation, and feasibility checks. Use summaries
or sampled views for large data while computing final statistics over the
declared dataset.

Choose plots for questions: distributions for shape, scatter/residual views for relationships, grouped intervals for comparisons, and time plots for temporal structure. Always retain denominators and uncertainty. Investigate anomalies before deleting them; record every exclusion or repair as a reproducible transformation.

EDA generates hypotheses and model requirements, not confirmatory claims. Return findings, alternative explanations, data-quality defects, transformations proposed, and tests needed before those findings can support the final model.
