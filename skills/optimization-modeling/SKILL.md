---
name: optimization-modeling
description: Formulate and verify linear, integer, nonlinear, or multi-objective optimization models with explicit feasibility and optimality evidence.
---

# Optimization Modeling

Define decision variables, domains, objective terms, constraints, indices, units, and parameter provenance. Distinguish hard constraints from penalties and policy preferences. Include a feasible baseline and explain why the selected formulation class fits convexity, integrality, scale, and solver budget.

Before solving, test units, bounds, sign conventions, empty sets, and a hand-checkable small instance. After solving, report solver status, feasibility residuals, objective decomposition, bound or optimality gap, runtime, and sensitivity to material parameters. An incumbent from a timed-out solver is not proven optimal.

For multi-objective problems, expose normalization and trade-offs rather than hiding weights. For infeasibility, produce diagnostics or relaxed constraints instead of invented solutions. Keep the formulation and data separated so alternatives remain comparable.

