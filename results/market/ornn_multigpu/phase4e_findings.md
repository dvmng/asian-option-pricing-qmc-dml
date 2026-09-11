# Phase 4E — Ornn multi-GPU descriptive findings

## Scope

This phase is descriptive robustness only. It does not replace the frozen H100 calibration, Q bands, ML datasets or trained neural networks.

- GPU benchmarks analysed: 5.
- Highest mean rental-price benchmark: **B200** (5.7409 USD/GPU-hour).
- Highest annualized daily-log-return volatility: **H200** (0.9694).

## Exploratory mean-reversion diagnostic

Descriptive support is noted only when `0 < phi < 1` and the fixed-parameter AR(1) improves OOS RMSE relative to a log-price random walk.

Benchmarks meeting both descriptive conditions: **A100 SXM4**, **B200**, **H100 SXM**, **H200**.

This is not a formal model-selection test and must not be described as proof of mean reversion.

## H100

- AR(1) phi: 0.893345.
- Exploratory half-life: 6.146 days.
- OOS AR(1) vs random-walk RMSE improvement: 3.088%.
- Annualized log-return volatility: 0.6332.

These H100 numbers use the NEW rolling multi-GPU window and must not replace the earlier frozen H100 calibration used for ML data generation.

## Interpretation

The five GPU benchmarks are compared separately. Base-100 normalization and return correlations are used for co-movement analysis, not to assert economic interchangeability or construct a synthetic GPU index.
