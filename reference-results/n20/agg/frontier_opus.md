# Frontier model — representation comparison (Task 3)

_model=claude-opus-4-8-vlm · modes=['summary', 'raw', 'plot'] · 180 scenarios · 70.9 min · seed=20260621 n=20/family._

| representation | acc | prec | recall | F1 | FPR | FNR | avg latency (s) |
|---|---|---|---|---|---|---|---|
| summary | 0.78 | 0.81 | 0.88 | 0.84 | 0.42 | 0.12 | 7.7 |
| raw | 0.87 | 0.83 | 1.00 | 0.91 | 0.40 | 0.00 | 7.6 |
| plot | 0.81 | 0.82 | 0.92 | 0.87 | 0.40 | 0.08 | 9.3 |
| hybrid:gated | 0.78 | 0.85 | 0.82 | 0.83 | 0.28 | 0.18 | - |
| hybrid:or | 0.78 | 0.85 | 0.82 | 0.83 | 0.28 | 0.18 | - |
| hybrid:and | 0.54 | 1.00 | 0.32 | 0.48 | 0.00 | 0.68 | - |
| statistical | 0.54 | 1.00 | 0.32 | 0.48 | 0.00 | 0.68 | 0.03 |

**Summary vs. plot at frontier scale**: accuracy 0.78 vs 0.81, F1 0.84 vs 0.87 -> **plot > summary**.

## Per-family accuracy

| judge | no_change | noise_equivalent | healed_transient | clean_mean_shift | variance_increase | tail_regression | gradual_drift | cross_metric_marginal | subtle_regression |
|---|---|---|---|---|---|---|---|---|---|
| statistical | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.90 |
| ai:plot | 1.00 | 0.80 | 0.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.50 | 1.00 |
| ai:raw | 0.90 | 0.90 | 0.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| ai:summary | 1.00 | 0.75 | 0.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.30 | 1.00 |
| hybrid:and | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.90 |
| hybrid:gated | 1.00 | 1.00 | 0.15 | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 | 0.90 |
| hybrid:or | 1.00 | 1.00 | 0.15 | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 | 0.90 |

