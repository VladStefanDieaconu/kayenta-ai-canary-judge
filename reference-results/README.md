# Reference results

These are the runs behind the numbers, tables, and figures in the top-level
[README](../README.md). They are committed so you can look at the results before
deciding whether to clone and run the harness yourself.

- `n20/` — the headline run: 180 scenarios (9 families × 20 instances) across the eight
  local models, plus the extended analyses under `n20/agg/` (multi-seed confidence
  intervals, the statistical ensemble, the false-positive guard, the frontier cloud
  model, ROC/PR curves, baselines, pooled McNemar, and test-retest).
- `n5/` — a smaller 45-scenario run (N=5), kept for the sample-size comparison in the
  README.

Each run directory holds the same shape: `summary.csv` / `summary.md` (the scoreboard),
`family_matrix.csv` (judge × family accuracy), `mcnemar.csv`, `kappa.csv`,
`results_raw.csv` (one row per scenario × judge × model), `results.md` (the stitched
report), and `figures/`.

## Running it yourself

`make experiment` writes to a `results/` directory at the repo root, which is
gitignored. Your own runs land there and never touch these committed reference runs. The
dataset is generated deterministically from a fixed master seed, so with the same local
models pulled you should reproduce `n20/` closely; see
[Reproducing the published numbers](../README.md#reproducing-the-published-numbers).
