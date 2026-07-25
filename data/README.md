# `data/`: runtime output (not a deliverable)

This directory holds **runtime artifacts** produced by the host tools, all
regenerable and git-ignored (see `.gitignore`):

- `eval_manifest.json`: the labelled dataset manifest for the most recent run
  (written by `make seed-eval` / `make experiment`). A tracked copy for the
  published run is kept at `reference-results/n20/eval_manifest.json`.
- `eval-configs/`: the programmatically-generated per-scenario canary configs.
- `ai-logs/`: one JSON per AI-judge call (prompt, chart hash, raw response, parsed
  verdict, resolved model) for reproducibility.
- `last_run.json`: Kayenta execution ids from the most recent `demo-dummy` /
  `scenario` run (used by `make referee`).

The published, tracked results are under `reference-results/n20/` (N=20) and
`reference-results/n5/` (N=5). A local `make experiment` writes to a gitignored
`results/` instead, so it never overwrites them.
