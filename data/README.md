# `data/`: runtime output (not a deliverable)

This directory holds **runtime artifacts** produced by the host tools, all
regenerable and git-ignored (see `.gitignore`):

- `eval_manifest.json`: the labelled dataset manifest for the most recent run
  (written by `make seed-eval` / `make experiment`). The manifest for the
  published run is in the data archive that accompanies the paper.
- `eval-configs/`: the programmatically-generated per-scenario canary configs.
- `ai-logs/`: one JSON per AI-judge call (prompt, chart hash, raw response, parsed
  verdict, resolved model) for reproducibility.
- `last_run.json`: Kayenta execution ids from the most recent `demo-dummy` /
  `scenario` run (used by `make referee`).

`make experiment` writes to a gitignored `results/`. The study's own result
artefacts are not in this repository; see the README section on where the paper's
data lives. `example-results/` is a small committed slice that `make demo` reads,
and is example data rather than the study's results.
