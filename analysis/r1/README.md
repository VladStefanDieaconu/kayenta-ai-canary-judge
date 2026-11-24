# analysis/r1: analysis of the first-revision runs

This folder holds the scripts that turn the runs of release line 1.2.0 into the numbers, tables and figures of the
first revision of the article. The runs judged two new datasets, `kind-controls-180` (nine families of metric-kind
controls) and `heldout-benign-100` (five held-out benign families), and repeated a few published windows as
session anchors. What was run, and how every result is analysed, was fixed in `protocol/r1-controls-protocol.json`
before any judge saw a new window; these scripts apply that plan and nothing else. No script here calls a model.

`posthoc/` holds the post-hoc analyses of the published data (cost-sensitive reading, label policies, rationale
check); its own README describes it.

## What the scripts read

- the long-format records of the runs (one CSV per invocation, columns of `tools/results_schema.py`), in a folder
  `runs/records/<invocation>/`, with `runs/STATUS.json`, `runs/markers/` and `runs/ledgers/` beside it;
- the protocol and the series files in `protocol/`; `original-180` and `generalisation-60` are regenerated with
  `tools/eval_dataset.py` at master seed 20260621, 20 windows per family;
- for the session anchors and the pooled figures, the published result data deposit, version 1.1.1
  (doi:10.5281/zenodo.24021904), and Archive S1 of the submitted article (`06_Supplementary_Reanalysis.zip`, unzipped).

## Order of the commands

Run from the repository root. `<runs>` is the runs folder, `<out>` a new folder (no script overwrites a file; a
rerun writes to a new folder). The analysis interpreter is Python 3.12 with numpy 2.3.5, pandas 3.0.1 and scipy
1.18.1, run with `-I -B`.

1. `python -I -B analysis/r1/inventory.py --runs <runs> --out <out>/inventory [--manifest <manifest.tsv>]`
   Per invocation and per configuration: judgements expected, scored, failed calls, outages, attempts; hosted token
   usage; the check that repeated rows of a judgement key agree.
2. `.venv/bin/python -I -B analysis/r1/comparators_live.py --out <out>/comparators`
   (the repository's host environment, with the stack up). The tuned ensemble with its published configuration and
   the guard signals of the guard rule, on `original-180` (reproduction check), `generalisation-60`,
   `kind-controls-180` and `heldout-benign-100`. It calls Kayenta's statistical judge once per window; it calls no
   model.
3. `python -I -B analysis/r1/comparators_bfks.py --archive-s1 <archive-s1> --out <out>/comparators`
   The BF+KS battery of Archive S1 on the same four datasets, after a reproduction check on `original-180`.
4. `python -I -B analysis/r1/analyze_r1.py --runs <runs> --comparators <out>/comparators --bfks <out>/comparators --deposit <data-deposit> --archive-s1 <archive-s1> --out <out>`
   Every analysis of the protocol's plan: statistical verdicts across invocations; the kind controls per family and
   configuration, false alarms by kind, detection, balanced accuracy (BA) and MCC with bootstrap intervals; the
   metric-kind rules on the controls and on the pooled 360 windows; the paired tests; the reading rules; the
   held-out sets, rubric effects and label policies P6 and P7; the session anchors; and the tables for the article
   (`tables/`).
5. `python -I -B analysis/r1/figures/figure5_controls.py --runs <runs> --deposit <data-deposit> --out <out>/figures`
   and `python -I -B analysis/r1/figures/figure_heldout_scores.py --runs <runs> --deposit <data-deposit> --out <out>/figures`,
   with `COMPOSE_PROJECT_NAME` and `COMPOSE_FILE` of the running stack in the environment (see below).
6. `python -I -B analysis/r1/posthoc/d3_rationale_check.py --records <runs>/records/*/*.csv --generate kind-controls-180 heldout-benign-100 generalisation-60 original-180 --out <out>/d3-r1runs --spotcheck-name rationale-spotcheck-r1runs.csv`
   The rationale check of `posthoc/`, unchanged, on the rationales of the new runs.

## How the statistics are computed

As in the submitted article, with the code that produced its numbers:

- Wilson 95% intervals: `tools/ci_utils.py` `wilson_interval`.
- Bootstrap intervals of BA and MCC: `boot_config` of `analysis/area-A-headline/analyze_A.py` of the data deposit,
  loaded from that file function by function (the loader of `posthoc/posthoc_common.py`), so the archived script's
  start-up code never runs. Windows are resampled with replacement within each family (family sizes fixed), 2,000
  replicates, percentile interval. The protocol fixes the stream: `random.Random(20261122)` for every
  configuration.
- Paired differences of proportions: Newcombe's method 10 (`newcombe10` of the same file); exact two-sided McNemar
  tests (`p_mcnemar_exact`) and the exact Holm adjustment (`holm_exact`) over the sets the protocol names: 42 tests
  on the controls, 18 rubric-effect tests on the held-out sets.
- Numbers are printed as the archived code prints them, `f"{x:.3f}"` of the floating-point value. This reproduces
  the published values that sit exactly on a half (Phi-4 on summaries, BA 0.7625, printed 0.762; gpt-oss-120b on
  raw series, BA 0.7125, printed 0.713). Archive S1 printed BF+KS on `original-180` (BA 63/80 = 0.7875) as 0.788;
  this code prints 0.787. Tables for the article print the minus sign as U+2212.
- FAIL is the positive class; Marginal counts as FAIL. One record per judgement key is the successful call; failed
  calls are counted per configuration and excluded from every denominator.
- Reading rules: exactly those of the protocol (`reading_rules`): "kind-driven", "not explained by kind on these
  controls", otherwise no label; a label needs 20 scored windows in every family the rule reads.

## The BF+KS battery on new data

Archive S1's `posthoc_reanalysis.py` computes the battery inside a loop over the 180 deposited windows and asserts
the published families, so it cannot read another dataset unchanged. `comparators_bfks.py` applies the two
statements of that loop (one-sided Brown–Forsythe test on absolute deviations from the median; exact one-sided
two-sample Kolmogorov–Smirnov test) and its decision rule (FAIL if any of the 2m p-values is below 0.05/(2m)) to
each window, and first checks itself against Archive S1 on `original-180`: all 180 verdicts and every p-value.

## Figures

Figures 2 and 4 of the article were drawn by the data deposit's analysis scripts (`write_a6` of `analyze_A.py`
and `figure_scores` of `analyze_C.py`) as SVG files at 6.9 inches, rendered to PNG at 600 dpi; no tool in `tools/`
draws them. The two figure scripts here keep their geometry, colours, markers, line widths and text sizes, and
reuse their tick and binning functions, but draw with matplotlib inside the judge-service container of the running
stack (`figures/render_in_service.py`, which follows `tools/container_plot.py`: only a finished drawing list crosses
into the container, through `docker compose exec -T judge-service`). Unlike `container_plot.render`, the renderer
never falls back to `docker run`, checks the name of the container it reaches, and returns the PNG on standard
output, so it creates no container and writes nothing into the shared `data/ai-logs` volume. The text font is the
DejaVu Sans of the image; the published PNGs were rendered with a Helvetica-type face. Figure 5 has fourteen panels
in five rows, so its plots are 53 points high instead of 84, to keep the figure at most 1.20 times as high as wide.

## Outputs (in `<out>`)

- `inventory/`: per invocation and configuration, failed calls, token usage, summary.
- `comparators/`: per-window verdicts of the model-free comparators and their reproduction checks.
- `s0_*` to `s7_*`: the analysis tables, machine-readable; `results.json`; `facts_analysis.json`.
- `tables/`: Tables 9 and 10 in the layout of the article's build templates, and the supplementary tables.
- `figures/`: `Figure_5.png`, `Figure_S18.png`, their data and facts.
- `d3-r1runs/`: the rationale check on the new runs.
