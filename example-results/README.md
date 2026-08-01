# `example-results/`: example data, not the study's results

One seed of one run, kept in the repository so `make demo` renders a figure and
the analysis path can be exercised from a fresh clone with no stack, no model and
no network.

**This is not the dataset behind the paper.** It is 45 scenarios from a single
seed. Do not read a rate off it and do not cite it. The study's own artefacts (the
180-scenario run, the five-seed sweep, the hosted-model runs and every derived
table) are archived separately. That archive is deposited when the
article is published, and its DOI is added to the main README then; see
[the study section](../README.md#the-study-and-its-data).

## What is here

| file | contents |
|---|---|
| `long_results.csv` | 1,695 rows in the long-format schema: 1,690 judgements and 5 error rows |
| `summary.csv` | the scoreboard, 38 configurations, with confusion counts |
| `family_matrix.csv` | judge × model × family accuracy |
| `errors.csv` | the 5 failed calls, with their `error_kind` |

Provenance: seed `20260621`, five instances of each of nine families, eight local
models, rubric `v1-frozen-2026-06`. Cut from `results/n5-corrected/`, the
five-seed sweep run under the corrected harness, by `build_example_results.py`.

## The error rows are the point

Five rows carry an `error_kind` and no verdict: `moondream-vlm` on the `plot`
representation, on all five `cross_metric_marginal` scenarios, each an
`api_error`. They are here deliberately.

That configuration is the one the study's fabricated-verdict defect landed on.
Under the old harness those five calls were written as verdicts of FAIL at score
0.0; because the family's ground truth is FAIL, all five scored as *correct* and
the cell read a perfect 1.000. Under this harness they are errors, that cell has
**no measurement at 0 of 5**, and `results_schema.load()` keeps them out of every
denominator unless you ask for them.

A `family_matrix.csv` cell with no surviving measurement is left **empty**, not
printed as 0.000.
