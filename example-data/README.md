# `example-data/`: a valid input file, not an example of your data

One file, `own-data-example.csv`, in the format specified in
[docs/own-data.md](../docs/own-data.md). It is here so that the bring-your-own-data
path can be run from a fresh clone before you have exported anything, and so that
there is a file to copy the shape from that is known to be correct.

```bash
make own-data-check OWN_DATA=example-data/own-data-example.csv   # no stack, no model
make own-data       OWN_DATA=example-data/own-data-example.csv   # the scored run
```

## What is in it

Nine windows: one instance of each of the nine families of this repository's own
seeded evaluation dataset, at master seed `20260621`, written out through
`tools/own_data.py --export`. 682 rows, 11 metric series, every window labelled.

**The numbers are the study's own, not plausible-looking numbers typed by hand.**
That is the point of generating it rather than writing it: a hand-made example
drifts away from the loader the moment either changes, and an example whose
values were invented teaches the wrong thing about what real telemetry looks
like. Regenerate it at any time with:

```bash
make own-data-template OUT=example-data/own-data-example.csv
```

which produces the same bytes, because the exporter uses a fixed window start.

## What it is not

**It is not an example of your data, and a result from it is not a result about
the judges.** These are synthetic windows built to embody four specific
statistical blind spots; four of the nine families exist precisely because the
rank test cannot see them. Scoring the judges on them reproduces a slice of the
study, not a measurement of anything on your service.

It is also not the round-trip test. That is
`tools/test_own_data_roundtrip.py`, which exports, re-imports and compares the
resulting `metricSetPairList` and canary config field by field.

## What the export does not carry back

The file records everything a judge is handed, and re-importing it reproduces the
`metricSetPairList` and the canary config exactly. Four things about a *generated*
window do not survive, because a window someone measured does not have them:
its `gid` in the generated global index, its generator `seed`, the generator
arguments in `params` (replaced by the loader's provenance record) and
`MetricSpec.base`, which for a re-imported export is the exported name. None of
the four reaches a judge; `tools/test_own_data_contract.py` asserts that list,
so it fails if a fifth thing joins it.

One thing the file deliberately does not carry is the *order* of the metrics
within a window. The exporter writes them sorted by name and the loader reads
them back sorted by name, so a file re-sorted on its way through a spreadsheet
or a second `ORDER BY` is the same run: the same prompt, the same metric names
and the same group weights. For `cross_metric_marginal`, the one window here
with three metrics, that order is latency, errors, memory, where the generator
declares latency, memory, errors.
