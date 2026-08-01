# Provenance: the fabricated-verdict defect

An earlier version of this harness recorded failed model calls as verdicts. The
published study reports it as a finding, and the correction is why some result
artefacts exist in both a pre- and a post-correction form. This document is the
record: what the defect was, how it was found, what it touched, what changed in
the code, and which artefact is which.

It ships with the repository rather than living only in the audit trail because
the behaviour it describes is the behaviour you get from this code. If you are
here because a result of your own looks too good on a family whose ground truth
is FAIL, §5 is the part to read.

---

## 1. What the defect was

`judge_clients._err()` returns a well-formed result object so that a caller
always has something structurally valid to inspect. That object carries
`verdict = "FAIL"`, `score = 0.0` and an empty `error` field — which downstream
is indistinguishable from a model that looked at the data and judged it a
failure.

The runner scored it. Three consequences followed, each of which hid the next:

- the retry never fired, because nothing reported a failure;
- the run summary reported zero errors, because there were no error records;
- the rows entered the accuracy denominator, and because both affected families
  have ground truth FAIL, **the fabricated rows scored as *correct*** and
  flattered exactly the configurations they damaged.

The only surviving trace in the published files is the rationale string, which
begins `AI judge error (`.

## 2. How it was found

Not by a failing test. The scored artefacts were being re-derived for the
manuscript, and `moondream-vlm / ai:plot` showed accuracy **1.000** on
`cross_metric_marginal` — a perfect score, from the weakest vision model, on the
hardest family, while the same model scored 0.83 or below everywhere else. A
cell that good on that family from that model was not credible.

Reading the rows behind it showed all twenty carrying score 0.0 with a rationale
beginning `AI judge error (`. The family's ground truth is FAIL, the fabricated
verdict was FAIL, so all twenty scored correct and produced a perfect cell.

Two things follow that are worth stating plainly, because both are easy to get
wrong:

**The reproduction check did not catch it, and could not have.** The benchmark
reproduced 0 of 6,840 rows differing five weeks after its original run. That
result stands. But both failure modes here are deterministic, so the fabricated
rows reproduced exactly along with everything else. **A reproduction check
confirms determinism, not validity.** They are independent properties.

**An anomalous *good* result deserves the same scrutiny as an anomalous bad
one.** The defect survived review because its symptom was a number nobody wanted
to question.

## 3. What it touched

| artefact | rows | fabricated | scored as correct |
|---|---|---|---|
| `reference-results/n20/results_raw.csv` | 6,840 | **22** | 21 |
| `reference-results/n5/results_raw.csv` | 1,710 | **5** | 5 |
| `reference-results/n20/agg/long_results.csv` | 8,550 | see §6 | see §6 |

In the 180-scenario run: 21 from `moondream-vlm / ai:plot` (20 in
`cross_metric_marginal`, 1 in `gradual_drift`) and 1 from
`deepseek-r1-llm / ai:summary` (in `no_change`). Twenty-one of the twenty-two
scored as correct.

Hybrid rows derived from a failed primary call are **contaminated** rather than
fabricated: the AI half of the merge was a fabrication, so the hybrid verdict was
computed from it. They are counted separately throughout.

### Two root causes the old harness collapsed into one

**A gateway failure.** `moondream-vlm` on the `plot` representation returns
`Error code: 500 — litellm.APIConnectionError` on every attempt. The gateway
cannot parse what Ollama returns for that model on that representation. It is
fully deterministic, and `cross_metric_marginal` — the three-metric family, and
therefore the largest rendered chart — failed on 20 of 20.

**Budget exhaustion.** `deepseek-r1-llm` on one scenario returned
`completion_tokens = 1024` against `max_tokens = 1024` with empty content and
`finish_reason = 'stop'`. The reasoning model spent its entire budget reasoning
and emitted no answer.

The second carries a practical lesson: **`finish_reason` is not a reliable
truncation signal.** It read `stop`, not `length`, while the completion was empty
and the budget exactly exhausted. A guard keyed on `finish_reason` would have
missed it. The check that catches it is the trivial one — did the model return
any content at all.

## 4. What changed in the harness

| change | where |
|---|---|
| A failed call produces an error record and no result row. It cannot reach an accuracy denominator. | `tools/run_experiment.py::run_scenario` |
| No hybrid row is derived from a failed primary call. | same |
| The judge reports failure explicitly through `judgeMetadata.ok` / `error_kind`, so a structurally valid 200 is no longer taken as evidence that a model answered. | `judge-service/app/llm_judge.py`, `tools/judge_clients.py::_normalise` |
| Error rows carry `error_kind`, and `verdict`, `score` and `correct` are empty. | `tools/results_schema.py` |
| `results_schema.load()` excludes error rows unless asked for them. | same |
| An `errors.csv` is written on every run, empty or not, so "no error file" cannot be read as "no errors". | `tools/run_experiment.py` |
| Rows carry `prompt_id`, `prompt_hash`, `finish_reason`, `tokens_in` and `tokens_out`, so a truncated or mis-rubriced completion is visible after the fact. | `tools/results_schema.py`, `tools/run_multiseed_corrected.py` |

**The runner that produced the fabricated verdicts has been deleted, not kept.**
`tools/run_experiment_multiseed.py` recorded failed calls as verdicts and wrote a
frame with no rationale column, and it was still wired to `make analysis-live`. A
warning in its docstring was not enough: the defect went unnoticed the first time
precisely because someone ran a Makefile target without reading the source. It is
replaced by `tools/run_multiseed_corrected.py`, which the target now calls.

Nothing is lost by that. The artefact it produced is preserved in the data
deposit as `reference-results/n20/agg/long_results.csv`, and this document
records what it did. The code is gone; the evidence is not.

The claim is checked rather than asserted.
`tools/test_failed_call_is_not_scored.py` stubs the AI judge to fail in each of
the five ways the harness can observe, runs one scenario of each, and inspects
the rows: it fails if any verdict row, any hybrid row, or any error row carrying
a verdict is produced. It also runs a control, so the test cannot pass by the
harness producing nothing at all.

```
$ python tools/test_failed_call_is_not_scored.py
  control (no error) ok
  api_error          ok
  empty_completion   ok
  parse_failure      ok
  transport          ok
  unknown_prompt     ok
  results_schema     ok
```

## 5. If you are reading your own results

Three checks, cheapest first:

1. **Read `errors.csv` and the `error_kind` column.** A configuration with error
   rows has a reduced denominator on the affected families. That is correct
   behaviour, not a problem to fix — but a rate over four scenarios is not the
   same measurement as a rate over twenty, and only the `n` tells you which you
   have.
2. **Distrust a perfect cell.** Especially from a weak model, on a hard family,
   where the family's ground truth matches the failure verdict.
3. **Do not read a reproduction check as a validity check.** If a model fails the
   same way on the same scenarios in two runs, both runs record the same failure,
   and a row-by-row diff will report zero differences.

## 6. Which artefact is which

Both the pre- and the post-correction artefacts are preserved. The paper reports
a defect in the original, and that report is only checkable if the original
survives alongside the correction. All of them are in the data archive
distributed with the paper, not in this repository. That archive is deposited when
the article is published and its DOI is added to the README then; every reference
to "the data archive" or "the data deposit" below means that deposit, which is not
available yet.

| artefact | state | how it was produced |
|---|---|---|
| `reference-results/n20/` | **pre-correction**, unaltered | the 180-scenario run exactly as first published, including its 22 fabricated rows |
| `reference-results/n5/` | **pre-correction**, unaltered | the 45-scenario pilot, including its 5 |
| `results/corrected/` | **post-correction, re-measured** | both affected configurations re-run in full under the fixed harness; failed calls dropped, not carried across |
| `results/corrected-n5/` | **post-correction, drop-only** | the pilot with its 5 failed calls removed and *nothing put in their place*. The affected cell has no measurement and must be read as *no data*, not as a rate |
| `results/n5-corrected/` | **post-correction, re-measured** | the five-seed sweep repeated under the fixed harness, written through the long-format schema |

The distinction between *re-measured* and *drop-only* is deliberate and is why
`results/corrected-n5/` is named separately rather than merged. Re-running the
pilot would have spent inference on an artefact no result depends on.

`reference-results/` was never rewritten. Correcting it in place would have
destroyed the evidence for the finding the paper reports.

### One difference between the committed reference and the archived copy

`reference-results/n20/agg/roc_pr_auc.csv` exists in two forms. The version in
this repository's git history has six columns; the archived copy has seven, with
`average_precision` added. No existing value differs.

The reason: the trapezoidal precision-recall area interpolates between attained
operating points and so credits a judge with points no threshold can reach. It
diverges most for a judge emitting few distinct scores — which is the incumbent
statistical judge, at two — so reporting the trapezoid alone biases the
comparison in its favour. Average precision is the stepwise integral over the
same sweep and is the primary summary. It was computed and added to the archived
copy after the file was committed.

### The multi-seed artefact

`reference-results/n20/agg/long_results.csv` **cannot be audited for this fault**
in the form it was published. It records no rationale, and its `error` column is
empty on all 8,550 rows — which is exactly what the defect produces. The marker
simply was not stored.

A weaker proxy exists: every fabricated row scores exactly 0.0. Calibrated
against the 180-scenario run, where the truth is known, the proxy is necessary
but not sufficient — 21 of 28 zero-scoring `moondream-vlm / ai:plot` rows were
fabricated, and 1 of 3 for `deepseek-r1-llm / ai:summary`. Applied to the
multi-seed file it gives an **upper bound of 37** for moondream, not a count.

That bound has since been replaced by a measurement. The five-seed sweep was
repeated under the fixed harness, which records failed calls explicitly; the
count and what it changes are reported with the paper, and the corrected
artefact is `results/n5-corrected/` in the data deposit.
