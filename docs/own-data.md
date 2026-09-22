# Bringing your own canary windows

The evaluation dataset in this repository is generated. That is deliberate: a judge can
only be scored against telemetry whose correct verdict is already known, and synthetic
data is the only kind that comes with one for free.

Your own telemetry has correct verdicts too. You know which deploys were rolled back and
which were promoted and stayed healthy. This page describes how to hand those windows to
the harness so that the **same** judges, the **same** statistical configuration and the
**same** scoring code run over your data as over the study's dataset — with no change to
any file in this repository.

```bash
make own-data-template OUT=windows.csv   # a valid file in the format, to copy the shape from
make own-data-check OWN_DATA=windows.csv # validate, and print exactly what the harness sees
make own-data OWN_DATA=windows.csv       # the scored run -> results/own-data/
```

A file in this format is committed at `example-data/own-data-example.csv`, so the
path can be run from a fresh clone before you have exported anything. It is nine
windows of this repository's own seeded dataset, written out through the exporter
below rather than typed by hand.

What this does not do is anywhere near as important as what it does, and is listed in
[What this does not give you](#what-this-does-not-give-you). Read that section before you
draw a conclusion from a number this produces.

---

## Contents

- [What a window is](#what-a-window-is)
- [The input format](#the-input-format)
- [Required and optional fields](#required-and-optional-fields)
  - [Metric order is the file's to declare](#metric-order-is-the-files-to-declare)
- [Running without labels](#running-without-labels)
- [Window shape: length, gaps and step](#window-shape-length-gaps-and-step)
- [Metric names, scenario names and leakage](#metric-names-scenario-names-and-leakage)
- [The statistical comparator](#the-statistical-comparator)
- [What changes in the code](#what-changes-in-the-code)
- [Make targets](#make-targets)
- [What this does not give you](#what-this-does-not-give-you)

---

## What a window is

One **window** (the harness calls it a *scenario*) is one promotion decision: a baseline
series and a canary series, for one or more metrics, over one time range, plus the verdict
that decision should have had.

That is all the harness needs. For each window it:

1. builds a Kayenta `metricSetPairList` from your two series;
2. feeds it to the genuine `NetflixACAJudge` through Kayenta's `/judges/judge`;
3. feeds the identical pairs to the AI judge, in each representation, for each model
   present locally;
4. derives the three hybrid policies from those two results;
5. writes one row per (window, judge, model), and — if the window carries a label —
   scores it.

Everything below that point in the stack is already indifferent to where the series came
from. Nothing between `build_pairs()` and the model knows whether a number was generated
or exported from your Prometheus.

---

## The input format

**Long-format CSV: one row per (window, metric, side, sample).** UTF-8, a header row,
comma-separated, RFC 4180 quoting (the Python `csv` default). Rows may appear in any
order; the loader groups them, sorts each series by timestamp, orders the windows by
your scenario identifier and the metrics within a window by
[what the file declares in `metric_order`](#metric-order-is-the-files-to-declare) —
falling back to your metric name when it declares nothing. Nothing
a judge sees depends on where a row sits in the file: two exports of the same windows
with the rows shuffled produce the same opaque ids, the same renamed metric names, the
same order in the prompt and the same group weights. Blank lines and
lines beginning with `#` *before* the header are ignored, which is what lets an
exported file explain itself in its first lines; after the header every line is
data, so a value may begin with `#`. A UTF-8 byte-order mark is stripped, because
a spreadsheet export carries one and it would otherwise turn `scenario` into a
column nobody recognises.

```csv
scenario,metric,side,timestamp,value,kind,direction,label
checkout-2026-03-14T09:00Z,checkout_p99_seconds,control,2026-03-14T09:00:00Z,0.201,latency,increase,FAIL
checkout-2026-03-14T09:00Z,checkout_p99_seconds,canary,2026-03-14T09:00:00Z,0.204,latency,increase,FAIL
checkout-2026-03-14T09:00Z,checkout_p99_seconds,control,2026-03-14T09:01:00Z,0.199,latency,increase,FAIL
checkout-2026-03-14T09:00Z,checkout_p99_seconds,canary,2026-03-14T09:01:00Z,0.207,latency,increase,FAIL
```

### Why one row per side rather than one row per sample carrying both sides

The alternative — `scenario,metric,timestamp,control_value,canary_value` — is shorter and
reads nicely, and it is wrong for this data. It asserts that the two sides have the same
number of samples and that sample *i* of one lines up with sample *i* of the other. Real
exports do not: a scrape is missed on one side and not the other, a canary is started
four minutes after the baseline window opens, a baseline is drawn from a longer stretch
of the previous release. Representing that shape in a both-sides-per-row file means
padding one column with blanks, which is a claim about alignment that the file cannot
support and the harness would have to un-do.

One row per side has no such claim in it. Each side is an independent list of
(timestamp, value); the two are compared as two samples, which is what the Mann-Whitney
comparator and every AI representation actually do. It is also the shape your telemetry
already has: a Prometheus `query_range` result or a Datadog metric export is one row per
timestamp per series, and turning it into this file is adding four constant columns —
`awk`, a spreadsheet, or ten lines of Python. No parser.

### JSON

Accepted as a secondary form, because it costs one additional reader function over the
same validator. Use it when you are generating the file from code and the CSV quoting is
more trouble than it is worth.

```json
[
  {
    "scenario": "checkout-2026-03-14T09:00Z",
    "label": "FAIL",
    "family": "deploy",
    "metrics": [
      {
        "metric": "checkout_p99_seconds",
        "kind": "latency",
        "direction": "increase",
        "unit": "s",
        "control": {"timestamps": ["2026-03-14T09:00:00Z", "..."], "values": [0.201, 0.199]},
        "canary":  {"timestamps": ["2026-03-14T09:00:00Z", "..."], "values": [0.204, 0.207]}
      }
    ]
  }
]
```

Both forms are normalised to the same internal rows and go through the same validator, so
they accept and reject exactly the same data. A `values` array may carry `null` for a
missing sample; the timestamps array must be the same length as the values array.

A metric object takes the optional keys the CSV takes as columns, including
`metric_order`. It is **stated, not inferred from the metric's place in the `metrics`
array** — for the same reason the CSV reads it out of a column rather than out of the row
sequence, and so that a file whose `metrics` array is re-sorted is still the same run.

---

## Required and optional fields

| column | type | required | meaning |
|---|---|---|---|
| `scenario` | string | **yes** | Identifies one window. All rows sharing it are one promotion decision, and it is the unit the verdict and the score are reported at. |
| `metric` | string | **yes** | Your own metric name. Used to group rows and recorded in the manifest. **Never shown to the model** — see [leakage](#metric-names-scenario-names-and-leakage). |
| `side` | enum | **yes** | `control` or `canary`. `baseline` is accepted for `control` and `experiment` for `canary`; all four normalise to Kayenta's `Control`/`Experiment`. |
| `timestamp` | RFC 3339 UTC, or epoch seconds, or epoch milliseconds | **yes** | Orders the samples, derives the step, and locates gaps. Integers of 10 digits are read as seconds and of 13 digits as milliseconds. A timestamp with no zone is read as UTC and counted in a warning: a uniform offset cannot change the ordering, the step or the gaps, which is all a timestamp is used for. |
| `value` | float | **yes** | The sample. Empty, `NaN`, `nan`, `null` and `None` all mean *missing*; see [gaps](#window-shape-length-gaps-and-step). |
| `kind` | enum | **yes** | One of `latency`, `errors`, `traffic`, `saturation`, `resource`, `memory`, `other`. A closed vocabulary, because this is the only thing about your metric the model is told. Constant per (scenario, metric). |
| `direction` | enum | **yes** | `increase` (higher is worse), `decrease` (lower is worse), `either`. Constant per (scenario, metric). |
| `label` | enum | no | `PASS` or `FAIL`, the verdict this window should have had. Constant per scenario. Absent = unlabelled; see [below](#running-without-labels). |
| `group` | string | no | The Kayenta scoring group. Defaults to `kind`. Group weights are split evenly over the groups present, as in the study. |
| `unit` | string | no | Recorded in the manifest. Not shown to the model, not used in judging. |
| `family` | string | no | A label *you* choose to group windows for reporting (`deploy`, `config-change`, `dependency`). Defaults to `own`. It never reaches the model, and it gives you the per-family accuracy table for free. |
| `metric_label` | enum | no | `pass` or `high`, the per-metric truth. Defaults from the scenario's `label`. Recorded, not scored. |
| `metric_order` | integer | no | This metric's place within its window. Distinct integers, and either every metric of a window declares one or none does. Absent = the metrics are ordered by `metric` instead. See [below](#metric-order-is-the-files-to-declare). |

Columns the loader does not recognise are ignored, and it prints which ones it ignored —
so an export that carries `job`, `instance` or `__name__` needs no editing, and a
misspelled required column still fails the required-column check rather than being
silently dropped.

### Metric order is the file's to declare

A window's metrics reach the judge in an order, and the order is not decoration.

- The only field of a metric pair the model is shown is its name
  (`judge-service/app/representations.py`), so the pair order **is** the order the
  metrics appear in the prompt.
- `eval_dataset._group_weights` splits 100 evenly over the groups present and hands the
  leftover hundredth to the **last** one: three groups are `33.33 / 33.33 / 33.34`, and
  which group gets the `33.34` is decided by which metric is last.

So the order is part of the configuration a window is judged under, and the file states
it. `metric_order` gives each metric its place, one-based by convention — any distinct
integers work, so numbering by tens or carrying a subset of a longer list is fine. The
loader reads the declaration out of a *column*, never out of the row sequence, which is
the one thing you are free to re-sort:

```csv
scenario,metric,side,timestamp,value,kind,direction,group,metric_order
checkout-2026-03-14,p99_seconds,control,2026-03-14T09:00:00Z,0.201,latency,increase,latency,1
checkout-2026-03-14,rss_bytes,control,2026-03-14T09:00:00Z,6.1e8,memory,increase,resources,2
checkout-2026-03-14,errors_per_second,control,2026-03-14T09:00:00Z,0.50,errors,increase,errors,3
```

Those three sort by name `errors_per_second`, `p99_seconds`, `rss_bytes` — a different
order, with a different group taking the leftover hundredth. Declared, the window is
judged as `latency, resources, errors` however its rows arrive.

A window that declares nothing is ordered by `metric`, which is what the loader has
always done. That rule makes a file shuffle-proof, and it is the only order two ends can
agree on without being told — but it is an order the *loader* invents, and for a window
whose metric names and metric kinds disagree it is usually not the one you had. Two
practitioners with the same telemetry, one exporting alphabetically and one by kind,
would otherwise be judged under two different classifier configurations with nothing in
either file recording the difference.

`tools/own_data.py --export` writes `metric_order` for every window, so an export and a
re-import give back the order as well as the values.

Both properties hold together, and `tools/test_own_data_contract.py::metric_order_is_declared`
asserts them on a three-metric window whose name order and kind order disagree:

- **deterministic** — two loads of one file, and of the same file with its rows
  shuffled, give identical scenarios, identical prompts and identical group weights,
  declared or not;
- **order-faithful** — a file that declares an order gets that order, whatever sequence
  its rows are in.

Declaring an order does not rename anything. The gid salt is a digest of the values, not
of the declaration, so adding the column to a file leaves every opaque name exactly as it
was.

The loader refuses, by line and column, a window that declares `metric_order` for some of
its metrics and not the others, one that gives two metrics the same number, and one whose
number is not an integer. There is no half-honoured declaration: every way of placing an
undeclared metric among declared ones is a guess about which group takes the leftover
hundredth.

### `direction` is required on purpose

It is the one field with no safe default. Guessing `increase` for a success-rate metric
inverts its verdict silently: the statistical comparator would then be asked whether the
success rate got *worse by going up*, and a collapse to 79% passes cleanly. That is a
wrong number that looks exactly like a right one, so the loader refuses to guess. (The
genuine judge handles all three values correctly; that was confirmed against a running
Kayenta while this was designed, and `decrease` and `either` both flag a 20% drop that
`increase` passes.)

Note the separate caveat in [What this does not give you](#what-this-does-not-give-you):
the *statistical* judge handles `decrease` correctly, but the frozen rubric the *AI*
judge uses is written for higher-is-worse metrics.

---

## Running without labels

**Labels are optional, and the judges run without them.** A practitioner will want to see
what the judges say about their windows before spending days labelling anything, and a
harness that refuses to start until every window is labelled makes that impossible.

With no `label` column, or with `label` empty for a given window:

- every judge runs exactly as it otherwise would, and every verdict, score, rationale and
  latency is written;
- `results/own-data/results_raw.csv` carries the full set of rows;
- the `correct` field of an unlabelled row is empty, never `0`;
- unlabelled rows are **excluded** from the confusion matrix, the accuracy, precision,
  recall, F1, FPR, FNR, McNemar and kappa. They are not counted as wrong, and they are
  not counted at all;
- `summary.csv` and `results.md` are still written, computed over the labelled subset
  only, and state the labelled count next to every figure. With nothing labelled, the
  scoreboard files say so in one line and carry no numbers.

Partial labelling is fine and is the expected case: label the incidents you remember,
leave the rest, and the scoreboard is computed over what you labelled while the verdicts
are produced for everything.

The reason for the hard separation is the defect recorded in the README's section on the
harness: a row that carries no judgement must never enter an accuracy denominator,
because an unlabelled window scored as `correct=0` inflates nothing and deflates
everything, and an unlabelled window scored as `correct=1` is a fabricated result.

---

## Window shape: length, gaps and step

The study's windows are exactly 31 samples at a 60-second step. Yours will not be. The
rule the loader follows throughout is: **never resample, never interpolate, never align.
Pass through what is there, or refuse it with a reason.** Anything else invents data,
and a judge's verdict on invented data is not evidence about the judge.

### A different sample count — supported

The number of samples is whatever your file contains. Nothing in the comparator or in
any representation is tied to 31: the Mann-Whitney test is a rank test over two samples,
and `summary`, `raw` and `plot` all read the arrays as given.

Two floors apply, both enforced at load time:

- **Fewer than 10 samples on either side: rejected.** At one sample the genuine
  `NetflixACAJudge` fails classification, and the harness records that as `score=100`,
  `verdict=PASS` — a silent false pass with no error anywhere. (Confirmed against a
  running Kayenta: a 1-vs-1 pair with a 35% regression returns classification `Error`,
  reason "Metric Classification Failed", and an overall score of 100.) Above that floor
  the judge classifies correctly but the tail statistics stop carrying information: for
  n < 21, `p95` is an interpolation between the two largest samples, and `p99` is for any
  window shorter than 101 samples. Since the rubric asks the model about p90/p95/p99
  explicitly, a short window quietly reduces three of the rubric's named signals to
  "the maximum". 10 is a harness choice, not a Kayenta requirement, and
  `--min-samples N` lowers it; the value used is recorded in the manifest, and lowering
  it below 10 prints a warning that names this paragraph.
- **A side shorter than half the other: rejected.** Comparing a 30-minute canary against
  a 3-hour baseline is legitimate practice, but every rubric variant in `prompts/` speaks
  about *the end of the observation window*, and with sides of very different spans that
  phrase refers to two different amounts of time. Trim the baseline and re-export.

Unequal lengths within that ratio are accepted, and the two sides need not have the same
sample count. This was confirmed against a running Kayenta: 31-vs-22 and 22-vs-31 pairs
classify correctly in both directions, on both a no-change and a +35% shift.

### Missing samples — supported, up to a point

An absent row and an empty/`NaN`/`null` `value` mean the same thing: no sample. Both the
statistical comparator (`nanStrategy: remove`) and the AI representations
(`representations.clean()`) drop missing samples before they are used, so a hole costs
you one sample rather than corrupting a series.

The harness emits JSON `null`, never a `NaN` float, because a `NaN` cannot be encoded in
JSON and the request to Kayenta would fail outright.

What a hole does distort is the one index-based signal: `slope_per_sample` in the summary
representation is fitted over the compacted array, so samples after a gap sit one step
earlier than they really were. A window missing 2 of 31 samples is unaffected in any way
that matters; a window missing a third of them has a trend line that describes a series
that never existed. So:

- **more than 20% of a side's expected samples missing: rejected**, naming the side and
  the count. `--allow-gaps` overrides it, and the manifest then records that the override
  was used and the missing fraction per side, so a result produced that way cannot be
  mistaken for one that was not.

Expected samples are derived as `round((last - first) / step) + 1` per side, from the
timestamps you supplied.

### A different step — supported

The step is per window, derived from the modal gap between consecutive timestamps, and
carried into the pairs as `stepMillis`. A 15-second or 5-minute step needs no flag.

Two constraints:

- **The two sides of a metric must share one step**, and **all metrics of one window must
  share one step** — the latter because a `metricSetPair` carries one scope per window.
  Split differently-sampled metrics into separate windows, or export them at one
  resolution.
- **Irregular spacing is rejected.** A gap is on-grid if it is within 5% of an integer
  multiple of the step; anything else is reported with the two offending timestamps.
  Resampling it for you is the one thing this loader will not do.

A step far from 60 seconds is honest but changes what the rubric's temporal language
refers to: 31 samples at 5 seconds is a two-and-a-half minute window, and "a degradation
still present at the end of the window" then means something much smaller than it does in
the study. The loader prints the derived window span per run for exactly this reason.

### Timestamps are not used to judge

Worth stating plainly: after the loader has used your timestamps to order the samples,
derive the step and find the gaps, they survive only as scope metadata. Neither the
comparator nor any representation reads them. **Wrong timestamps with the right order
give you the right verdict; the right timestamps in the wrong order do not.**

---

## Metric names, scenario names and leakage

The study namespaces its metric names with an opaque global id (`..._g0007`) so the
family name — and with it the ground truth — cannot reach the prompt. Your names carry
your meanings, and some of those meanings are the answer:
`checkout_latency_during_the_bad_deploy`, or a window called
`2026-03-14-rollback-FAIL`.

**The protection is on by default and is not a formatting preference.** The loader
renames everything the model could see:

- Each window is assigned an opaque id `own_0000`, `own_0001`, … by sorting your distinct
  scenario identifiers, so the mapping is stable: the same file always yields the same
  ids, and two runs over it are comparable.
- Each window is assigned a global id in `2000…9999` — disjoint from the seeded
  datasets' ranges (`0…` and `1000…`), so an own-data metric can never be confused with a
  study metric in a metric store, a log filename or a cache key. **Which** id inside that
  range is a salted hash of your own window identifier, not the window's place in the
  sorted file. It used to be the place (`2000 + index` over your sorted scenario ids), and
  that made the opaque suffix monotone in the alphabetical order of your window names: if
  your ids sort by outcome — and they do, if the outcome is in the name — the suffix ranked
  your windows by their answer, which is precisely the leak this renaming exists to
  prevent. The salt is a digest of the file's content, so the same window name in two
  different files gets two different suffixes and a suffix seen in one run says nothing
  about a window in another; it is recorded in the run manifest as `gid_salt`, so the names
  a run was judged under are re-derivable from the file it was judged from. Collisions are
  detected and resolved deterministically — the window that keeps a contested id is decided
  by the hash, never by where its name sorts — and a file carrying more than 8000 windows
  is refused rather than given two windows one id.
- Each metric is renamed `{kind}_g{gid:04d}` — `latency_g4831`, `errors_g4831` (4831 is an
  example; the number means nothing outside the one file it was drawn in). When a
  window carries more than one metric of a kind they are numbered from one and padded to
  the width of the count: `latency_1_g4831`, `latency_2_g4831`, `latency_3_g4831`, and
  `latency_01_g4831` … `latency_12_g4831` for a window with twelve of them. Numbering the
  first one and padding the rest is what makes the naming a fixed point — sorting these
  names gives back the order they were assigned in, so a window exported through
  `own_data.export()` and re-imported is the same window rather than a rotation of it. A
  re-import also keeps the id the file already carries rather than drawing a new one, for
  the same reason: the window id in an exported file is the `own_NNNN` the loader invented,
  not your identifier, so hashing *that* would rename every series on every round trip.
  The name the model sees is therefore drawn from a seven-word closed vocabulary plus a
  number, and cannot carry a meaning you did not intend.
- The metrics of a window are then ordered by that name, which is the order the file
  stores, the order the `metricSetPairList` carries into the prompt, and the order
  `eval_dataset._group_weights` walks when it hands the leftover 0.01 to a group.
- Your real metric name, your real window name, the unit and every unrecognised column
  are written only to the manifest, which stays on your disk.

This is worth being precise about, because it is checkable. The prompt the model receives
is `representation_text(mode, metricSetPairList) + rubric` (`judge-service/app/llm_judge.py`),
and the only field of a pair that reaches it is `name` (`representations.py`). Groups,
units, directions, the canary config's own name and description, and the `family` column
are not part of any representation. So renaming `name` is sufficient, not merely helpful.

`--keep-ids` disables the renaming for someone whose names are already neutral and who
would rather read their own metric names in `results_raw.csv`. The manifest records
`anonymised: false` when it is used, so a published result always says which was run.

---

## The statistical comparator

`STAT_CANARY_TEMPLATE` in `tools/eval_dataset.py` is one tuned, best-practice
configuration, applied uniformly to every metric of every scenario, and checked against
the Kayenta source. It is what makes the study's comparison fair, and it is the default
here for the same reason.

**Your service's tolerances are yours, and you can say so.** Pass a JSON file:

```bash
make own-data OWN_DATA=windows.csv STAT_CONFIG=tolerances.json
```

```json
{
  "effectSize": { "allowedIncrease": 1.10, "criticalIncrease": 1.50 },
  "outliers": { "strategy": "remove" }
}
```

- The file is deep-merged onto the template, key by key inside `effectSize` and
  `outliers`, so setting `allowedIncrease` alone leaves the other four gates at their
  tuned values.
- Overridable keys are `nanStrategy`, `outliers`, `effectSize` (its five sub-keys),
  `critical` and `mustHaveData`. A key outside that set is an error, not a no-op, so a
  typo cannot silently do nothing.
- **`direction` is not overridable here.** It is per-metric and comes from your CSV
  column; a global override would quietly contradict it.
- **`scoreThresholds` (pass 75 / marginal 50) are not overridable at all.** They are not
  a property of the statistical judge: the same cut turns the AI judge's self-reported
  score and all three hybrid policies into PASS or FAIL. Moving it changes what "FAIL"
  means for every judge at once, so a run under a different cut is not comparable with
  the article's on *any* judge, which is a much larger claim than "my service tolerates
  10%". If you need it, that is a fork, and it should be described as one.

Every run records the effective template in the manifest and prints it in the banner.
A run that used an override is marked in `results.md` with a line stating that its
numbers are not comparable with the published ones — including the derived
"Statistical config" line, which is generated from the effective template rather than
from the hardcoded study sentence.

---

## What changes in the code

Nothing you have to edit. This section is here because the article points at this
repository, and a reader should be able to see how small the change is and satisfy
themselves that the seeded path is untouched.

### `tools/eval_dataset.py` — three edits

**1. `Scenario` gains four optional fields.** Every existing constructor call passes the
same seven arguments and is unchanged.

```python
@dataclass
class Scenario:
    id: str
    family: str
    gid: int
    seed: int
    truth: str           # PASS | FAIL, or "" for a window with no label
    metrics: List[MetricSpec]
    params: Dict[str, Any] = field(default_factory=dict)
    # Supplied rather than generated. None -- the default, and every seeded
    # caller -- means the series come from the family generator.
    series: Optional[Dict[str, Dict[str, List[Optional[float]]]]] = None
    step_seconds: int = STEP_SECONDS
    base_millis: Optional[int] = None
    canary_overrides: Optional[Dict[str, Any]] = None
```

**2. `series_for_scenario()` returns supplied series when it has them.** Signature
unchanged:

```python
def series_for_scenario(scenario: Scenario) -> Dict[str, Dict[str, List[float]]]
```

An early return when `scenario.series is not None`, handing back copies of the lists so a
caller that mutates them cannot corrupt the source, and leaving `scenario.params` and
`MetricSpec.truth` alone — for a supplied scenario the loader has already set both, and
the generated path's two side effects on them must not run. When `series is None` the
existing body executes verbatim, which is every caller in `tools/` today:
`seed_eval_dataset`, `run_experiment`, `run_ensemble_experiment`, `run_frontier_experiment`,
`run_fp_guard_experiment`, `reproducibility_test_retest`, `figure_distributions`,
`figure_blindspot_series`, `figure_new_families`.

**3. `build_canary_config()` takes `direction` from the metric and merges the override.**
Signature unchanged:

```python
def build_canary_config(scenario: Scenario, judge: Dict[str, Any]) -> Dict[str, Any]
```

The per-metric block becomes
`{**STAT_CANARY_TEMPLATE, "direction": m.direction, **(scenario.canary_overrides or {})}`
with `effectSize` and `outliers` merged sub-key by sub-key.

This is byte-identical for the seeded path, and that is checkable rather than asserted:
every metric of both seeded datasets has `direction == "increase"`, which is the
template's own value, and updating an existing key leaves its position in the dict
unchanged. Serialising both forms for all 240 scenarios / 280 metrics of
`original-180` + `generalisation-60` at n=20 gives zero differences.

The override rides on the `Scenario` rather than on a new keyword argument so that not
one of the nine existing `build_canary_config` call sites has to change; it also leaves
room for per-service tolerances in one file later, without another API change.

### `tools/own_data.py` — new, and where all the work lives

Standard library only, like every host tool here.

```python
load(path, *, anonymise=True, min_samples=MIN_SAMPLES_PER_SIDE, allow_gaps=False,
     stat_config=None, limit=None) -> Tuple[List[ed.Scenario], Dict[str, Any]]
read_csv(path)  -> List[Dict[str, str]]     # normalised rows
read_json(path) -> List[Dict[str, str]]     # the same normalised rows
validate(rows, *, min_samples, allow_gaps) -> List[str]   # every problem, not the first
to_scenarios(rows, *, anonymise, stat_config) -> List[ed.Scenario]
build_manifest(scenarios, source, decisions) -> Dict[str, Any]
export(scenarios, path, *, base_millis=None, step_seconds=None, header=None) -> int
export_seeded(path, n_per_family=1) -> int  # what `make own-data-template` runs
describe(scenarios, manifest) -> str        # the per-window table
main() -> int                               # --check | --describe | --export
```

`validate()` returning every problem rather than raising on the first is deliberate: a
900-window export with three bad columns should cost one run, not three.

`export()` writes scenarios out in this format. It is how `make own-data-template`
produces a file that is valid by construction, it is what makes the round-trip test
below a real test, and it is there for anyone who wants to see what the format looks
like for data whose answers they already know. It uses a fixed window start, so the
same export is the same bytes every time.

`limit` on `load()` is what `--quick` uses. The whole file is still validated — a smoke
run that reported success for a file the real run then refuses would be worse than no
smoke run — and the manifest records three things rather than one: `n_windows`, the
number that ran; `n_windows_in_file`, the number the file held; and `limited_to`, the
cut. The count of the file is taken before the truncation, because nothing after it
remembers the file was longer.

That is recorded so that the scoreboard can carry it. A `--quick` run writes a trailing
`note` column in `summary.csv` — added only when there is something to say, so a full
run's header is unchanged — and a **This is a smoke run** line at the top of
`results.md` and `summary.md`, both naming the two counts and the file. A scoreboard read
a week later, without the console output beside it, is otherwise a perfectly ordinary
scoreboard over a subset, with nothing in it to say so.

### `tools/run_experiment.py` — flags and one branch

New arguments, all defaulting to the current behaviour: `--own-data` (default
`$OWN_DATA`), `--stat-config` (default `$STAT_CONFIG`), `--out` (default `results`, or
`results/own-data` when `--own-data` is given), `--keep-ids`, `--min-samples`,
`--allow-gaps`. With `--own-data`, `main()` takes its scenarios and manifest from
`own_data.load()` instead of `ed.build_scenarios()`, skips the VictoriaMetrics seeding
step, and passes each window's own `base_millis` and `step_seconds` to `run_scenario()`.
`_row()` writes an empty `correct` for an unlabelled window, and the scoreboard is
computed over labelled rows only. `--quick` means the first two windows instead of
n=1 per family. The run's manifest is written to
`results/own-data/own_data_manifest.json`; `data/eval_manifest.json`, which
describes the seeded dataset, is left alone.

Extending the existing runner rather than adding a second one is the point: the
scoreboard, the confusion matrix, the McNemar test and the kappa are then literally the
same code that produced the article's numbers, not a reimplementation that can drift away
from it.

`--out` is resolved to an absolute path before anything downstream uses it, and it may
point anywhere you can write. The one thing it costs you outside the repository is the
figures: both figure runners read a path inside this repository — the running
judge-service has only `data/ai-logs` mounted, and the one-off container mounts the
repository — so a spec outside it is reported as a skip, in one line, and the run
finishes. The tables and both reports are written either way; they are what the run is
for. A relative `--out` used to leave the figure spec relative too and end the run in
`ValueError: '<out>/_exp_figdata.json' is not in the subpath of '<repo>'` — after every
window had been judged and every table written, and before `results.md` and `summary.md`
were.

Your judge-call payloads go to `data/ai-logs/own-data/<run timestamp>/`, one directory
per run, and the run prints the path. `data/ai-logs/` itself is the study's provenance
archive, and `tools/replay_ai_logs.py` — which `make replay-logs` runs — globs `*.json`
over that directory and replays everything in it. Writing your payloads there meant
scoring your own windows changed what the replay reports about the study's archive, and
attributed your calls to the study's models. The subdirectory is outside that glob. The
service is told where to write per request, in the canary config beside `mode` and
`model`, so the seeded run keeps writing exactly where it always has.

Your data is **not** written to VictoriaMetrics. It already lives in your own metric
store, the judges read the in-memory pairs, and copying it into a container adds a second
copy and no capability. Running a live analysis through Kayenta against your own
Prometheus is a different workflow, and the README's
[Using your own metrics](../README.md#using-your-own-metrics) covers it.

### `tools/test_own_data_roundtrip.py` — new

Exports the seeded scenarios through `own_data.export()`, reads them back through
`own_data.load()`, and asserts that the resulting `metricSetPairList` and canary config
are identical to the ones the generated path produces. If the two ever differ, the claim
this whole page rests on — that your windows reach the judges the same way the study's do
— is false, and the test says so.

The file carries the order of the metrics within a window, in `metric_order`, and the
comparison is made in sequence with nothing canonicalised: a round trip that reorders a
window is a failure. It used not to be. The exporter wrote a window's metrics sorted by
the `metric` column and the loader read them back sorted by the same column, and this
test compared against the source scenario re-sorted the same way — so the reordering was
the fixture rather than the finding. Eight of the nine families carry one metric, so the
two orders coincide there; `cross_metric_marginal` declares latency, memory, errors where
sorting the names gives latency, errors, memory, and that moves both the prompt order and
the group that takes the leftover hundredth of the classifier's weights. See
[Metric order is the file's to declare](#metric-order-is-the-files-to-declare).

Sharing the key is necessary and was not sufficient, because under the default the loader
does not read the name back — it derives a new opaque one from it. `anonymised` is the
second round-trip check and the one that covers that: it exports an already-renamed window
and re-imports it twice, over three windows (three metrics of one kind, three kinds, and a
single metric), and compares the `metricSetPairList` and the canary config each time. It
fails on the naming scheme this page used to describe, where the first metric of a kind was
unnumbered: `latency_gNNNN`, `latency_2_gNNNN` and `latency_3_gNNNN` sort `_2`, `_3`, then
the bare one, because a digit sorts before a letter, so each series came back under its
neighbour's name and rotated once more on every round trip. Nothing in the harness reaches
`export()` with a renamed window today — `--export` exports seeded scenarios, and every
seeded window carries one metric per kind — which is the reason it is checked here rather
than left for the first person whose window has two latencies in it.

`tools/test_own_data_contract.py` goes further, over all 180 scenarios: the same
comparison, plus a field-by-field inventory of what an export does *not* carry back
(`Scenario.gid`, `Scenario.seed`, `Scenario.params`, `MetricSpec.base` — each asserted,
so a fifth joining them fails the check), plus a digest comparison of the current
`eval_dataset` against `git show main:tools/eval_dataset.py` over both seeded datasets.

---

## Make targets

```make
OWN_DATA ?=
STAT_CONFIG ?=
MIN_SAMPLES ?=
ALLOW_GAPS ?=
KEEP_IDS ?=

own-data-template: venv ## Write a valid example input file in the own-data format (OUT=<file>)
own-data-check: venv require-own-data ## Validate your own windows and print what the harness will see (OWN_DATA=<file>)
own-data: ensure-up venv wait-kayenta require-own-data ## Run the scored experiment over your own windows -> results/own-data/ (OWN_DATA=<file>)
own-data-quick: ensure-up venv wait-kayenta require-own-data ## Smoke run over the first two of your own windows (OWN_DATA=<file>)
```

Every switch this page names has a variable, and all three targets that read your file
pass all of them, so nothing documented here is reachable only by invoking the Python
directly:

```bash
make own-data-check OWN_DATA=windows.csv MIN_SAMPLES=8 ALLOW_GAPS=1
make own-data       OWN_DATA=windows.csv MIN_SAMPLES=8 ALLOW_GAPS=1
make own-data       OWN_DATA=windows.csv KEEP_IDS=1 STAT_CONFIG=tolerances.json
```

A variable left empty is not passed, so the loader's own default applies.
`own-data-template` takes none of them: it writes an example file rather than reading
yours, and its only knob is `OUT=<file>`.

`own-data-check`'s closing line echoes the switches actually in force, so the command it
hands you accepts the file it has just accepted. A line that printed a bare
`make own-data OWN_DATA=…` after validating under `MIN_SAMPLES=8` would be handing you a
command that rejects that very file.

`require-own-data` follows `require-results` and `require-stack`: no `## ` comment, so it
stays out of `make help`, and it fails with a sentence telling you to set `OWN_DATA` or
put it in `.env` — which the Makefile already reads, so `OWN_DATA=./windows.csv` in `.env`
makes every target above work with no argument.

`own-data-check` needs neither the stack nor a model. It is the target to run first, and
it prints, per window: the opaque id, your original name, the metric count, the samples
per side, the derived step and span, the missing count, the label if any, and the
renamed metrics — so you can see precisely what will be judged before a single model call
is paid for.

### Re-deriving the tables offline

`make analysis` reads one results directory, named by `RESULTS`, which defaults to
`results/` (where `make experiment` writes). Your run is one argument away:

```bash
make analysis RESULTS=results/own-data
```

That re-derives the scoreboard with trivial baselines and MCC, and the threshold-free
ROC/PR curves, from `results/own-data/` — no stack, no model, no network. The trivial
baselines are computed from *your* class balance, counted from your own labelled
windows, not from the study's 120 FAIL / 60 PASS. The ROC/PR table is over *your*
configurations: `roc_pr_curves.py` reads which ones to sweep off the run in front of it —
the statistical judge, the strongest `ai:*` configuration in your `summary.csv`, the
gated hybrid on that same model, and any hosted model you have a `frontier_long_*.csv`
for — rather than off a list of the study's five. It picks the strongest local model by
the same rule and out of the same file that `figure_approach_bars.py` uses for its "best
local" column, so the two artefacts of one run cannot name different models, and it says
"tied N ways" when the lead is shared instead of calling one of them best. Because your
run may hold windows you did not label, the table carries a trailing
`n_unlabelled_dropped` column beside `n`: an ROC over 7 of 8 windows and an ROC over 8
are different measurements, and the dropped ones are not negatives.

That derivation is for **your** run. The seeded run sweeps the five configurations the
article reports, named in `roc_pr_curves.py` rather than inferred from a directory
listing, and its `roc_pr_auc.csv` keeps the seven columns it has always had — no
`n_unlabelled_dropped`, because the seeded dataset labels every window. A run is read as
yours when it carries an `own_data_manifest.json` that names the windows in the
`results_raw.csv` beside it. That file is what `run_experiment.py --own-data` writes and
nothing else does, so the answer travels with the directory rather than depending on
where it was unpacked; it is checked against the run it claims to describe, so a manifest
left behind by an earlier run cannot answer for a later one. A seeded run also unlinks
any manifest it finds in the directory it is about to fill, so the two halves are that
the marker is not left behind and that one left behind by any other route does not
count. Derived for the seeded run instead, the published five-row table came out with
nine rows — the study's archived `agg/` holds three hosted models, not one — and
`fig13_threshold_summaries.png`, whose layout is built for five groups, regenerated with
overprinted tick labels on numbers that had not moved.

`RESULTS` is compared as a directory rather than as a string, so `RESULTS=results`,
`RESULTS=./results`, `RESULTS=results/` and an absolute path all mean the seeded run.

A run in which nothing is labelled is a documented mode (above), and `make analysis`
completes on one: `baselines_and_mcc.py` reports that there is no class balance to
compute a baseline from and skips, rather than failing the target before
`roc_pr_curves.py` — which handles the same case — has run.

Three of the offline scripts are not run for a results directory other than the seeded
one, and the target says so rather than running them against `results/`:
`bounded_confidence_intervals.py` and `pooled_mcnemar.py` need the five-seed sweep, which
an own-data run does not produce, and `run_fp_guard_experiment.py`'s guard ablation is
defined against the seeded families (see caveat 6 below). Run bare, each of the three now
says what is missing and why it cannot stand in for it.

### Figures from your own run

```bash
make figures RESULTS=results/own-data
```

Run `make analysis RESULTS=results/own-data` first: one of the generators draws from the
ROC/PR table that target writes.

The figure generators are split the same way, and for the same reason. Every one of them
*accepts* `--results`; four of them do not *read* it. Those four —
`figure_distributions.py` and `figure_blindspot_series.py`, which draw seeded instances
from `eval_dataset.py`; `figure_multiseed_intervals.py`, which needs the five-seed sweep;
and `figure_diagrams.py`, which is four schematics that read nothing — are run for the
seeded run only, and named as skipped otherwise. They are not merely useless against your
run: run against it they wrote the study's figures into your directory under the study's
filenames.

The figure spec is part of the run too. `run_experiment.py` writes it beside your results
as `results/own-data/_exp_figdata.json`, and `make figures RESULTS=results/own-data` reads
it from there. The seeded run keeps writing it to `data/ai-logs/_exp_figdata.json`, where
it has always lived. Sharing one path meant an own-data run overwrote the study's figure
spec in place, and the next `make figures` redrew three of the article's figures from
someone else's windows with nothing to say so.

A run in which nothing is labelled gets the same treatment here as it gets from
`make analysis`. Every figure this target draws is a picture of accuracy, and such a run
has none, so `figures` says so up front, names the three scripts it is not running and
why, writes nothing, and exits 0 — an unlabelled run is a documented mode, not an error.
It used to write an empty `fig10_approach_bars.png` with all five columns SKIPPED and then
exit 1 out of `figure_threshold_summaries.py` telling you to run `tools/roc_pr_curves.py`
first, which cannot help: `roc_pr_curves.py` correctly writes nothing for a run with no
labels.

---

## What this does not give you

This list is the honest limit, and it belongs next to every number this harness produces
from your data.

1. **You still have to label your windows, and labelling is the hard part.** Nothing here
   automates it, and nothing here checks that a label is right. A wrong label does not
   produce an error; it produces a confident, wrong accuracy.
2. **You have to label benign look-alikes too.** A dataset made only of real incidents
   makes every judge look excellent, because a judge that always says FAIL scores 100%.
   The study's dataset is a third genuine-PASS families — the healed transient, the
   equal-variance noise, the no-change deploy — precisely so its FAIL numbers mean
   something. Windows you promoted and did not regret are as much data as the rollbacks,
   and are harder to remember.
3. **Your telemetry values are sent to whatever model you configured.** The renaming
   protects your metric and incident *names*; it does nothing about the numbers, which
   are the payload. With a local Ollama model nothing leaves the machine. With a hosted
   model, your production values leave your network, under that provider's terms. That is
   your decision to make, and the harness will not make it quietly: a run against a
   non-local alias prints which provider it is about to send to.
4. **Hosted models need credentials, local models need to be pulled.** Absent models are
   skipped rather than failed, so a run can quietly evaluate fewer judges than you
   expected; the skipped list is printed and recorded.
5. **Without labels there is no accuracy**, only verdicts. That mode is supported and
   useful, and it is not an evaluation.
6. **A handful of windows cannot settle anything, and nothing here will put an interval
   around yours.** Twenty windows give an accuracy whose confidence interval spans most
   of the plausible range, and the harness computes no power analysis and will not warn
   you that your sample is too small for the difference you are looking for. The bundled
   interval tooling does not help: `tools/bounded_confidence_intervals.py` is built for
   the study's five-seed sweep — its main table re-aggregates a multi-seed
   `long_results.csv` that only `make analysis-live` produces, and its single-seed table
   reads `results/` directly, checks the published statistical accuracy of 0.544 and
   assumes the nine seeded families at twenty instances each. An own-data run produces
   none of that, so `make analysis RESULTS=results/own-data` does not run it. Run bare
   after a `make own-data` on a machine that has also run `make experiment`, it reports
   the **study's** intervals, which are not about your data at all. The same goes for
   `tools/pooled_mcnemar.py` and for the guard ablation in
   `tools/run_fp_guard_experiment.py`, whose two detectors are defined against the seeded
   families and which refuses a results directory whose scenarios are not those rather
   than reporting an ablation where nothing could fire. Treat a scoreboard from a small
   own-data run as a description of those windows, not as a measurement of the judges.
7. **The frozen rubric assumes higher-is-worse.** It tells the model that "clearly
   better/lower is `low`". For a `direction=decrease` metric — a success rate, a cache hit
   ratio, a throughput floor — the statistical judge is correct (verified: a 20% drop is
   flagged under `decrease` and `either`, and passed under `increase`), but the AI and
   hybrid judges are being run outside the conditions the study measured, and may read a
   harmful drop as an improvement. The loader prints a warning naming each such metric.
   Selecting a different rubric is one environment variable (`JUDGE_PROMPT_ID`, see
   `make prompts`), and doing so also puts you outside the study's conditions.
8. **Nothing is resampled, interpolated, aligned or outlier-trimmed for you.** Windows the
   harness cannot judge honestly are rejected with a reason rather than repaired.
9. **Results from an overridden statistical template are not comparable to the published
   ones**, and are marked as such wherever they are written.
10. **Window shapes far from 31 samples at 60 seconds change what the rubric's temporal
    language refers to**, even where the harness accepts them without complaint.
11. **This does not fetch from Prometheus, Datadog or anything else.** You export; the
    harness reads a file. That boundary is deliberate — it is one format to support
    instead of a dozen APIs to keep working — but it is a step you do.
12. **An own-data run produces no Referee report and writes nothing to VictoriaMetrics.**
    You get `results/own-data/`, not a rendered canary dashboard.

---

## The claim this supports

Stated as conservatively as it can be, and about the harness rather than about anyone's
experience of using it:

> The released testbed reads an organisation's own labelled canary windows from a
> documented long-format file and runs the same statistical, AI and hybrid judges, under
> the same configuration and the same scoring code, as it runs over the article's seeded
> dataset — without modification by the user.

It does not claim that anyone has done this in production, that the study's measured
differences between judges hold on other telemetry, or that a result obtained this way is
comparable to the published one unless the run reports that it used the default
configuration.
