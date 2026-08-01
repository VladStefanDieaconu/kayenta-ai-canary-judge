<p align="center">
  <img src="assets/canary.svg" alt="Kayenta AI Canary Judge" width="150" height="150">
</p>

<h1 align="center">Kayenta AI Canary Judge</h1>

<p align="center">
  An AI judge for Kayenta automated canary analysis.<br>
  Statistical, LLM/VLM, and hybrid judges, scored on a labelled dataset.
</p>

A Docker Compose testbed that runs a complete automated canary analysis pipeline
three ways and scores the verdicts against ground truth. The three judges are the
statistical test Spinnaker/Kayenta ships with, a configurable AI judge (a language
model or a vision model), and a hybrid of the two.

**The question it answers.** Kayenta's `NetflixACAJudge` compares control and
canary with a Mann-Whitney rank test, which detects a shift in the median. It is
therefore blind to a change in variance and blind to temporal order: a canary
whose latency has the same median but ten times the spread passes, and so does one
that is flat for most of the window and ramps at the end. Can a model that reads
the whole distribution cover that blind spot, and can the two be combined without
giving up the statistical judge's precision? This testbed measures it, on a
labelled synthetic dataset, and writes the numbers to `results/`.

It is built to be pointed at **your** metrics and **your** models. Everything runs
locally on open-weight models through Ollama; the same judge code reaches AWS
Bedrock, OpenAI or Anthropic by editing one config file, with no code change.

The results of the study this testbed was built for are published separately; see
[the study section](#the-study-and-its-data).

## Contents

Start here:

- [What automated canary analysis is](#what-automated-canary-analysis-is)
- [The three judges](#the-three-judges)

- [Quick start](#quick-start)
- [Hardware and runtime](#hardware-and-runtime)

Making it yours:

- [Using your own metrics](#using-your-own-metrics)
- [Bringing your own model or provider](#bringing-your-own-model-or-provider)
- [Adding your own scenario family](#adding-your-own-scenario-family)
- [Rendering figures from your own results](#rendering-figures-from-your-own-results)
- [The study and its data](#the-study-and-its-data)

Reference:

- [Command reference](#command-reference)
- [Architecture](#architecture)
- [The AI judge](#the-ai-judge)
- [The hybrid judge](#the-hybrid-judge)
- [The evaluation dataset](#the-evaluation-dataset)
- [Configuring the statistical judge fairly](#configuring-the-statistical-judge-fairly)
- [Running the experiment and reading `results/`](#running-the-experiment-and-reading-results)
- [The stress test](#the-stress-test)
- [Reading a single result](#reading-a-single-result)
- [Viewing a result in Referee](#viewing-a-result-in-referee)
- [Configuration reference](#configuration-reference)
- [Build notes and operational constraints](#build-notes-and-operational-constraints)
- [Repository layout](#repository-layout)
- [Scope and limitations](#scope-and-limitations)
- [License](#license)

## What automated canary analysis is


Automated canary analysis is how a continuous-delivery system decides whether a new
build is safe to promote. Route a slice of traffic to the new version (the canary,
or experiment), keep the rest on the current version (the baseline, or control),
measure both with the same metrics, and hand the two sets of measurements to a judge.
The judge returns a score and a verdict, and the verdict gates promote-versus-roll-back.

The de-facto standard judge is Spinnaker/Kayenta's `NetflixACAJudge`. It compares the
two series with a Mann–Whitney U rank test, which tests for a shift in the median
(the location) of the distribution. Two consequences fall straight out of that
choice: the test is blind to a change in variance, and it is blind to temporal order.
Both are documented upstream in
[spinnaker/spinnaker#6278](https://github.com/spinnaker/spinnaker/issues/6278) and
visible in the Kayenta source (`MannWhitneyClassifier.scala`). The per-metric knobs
that operators reach for (`direction`, `effectSize`, critical thresholds) only tune
how a median shift is graded; none of them help when the median has not moved.

That blind spot is the thing this harness measures.

## The three judges


All three run as real Kayenta analyses and are chosen entirely by canary config. One
judge service plays every role:

| Judge | `judge.name` | How it works |
|---|---|---|
| Statistical | `NetflixACAJudge-v1.0` | Kayenta's built-in Mann–Whitney judge, in-process. Not reimplemented. |
| AI | `RemoteJudge-v1.0` | Kayenta forwards the metric pairs to `judge-service`, which builds a representation (`summary`, `raw`, or `plot`), calls one OpenAI-compatible gateway (LiteLLM in front of Ollama / Bedrock / OpenAI / Anthropic), and returns a verdict. A no-op `dummy` mode is the model-free default. |
| Hybrid | `RemoteJudge-v1.0` (`mode=hybrid`) | The service calls back into Kayenta to run the genuine `NetflixACAJudge` on the same pairs, computes its own AI verdict, and merges the two under a selectable policy (`gated`, `or`, or `and`). |

The AI judge never branches on the provider. Provider selection lives in
`litellm/config.yaml`; the judge only branches on modality (text versus vision).

## Quick start

```bash
cp .env.example .env
# then set MINIO_ROOT_USER and MINIO_ROOT_PASSWORD in .env to values of your own
make validate               # brings the stack up and runs the model-free regression
```

Those two are the only values you must change. They ship as `CHANGE_ME_*`
placeholders rather than as a working pair, so the stack fails loudly instead of
starting on a password published in a public file. MinIO wants at least eight
characters. Everything else in `.env.example` is a port, an image tag or a
default you can leave alone.

`make validate` is the functional gate and the smallest possible first run. It
starts the stack, waits for Kayenta to report healthy, seeds a small dummy
dataset with `tools/seed_dummy_data.py`, and runs the statistical, dummy-AI and
hybrid judges end to end. It exits 0 only if all three return a verdict, and it
needs no model.

**Two seeders, for two different jobs.** `seed_dummy_data.py` writes a handful of
points so the pipeline can be exercised in seconds; `seed_eval_dataset.py`
(`make seed-eval`) generates the full labelled nine-family benchmark that the
scored experiment is measured against. Start with the first; you only need the
second when you want a scoreboard.

### Running against the bundled example data

`make demo` renders a figure from data committed to this repository — no stack,
no model, and after the first run no network:

```bash
make build                  # once: figures render inside the judge-service image
make demo                   # -> results/figures-demo/
```

The `make build` step is what needs the network. Figures are drawn inside that
image because matplotlib is deliberately kept out of the host virtualenv, so the
image has to exist before `make demo` will run. Once it does, the demo reads
nothing but committed files.

The input is `example-results/`, a small slice of a real run kept so the analysis
and figure path can be exercised without spending hours of inference first. It is
example data and nothing more: it is far too small to draw a conclusion from, and
it is not the dataset behind the paper. See
[the study section](#the-study-and-its-data) for that.

### Running the real thing

To exercise the AI judges and the scored experiment, run some models locally with
Ollama:

```bash
brew install ollama
OLLAMA_HOST=0.0.0.0:11434 ollama serve &            # bind 0.0.0.0 so the containers can reach it
ollama pull qwen2.5:7b && ollama pull moondream:v2  # plus any others you want scored
make up                                             # (re)build and start the full stack
make validate-ai                                    # sanity-check each model present
make experiment-quick                               # a small smoke run of the experiment
make experiment                                     # the full scored experiment -> results/
```

The harness scores whatever models are present locally and skips the rest, so a
run is bounded by what you have pulled.

## Hardware and runtime

The stack itself is small. What costs time is inference.

| | |
|---|---|
| Docker with Compose v2, and `make` | required |
| Python 3.11 on the host | the `make` targets build a `.venv` from `tools/requirements.txt` |
| one reachable Python package, `requests` | the venv is created with `--system-site-packages`, so a host-installed copy is inherited; behind a gated index with no system copy, `make venv` fails loudly and every host-side target is unavailable |
| RAM | ~6 GB for the seven-service stack, plus whatever the models need |
| Disk | the stack images are ~3 GB; the eight models in the study are ~40 GB together, from 1.7 GB (Moondream) to 9.1 GB (Phi-4) |
| GPU | not required. Ollama runs on the host because the Metal GPU is not reachable from Docker on macOS |
| Tested on | macOS (Apple Silicon) and Linux (amd64) |

Measured runtimes on an Apple Silicon laptop, eight local models between 1.8B and
14B parameters:

| run | scale | model calls | wall clock |
|---|---|---|---|
| `make demo` | committed data | 0 | seconds |
| `make validate` | model-free regression | 0 | ~2 min including stack start |
| `make experiment-quick` | 9 scenarios, 2 models | 27 | ~2 min |
| `make experiment` (`EVAL_N=20`) | 180 scenarios, 8 models | 2,340 | ~5 h |
| `tools/run_multiseed_corrected.py` | 225 scenarios, 5 seeds, 8 models | 2,925 | ~7 h |

Per-call latency is what scales, and it varies by roughly an order of magnitude
across models: the median AI call ranges from 2.5 s (Moondream on a rendered
plot) to 16.4 s (DeepSeek-R1 on summary statistics), because a reasoning model
spends its budget reasoning. Estimate a run as the sum of your models' median
latencies times the scenario count, then add model-swap overhead if Ollama has to
page models in and out.

Both long runs checkpoint after every seed and can be killed and restarted.

## Using your own metrics


The synthetic dataset exists so the judges can be scored against ground truth. To point
the harness at a real service instead, four things change and one capability is lost.
The worked example below uses a service that exports
`http_request_duration_seconds_p95` and `http_requests_errors_total`, with a `version`
label separating the two deployments.

### 1. Point Kayenta at your own metric store

`kayenta/config/kayenta.yml` declares one Prometheus-compatible account named `vm`:

```yaml
kayenta:
  prometheus:
    enabled: true
    accounts:
      - name: vm
        endpoint:
          baseUrl: http://victoriametrics:8428
        supportedTypes:
          - METRICS_STORE
```

Change `baseUrl` to your own Prometheus, VictoriaMetrics or Thanos query endpoint. Keep
the account name `vm`, or rename it and pass the new name as `metricsAccountName` on
every request. From inside the compose network the URL must be reachable from the
Kayenta container, so a store on your host is `http://host.docker.internal:9090`, not
`http://localhost:9090`. Reload with:

```bash
docker compose up -d kayenta
```

The bundled VictoriaMetrics can be left running and unused; nothing writes to it unless
you run one of the seeders.

### 2. Write a canary config for your own metrics

Copy `kayenta/canary-configs/statistical.json` and replace the `metrics` array. Each
entry names the metric, the PromQL that fetches it, and the group it scores under:

```json
{
  "name": "p95_latency",
  "query": {
    "type": "prometheus",
    "serviceType": "prometheus",
    "metricName": "http_request_duration_seconds_p95",
    "customInlineTemplate": "PromQL:sum without(version) (http_request_duration_seconds_p95{job=\"checkout\",version=\"${scope}\"})",
    "labelBindings": []
  },
  "groups": ["latency"],
  "analysisConfigurations": { "canary": { "direction": "increase" } },
  "scopeName": "default"
}
```

The `customInlineTemplate` **must** collapse the label that distinguishes the two
deployments — `sum without(version) (...)` above. Kayenta substitutes `${scope}` once
for the control and once for the experiment and expects each query to return exactly
one series. Leave the label in place and each query returns two, the two scopes never
pair, and every metric comes back `Nodata` with no error.

`direction` is `increase` when higher is worse, `decrease` when lower is worse, and
`either` when any movement matters. Group weights are distributed evenly across the
groups present.

### 3. Substitute your own control and experiment labels

The synthetic dataset uses `Canary="Control"` and `Canary="Experiment"`. A real
deployment usually separates the two by version, replicaset or colour, so
`version="1.4.2"` and `version="1.5.0"` take those places. Whatever the label is, it is
the one the template must collapse in step 2.

The scope object is flat. `controlScope` and `experimentScope` are plain strings with
`startTimeIso`, `endTimeIso` and `step` as siblings, not nested objects:

```json
{
  "scopeName": "default",
  "controlScope": "1.4.2",
  "controlLocation": "vm",
  "experimentScope": "1.5.0",
  "experimentLocation": "vm",
  "startTimeIso": "2026-07-27T09:00:00Z",
  "endTimeIso": "2026-07-27T09:30:00Z",
  "step": 60,
  "extendedScopeParams": {}
}
```

`controlLocation` and `experimentLocation` are the metric account name.

### 4. Run one analysis over a real window

```bash
curl -s -X POST \
  "http://localhost:8090/standalone_canary_analysis/?metricsAccountName=vm&storageAccountName=minio-store&application=checkout&user=you" \
  -H 'Content-Type: application/json' \
  -d '{
    "canaryConfig": { ...the config from step 2... },
    "executionRequest": {
      "scopes": [ ...the scope from step 3... ],
      "thresholds": { "pass": 75, "marginal": 50 },
      "lifetimeDurationMins": 25,
      "analysisIntervalMins": 25,
      "beginAfterMins": 0,
      "lookbackMins": 0
    }
  }'
```

The response carries an execution id. Poll it at
`GET /standalone_canary_analysis/{id}`, and read the verdict as described under
[Reading a single result](#reading-a-single-result). To view it rendered, open
`http://localhost:3001/dashboard/reports/standalone_canary_analysis/{id}`.

To use the AI or hybrid judge instead, set `judge.name` to `RemoteJudge-v1.0` and put
`mode` and `model` in `judgeConfigurations`, exactly as the shipped AI canary configs do.

### 5. What you lose

**Without ground-truth labels there is no accuracy, precision or recall.** The
scoreboard exists because every synthetic scenario carries a known PASS or FAIL label to
score the verdict against. Your own traffic has no such label, so on your own metrics
you get verdicts, scores and Referee reports — the operational output — and no
scoreboard.

`make experiment` will not work on your data. It generates its own labelled dataset,
seeds it, and scores against it; pointing it at a real service is not a matter of
configuration. Scoring judges on your own service means labelling your own incidents
first, which is a data-collection exercise this repository does not automate.

## Bringing your own model or provider


The judge always calls the one OpenAI-compatible gateway with a model alias, so
switching between local, Bedrock, OpenAI, and Anthropic is three edits and no code
changes:

1. Add an alias in `litellm/config.yaml` (commented examples are included):
   ```yaml
   - model_name: bedrock-claude
     litellm_params:
       model: bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0
       aws_region_name: os.environ/AWS_REGION_NAME
       aws_access_key_id: os.environ/AWS_ACCESS_KEY_ID
       aws_secret_access_key: os.environ/AWS_SECRET_ACCESS_KEY
   ```
2. Provide credentials in `.env` (`AWS_*`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`; they
   pass through and are no-ops when unset).
3. Register the alias's modality in `judge-service/models.yaml` and point a canary
   config, or `JUDGE_MODEL`, at it.

Then `make up` to reload the gateway config, and run an analysis. For vision or `plot`
mode, mark the alias `modality: vision` so the judge attaches the chart image.

## Adding your own scenario family


The evaluation dataset is generated, not recorded: 31 samples per series over a 30-minute
window, from a per-instance seeded RNG, so every scenario is reproducible from the master
seed alone. A **family** is one failure shape plus its ground-truth label, and each
instance draws its parameters from a range so the family is a distribution of cases rather
than one hand-picked example.

Everything lives in `tools/eval_dataset.py`:

1. **Declare it** in a dataset's spec — the truth label and which metric kinds it uses:

   ```python
   "my_family": {"truth": "FAIL", "kinds": ["latency"]},
   ```

2. **Generate it** by adding a branch to `_gen_kind()` that returns
   `(control, experiment, per_metric_truth, params)`. Record every drawn parameter in
   `params`; it goes into the manifest and is what makes a result explicable later.

   ```python
   if family == "my_family":
       shift = rng.uniform(1.2, 1.4)
       p["shift_factor"] = round(shift, 3)
       return _noisy(rng, base, noise, n), _noisy(rng, base * shift, noise, n), "high", p
   ```

3. **Validate it before spending anything on a model.** Run the statistical judge and the
   tuned ensemble over it and check they do what you expect;
   `tools/validate_new_families.py` is a worked example that also measures the shape
   properties each family claims to have.

Two constraints that are easy to miss:

- **Never insert a family into the middle of `FAMILY_SPEC`.** The `gid` counter runs across
  all families in declaration order and feeds each scenario's seed, so inserting one
  re-seeds every scenario after it and renames every metric. Append, or add a new dataset
  entry in `DATASETS` with its own `gid_base` — which is what `generalisation-60` does, so
  that the original 180 stay byte-identical.
- **The family name must never reach the prompt.** Metric names are namespaced with an
  opaque global id (`http_request_duration_seconds_g1042`), never the family, or the
  ground truth leaks into the input.

The three families in `generalisation-60` (`partial_recovery`, `late_transient`,
`sustained_excursion`) are the reference implementations of all of this.

## Rendering figures from your own results

Every figure generator takes the results directory as an argument. None of them
globs a fixed path — that is how a figure in the original study silently gained
six columns when unrelated files landed in the directory it was scanning.

```bash
make figures                                    # every figure, from results/
python tools/figure_approach_bars.py --results results --out results/figures
python tools/figure_threshold_summaries.py --results my-run --out my-run/figures
python tools/figure_prompt_comparison.py --experiments my-run/agg/experiments
```

Every one takes an input and an output:

| generator | input | output | draws |
|---|---|---|---|
| `figure_distributions.py` | `--results`, `--seed`, `--n` | `--out` | location shift against equal-median shape change |
| `figure_blindspot_series.py` | `--results`, `--seed`, `--n` | `--out` | two blind-spot scenarios as time series |
| `figure_new_families.py` | `--seed`, `--n` | `--out`, `--outfile` | one instance of each added family |
| `figure_approach_bars.py` | `--results` (`summary.csv` + `agg/`) | `--out` | the strongest configuration of each approach |
| `figure_threshold_summaries.py` | `--results` (`agg/roc_pr_auc.csv`) | `--out` | AUC-ROC and both precision-recall summaries |
| `figure_multiseed_intervals.py` | `--results` (`agg/metrics_with_ci.csv`) | `--out` | per-configuration accuracy with confidence intervals |
| `figure_prompt_comparison.py` | `--experiments` long-format frame | `--out`, `--outfile` | one rubric against another, per family |
| `figure_ablation_summary.py` | `--experiments` long-format frame | `--out`, `--outfile` | the same comparison in one panel |
| `render_figures.py` | `--results`, `--spec` | `--out` | accuracy bars, family heatmap, confusion matrices |

The three that take `--seed`/`--n` draw from the scenario generator rather than
from a results file, because what they illustrate is the dataset itself.

Figures render inside the `judge-service` image, which already carries matplotlib
for the `plot` representation, so the host virtualenv stays dependency-light:

```bash
docker run --rm -v "$PWD":/work -w /work canaryllm-judge-service \
    python tools/figure_approach_bars.py --results results --out results/figures
```

A host that has matplotlib can skip the container and run any generator directly.

Two generators still name the study's three hosted models in a display-label
constant (`figure_prompt_comparison.py`, `figure_ablation_summary.py`). Point them
at your own frame and edit that constant to your own aliases.

## The study and its data

Everything to do with the paper is in this one section. The rest of this README
is about the testbed.

This repository was built for a study of whether a language or vision model can
cover the Mann-Whitney judge's blind spots. [`CITATION.cff`](CITATION.cff) is how
to cite the testbed itself; the study is a separate work and carries its own
citation. [`PROVENANCE.md`](PROVENANCE.md) documents a defect found in the study's
run and what changed in the harness because of it.

### The data

The result artefacts are data for the paper, not code for this testbed, and they
are large: 8,550 verdicts across eight models in the multi-seed run alone. Cloning
this repository to judge your own canaries should not cost you a copy of somebody
else's DeepSeek-R1 verdicts, so they are archived separately and are **not** in
this repository.

> **Where to get the archive.** It is deposited when the article is published, and
> its DOI is added here at that point. It is not available yet, and this section
> is the place that will carry the link — nothing else in the repository holds a
> copy. A DOI is deliberately not guessed in advance: `CITATION.cff` leaves the
> same field blank for the same reason, because a wrong identifier is propagated
> automatically by every tool that reads these files.

| artefact | what it is |
|---|---|
| `reference-results/n20/`, `reference-results/n5/` | the two runs exactly as first published, including the rows in which a failed call was recorded as a verdict |
| `results/corrected/` | the 180-scenario run with both affected configurations re-measured under the fixed harness |
| `results/corrected-n5/` | the pilot with its failed calls dropped rather than re-measured |
| `results/n5-corrected/` | the five-seed run repeated under the fixed harness, written through the long-format schema |
| `results/agg/` | the derived tables: confidence intervals, McNemar, ROC/PR, ensemble and frontier sweeps |

Both the pre-correction and the post-correction artefacts are kept. The paper
reports a defect in the original, and that report is only checkable if the
original survives alongside the correction.

[`PROVENANCE.md`](PROVENANCE.md) is the short version: what the defect was, how it
was found, how many rows it touched, and what changed in the harness. It ships
with this repository because the harness behaviour it describes is the behaviour
you get.

The dataset is generated deterministically from a fixed master seed (`20260621`),
so the published runs can be regenerated on any machine with the same model tags
pulled — `make up && make experiment`. Two cautions before you read a zero from a
comparison:

- A model must be pulled at the **same tag**. `ollama pull <model>:latest` will
  overwrite a pinned tag and change the artefact behind the numbers.
- A reproduction check confirms determinism, not validity. If a model failed the
  same way on the same scenarios in both runs, both runs record the same failure.
  Read the `error_kind` column, not just the verdict.

## Command reference


| Command | What it does | Needs a model? |
|---|---|---|
| `make validate` | Regression gate: up, seed, and the three judges (statistical, dummy-AI, hybrid). | No |
| `make validate-ai` | Sweep over the models present locally; assert each returns a valid verdict. | Yes (the pulled ones) |
| `make seed-eval` | Seed the labelled nine-family dataset into VictoriaMetrics and write the manifest. `--probe` runs the genuine `NetflixACAJudge` per family. | No |
| `make experiment` | The full scored experiment: every judge × every scenario × every present model, into `results/`. Absent models are skipped. | Yes |
| `make experiment-quick` | Smoke run of the experiment (n=1, two default models). | Yes |
| `make scenario` | Run the blind-spot stress test through all judges, with Referee links. | Yes |
| `make demo-dummy` | The three judges over the dummy dataset, standalone. Prints Referee links. | No |
| `make results` | Table of the latest recorded Kayenta result per judge (verdict, score, model, Referee link). | No |
| `make referee [JUDGE=<key>] [EXEC=<id>]` | Open a recorded report in Referee. | No |
| `make test-judge-mock` | AI judge with the model stubbed; asserts a valid result (parsing and mapping). | No |
| `make up` / `make ensure-up` | Start the stack (force rebuild / build only what is missing). | No |
| `make seed` / `make logs` / `make ps` / `make down` / `make clean` | Seed dummy data / tail logs / status / stop / stop and wipe volumes. | No |

You can override the experiment size with environment variables: `EVAL_N` (instances
per family, default 12), `EVAL_SEED` (master seed), and `EVAL_MODELS` (a comma-separated
allow-list). The host tools run in a local `.venv` that `make` creates for you.

These are read from `.env`, and **the tools load `.env` themselves** (`tools/repo_env.py`,
imported by every entry point) rather than relying on the shell. Docker Compose reads that
file automatically but a bare `python tools/…` does not, and neither does
`nohup python tools/…`; a run launched that way silently used the built-in default of 12
instead of the configured 20 and produced a differently-seeded dataset while looking
entirely healthy. A variable already set in the real environment still wins, so
`EVAL_N=5 python tools/run_experiment.py` behaves as written. Every runner prints where
its configuration came from in its banner.

## Architecture


Seven Compose services share one Docker bridge network (`canary-net`) and address each
other by service name. Host ports are published so that you, and the host-run tools,
can reach them. Ollama runs on the host rather than in Compose, because the
Apple-Silicon Metal GPU is not reachable from inside a container; LiteLLM reaches it at
`host.docker.internal:11434`.

| Service | Image or build | Host port | Role |
|---|---|---|---|
| `redis` | `redis:7-alpine` | 6379 | Backs Kayenta's embedded Orca |
| `minio` (+ `minio-init`) | `minio/minio` | 9000 / 9001 | S3 object store (configs, results) |
| `victoriametrics` | `victoriametrics/victoria-metrics` | 8428 | Metric store (Prometheus API, back-dated writes) |
| `kayenta` | `armory/kayenta:2.36.9` | 8090 | The ACA engine (standalone, no Spinnaker) |
| `judge-service` | build `./judge-service` | 5001 | The configurable remote judge (FastAPI) |
| `litellm` | `ghcr.io/berriai/litellm` | 4000 | One OpenAI-compatible gateway for every provider |
| `referee` | build `./referee` | 3001 | Kayenta UI, built from source |
| Ollama (host, not Compose) | `brew install ollama` | 11434 | Local model runner; needed only for the AI modes |

```
                       ┌─────────────────────────── host ───────────────────────────┐
  you / browser ──▶ Referee UI :3001 ──proxy /kayenta/*──▶                            │
                                                          │                            │
  host tools (.venv):                                     ▼                            │
   seed_*_data.py / seed_eval_dataset.py ─▶ VictoriaMetrics :8428 ◀──fetch metrics──┐  │
   run_pipeline / run_scenario ──▶ Kayenta :8090 ───────────────────────────────┐  │  │
   run_experiment.py ──▶ Kayenta /judges/judge + judge-service /judge            │  │  │
                                                                                 │  │  │
   ┌───────────────────────────── canary-net (compose) ──────────────────────┐  │  │  │
   │  Kayenta ──Orca queue──▶ Redis :6379                                      │  │  │  │
   │  Kayenta ──configs/results (S3)──▶ MinIO :9000                            │◀─┘  │  │
   │  Kayenta ──PromQL──▶ VictoriaMetrics :8428 ──────────────────────────────┼─────┘  │
   │  Kayenta ──POST /judge──▶ judge-service :5000                             │        │
   │      judge-service ──(hybrid) callback /metricSetPairList,/canaryConfig,  │        │
   │                       /judges/judge──▶ Kayenta :8090                      │        │
   │      judge-service ──(AI modes) POST /v1/chat/completions──▶ litellm :4000│        │
   │  litellm ──OpenAI-compat──▶ Ollama (host) :11434  /  Bedrock·OpenAI·Anthropic ─────┘
   └──────────────────────────────────────────────────────────────────────────┘
```

> A note on the judge-service port. It is published on host **5001**, not 5000,
> because macOS Control Center (AirPlay Receiver) binds 5000. Kayenta reaches the
> service in-network at `judge-service:5000` regardless. Change `JUDGE_SERVICE_PORT`
> in `.env` if 5001 is taken too.

Useful URLs once the stack is up:

- Kayenta health: <http://localhost:8090/health> · Swagger: <http://localhost:8090/swagger-ui/index.html>
- VictoriaMetrics: <http://localhost:8428/>
- judge-service health: <http://localhost:5001/health>
- LiteLLM gateway: <http://localhost:4000/v1/models>
- Referee UI: <http://localhost:3001/>
- MinIO console: <http://localhost:9001/>

## The AI judge


The AI judge is a single code path for every model and provider. Two things are pure
configuration.

The **representation** (`mode`) decides how the metrics are shown to the model:

- `summary`: per-metric descriptive statistics (`n`, mean, median, p90/p95/p99,
  stddev, min, max, slope, and the experiment-minus-control deltas);
- `raw`: the value arrays themselves, as text;
- `plot`: a rendered matplotlib chart with control and experiment overlaid, attached as
  an image for a vision model. The text summary rides along too.

The **model** is a registry alias such as `qwen-llm` or `moondream-vlm`. Switching
models, or switching between text and vision, is a one-line config change.

### Flow

```
Kayenta ──POST /judge {canaryConfig, metricSetPairList, scoreThresholds}──▶ judge-service
   reads judge.judgeConfigurations.mode + model (env fallbacks JUDGE_MODE / JUDGE_MODEL)
   builds the representation (summary/raw text, or a matplotlib PNG for plot)
   one OpenAI-compatible call ──▶ litellm ──▶ Ollama (host) / Bedrock / OpenAI / Anthropic
   strict-JSON verdict ──▶ parsed and mapped to a valid CanaryJudgeResult
```

### Selecting the prompt

The rubric is a file, not a string literal. Each variant lives in `prompts/` under an
explicit identifier, and the identifier plus a hash of the prompt text is recorded on
every result row and in every `data/ai-logs/` payload — so no row is ever ambiguous
about which rubric produced it.

```bash
ls prompts/
# v1-frozen-2026-06.prompt  v2-operational-2026-07.prompt  v3-temporal-2026-07.prompt
```

| identifier | hash | what it is |
|---|---|---|
| **`v1-frozen-2026-06`** | `ae9455dfe48cb0f3` | **The default, and the rubric that produces every published number.** |
| `v2-operational-2026-07` | `db72944909fb423b` | v1 plus one sentence naming what the promotion decision is. |
| `v3-temporal-2026-07` | `66093d87318d11c5` | v1 plus two sentences making the temporal rule explicit. |

Selection is configuration, in the same order of precedence as `mode` and `model`: the
canary config's `judge.judgeConfigurations.prompt_id`, then `JUDGE_PROMPT_ID`, then the
frozen default.

```bash
JUDGE_PROMPT_ID=v3-temporal-2026-07 make experiment
python tools/run_frontier_experiment.py --model <alias> --prompt-id v2-operational-2026-07
```

**An unknown identifier fails the call.** It does not fall back to another prompt: a
silent fallback is how this study once spent 540 calls on a vision model that never
received an image.

**Adding one.** Copy a file, change `@id` to match the new filename stem, and edit the
`@rubric` body. `prompts/` is mounted into the container, so no rebuild is needed.

```
@id my-variant-2026-08
@description one line, shown in listings

# Lines beginning with '#' outside a body are commentary, not part of the prompt.

@system
...system prompt text, verbatim...

@rubric
...verdict instruction text, verbatim...
```

The hash covers the system and rubric text only, so editing the commentary does not
change a prompt's identity.

**Why v1 is frozen.** It was deliberately not tuned against the scenario outcomes;
tuning it would overfit to the statistical judge's known weaknesses and make the
comparison meaningless. It is one fixed template for every model and every
representation. A system message sets the role; the user message is the representation
plus a strict-JSON verdict instruction that asks the model to judge holistically, to
weigh variance and instability, the tail (p90/p95/p99/max), and emerging trends
(slope), not just the mean and median. The ground-truth label never appears in the
prompt, control is always presented before experiment (to control for position bias),
and the rationale is length-bounded (to control for verbosity bias).

Editing `prompts/v1-frozen-2026-06.prompt` invalidates every number produced under
it, including the published ones. Add a new file instead; the registry is
versioned so that a rubric change is always visible in the `prompt_id` and
`prompt_hash` columns of the results.

### Determinism and robustness

Calls use `temperature=0`, fixed `top_p` and `max_tokens`, a `seed` where the backend
supports one, and structured JSON output. The parser is built to tolerate what real
models actually emit: it strips reasoning `<think>…</think>` preambles (deepseek-r1
does this) and markdown code fences, extracts the first balanced JSON object, retries
once, and on a final failure returns a valid `CanaryJudgeResult` with an explicit
error classification. No run ever crashes the analysis or returns a raw error. The
experiment includes a determinism preflight that runs one scenario twice and asserts
an identical verdict. Every AI call is logged to `data/ai-logs/`: the prompt, the chart
PNG and its sha256, the raw response, the parsed verdict, and the resolved model.

### Models evaluated

The evaluated set is open-weight (Apache-2.0, MIT, or fully open) and served locally by
Ollama. Aliases are registered in `judge-service/models.yaml` (for modality) and
`litellm/config.yaml` (for provider and tag).

| Alias | Model | Modality | License |
|---|---|---|---|
| `qwen-llm` | Qwen2.5 7B Instruct | text | Apache-2.0 |
| `phi4-llm` | Phi-4 14B | text | MIT |
| `olmo2-llm` | OLMo 2 7B | text | Apache-2.0 (fully open) |
| `mistral-nemo-llm` | Mistral-Nemo 12B | text | Apache-2.0 |
| `deepseek-r1-llm` | DeepSeek-R1-Distill-Qwen 7B | text | MIT |
| `moondream-vlm` | Moondream2 (~1.8B) | vision | Apache-2.0 |
| `granite-vision-vlm` | Granite 3.2 Vision 2B | vision | Apache-2.0 |
| `minicpm-v-vlm` | MiniCPM-V 8B | vision | (vision) |

Any alias whose Ollama tag is not pulled is skipped automatically. Text models use
LiteLLM's `ollama_chat/` provider; vision models use `ollama/`, which converts the
OpenAI `image_url` parts into Ollama's `images` field.

## The hybrid judge


The hybrid is not a reimplementation of either judge. When Kayenta calls
`judge-service` for a `mode=hybrid` analysis, the service:

1. computes its own AI verdict on the received pairs (the no-op dummy by default, or a
   real model if `aiMode=summary|raw|plot` is set);
2. calls back into Kayenta (`POST /metricSetPairList`, `POST /canaryConfig` with
   `judge.name=NetflixACAJudge-v1.0`, then `POST /judges/judge`) to run the genuine
   statistical judge on the same already-fetched pairs. No metric is re-fetched and
   there is no recursion, because the temporary config names an in-process judge;
3. merges the two verdicts under a selectable policy and returns one
   `CanaryJudgeResult`. The hybrid is itself a normal Kayenta analysis with its own
   execution id, so it renders in Referee like any other.

The merge policy lives in `judge-service/app/hybrid_policy.py` as a pure
`decide_policy()` function, and the experiment runner imports the same function, so the
scored hybrid is identical to what the service returns. Select a policy with
`judgeConfigurations.hybridPolicy`:

- `gated` (the default, and the recommended one): trust statistical failures, since
  they are high precision. On a clean statistical pass, escalate to the AI only when it
  fails with high confidence or cites a blind-spot dimension (variance, tail, or trend)
  in its reasoning. In the statistical judge's marginal, near-threshold regime, where
  it is least reliable, defer to the AI. This is the policy that delivers
  statistical-like precision with AI-like recall on the blind spots.
- `or`: fail if either judge fails. A safety net with high recall, but it inherits the
  AI's benign-family false positives.
- `and`: fail only if both judges fail. Strict, high precision, low recall.

The policy parameters (the marginal band around the threshold, the AI fail-confidence
cutoff, and the blind-spot keyword set) have documented defaults in `hybrid_policy.py`.
The returned score is always set so that its threshold mapping agrees with the verdict,
because Kayenta re-derives promote-versus-roll-back from the score.

## The evaluation dataset


`tools/eval_dataset.py` generates, deterministically from one master seed, N instances
per family. Each instance is a control-and-experiment pair of series for one to three
metrics over a 30-minute window at 60-second resolution, written to VictoriaMetrics
with the framework's `Canary="Control"` / `Canary="Experiment"` scheme. Instances are
isolated by namespacing the metric names with an opaque global id (for example
`http_request_duration_seconds_g0007`), never the family name, so the ground truth
never leaks into what the model sees. `data/eval_manifest.json` records, per scenario:
id, family, seed, parameters, ground-truth verdict, and per-metric truth.
`data/eval-configs/` holds the generated per-scenario configs.

| Family | Truth | What it probes | Statistical behaviour |
|---|---|---|---|
| `no_change` | PASS | baseline ≈ experiment, mild noise | correct PASS |
| `noise_equivalent` | PASS | both high-variance, same distribution | correct PASS (no location shift) |
| `healed_transient` | PASS | degraded early, then recovers | correct PASS (median robust to a transient) |
| `clean_mean_shift` | FAIL | median clearly worse | correct FAIL (its home turf) |
| `variance_increase` | FAIL | equal median, much higher variance | blind, false PASS |
| `tail_regression` | FAIL | mean/median flat, p95/p99 worse | blind, false PASS |
| `gradual_drift` | FAIL | late, progressive ramp worse | blind, false PASS (order-insensitive) |
| `cross_metric_marginal` | FAIL | several metrics each in tolerance, jointly degraded | blind, false PASS (group dilution) |
| `subtle_regression` | FAIL | small real shift near tolerance | correct FAIL (AI false-negative risk) |

The "blind" rows were confirmed against the genuine `NetflixACAJudge` by the
`make seed-eval --probe` step. The dataset deliberately includes genuine PASS families
so the FAIL numbers mean something: a judge that always says FAIL is penalised on
`no_change`, `noise_equivalent`, and `healed_transient`.

## Configuring the statistical judge fairly


For the comparison to be honest, the statistical judge has to run at its best, so any
shortfall is a real structural blind spot and not a misconfiguration. One consistent,
best-practice config is applied to every metric in every scenario, with no
per-scenario hand-tuning. Each choice was checked against the Kayenta source
(`NetflixACAJudge.scala`, `MannWhitneyClassifier.scala`, `EffectSizes.scala`,
`WeightedSumScorer.scala`).

| Setting | Value | Why |
|---|---|---|
| `direction` | `increase` | every metric here is "higher is worse" |
| `effectSize.measure` | `meanRatio` | Kayenta default; thresholds are `experimentMean / controlMean` |
| `effectSize.allowedIncrease` | `1.05` | a sensitive 5% gate; best recall on clean shifts without flagging benign noise |
| `effectSize.criticalIncrease` | `1.25` | a regression over 25% is a hard failure |
| `nanStrategy` | `remove` | drop missing samples cleanly |
| `outliers.strategy` | `keep` | deliberately not IQR-trimming; removing outliers would discard the tail and variance spikes and manufacture the blind spot |
| `critical` / `mustHaveData` | `true` / `true` | best-practice promotion gating (verdict-neutral on this dataset, but the honest default) |
| `scoreThresholds` | `pass 75 / marginal 50` | promote if score ≥ 75 |
| `groupWeights` | even split over present groups | structural (weights must sum to 100), not a tuning knob |

The key detail from the source: a metric is flagged `High` only when the rank test finds
a significant location shift and the mean ratio exceeds `allowedIncrease`, an AND of two
conditions. That is why `variance_increase`, `tail_regression`, `gradual_drift`, and
`cross_metric_marginal` slip through even with a sensitive gate and outliers kept.
There is no location shift for the test to find.

## Running the experiment and reading `results/`


```bash
make seed-eval                 # load the dataset into VM + manifest (+ probe the real judge per family)
make experiment-quick          # smoke run: n=1, two default models
make experiment                # the full run: every judge × scenario × present model
EVAL_N=20 make experiment      # override instances per family (also EVAL_SEED, EVAL_MODELS)
```

A run writes to a `results/` directory at the repo root, which is gitignored, so
your own runs stay yours. `example-results/` holds a small committed run with the
same file names, which is what `make demo` reads.

For each scenario the runner feeds the genuine `NetflixACAJudge` (through Kayenta's
`/judges/judge`) and the AI judge the identical in-memory metric pairs, then derives the
three hybrid policies from those results using the shared policy code. Every judge sees
the same data, so the comparison is fair. FAIL is the positive class. Output goes to
`results/`:

- `results_raw.csv`: one row per scenario × judge × model (truth, verdict, score,
  correct, latency).
- `summary.csv` / `summary.md`: the scoreboard, judge × model, with accuracy,
  precision, recall, F1, FPR, and FNR.
- `family_matrix.csv` and `figures/family_heatmap.png`: judge × family accuracy.
- `mcnemar.csv`: McNemar's exact paired test for the key judge pairs.
- `kappa.csv`: Cohen's kappa against ground truth and between judges.
- `figures/`: `accuracy_bars.png`, `family_heatmap.png`, `confusion_matrices.png`.
- `results.md`: the stitched report with an auto-generated summary (best judge per
  family, best overall, the hybrid's balance).

Absent models are recorded and skipped, not failed. The figures are rendered inside the
`judge-service` container, which already has matplotlib for the `plot` representation;
the host Python environment is kept dependency-light on purpose.

A few extra analysis scripts under `tools/` go beyond the headline run: multi-seed
confidence intervals (`run_multiseed_corrected.py`, `ensemble_multiseed_ci.py`,
`bounded_confidence_intervals.py`), a statistical-ensemble judge and a false-positive
guard (`run_ensemble_experiment.py`, `run_fp_guard_experiment.py`), a frontier run
against a larger cloud model (`run_frontier_experiment.py`), ROC/PR curves
(`roc_pr_curves.py`), baseline classifiers and MCC (`baselines_and_mcc.py`), pooled
McNemar (`pooled_mcnemar.py`), and test-retest reproducibility
(`reproducibility_test_retest.py`). Their outputs land in `results/agg/`.

### The long-format result schema

The per-experiment CSVs above each have their own columns, which makes comparing two
experiments a parsing exercise. Every run therefore *also* appends to a single
long-format frame under `results/agg/experiments/`, one row per judged scenario, same
columns for every judge and every run:

| column | meaning |
|---|---|
| `run_id`, `ts` | which invocation wrote the row, and when |
| `dataset_id` | `original-180` or `generalisation-60` |
| `prompt_id`, `prompt_hash` | the rubric that produced it; **empty** for judges that make no model call |
| `model`, `judge`, `representation` | `judge` is `statistical`/`ai`/`hybrid`/`ensemble`; `representation` is `summary`/`raw`/`plot`, or the hybrid policy |
| `family`, `scenario_id`, `seed`, `truth_label` | which scenario, and its label |
| `verdict`, `score`, `correct` | the judgement; all three **empty on an error row** |
| `latency_s`, `tokens_in`, `tokens_out`, `finish_reason` | cost and completion telemetry |
| `error_kind`, `error` | `empty_completion`, `parse_failure`, `api_error`, `transport`, `unknown_prompt` |
| `rationale` | the model's own explanation |

Read it through `tools/results_schema.py`, which is the only loader:

```python
import results_schema as rs

rows = rs.load(prompt_id="v1-frozen-2026-06", dataset_id="original-180", judge="ai")
rs.confusion(rows)          # {'TP': .., 'FP': .., 'TN': .., 'FN': ..}
print(rs.available())       # what run_ids / prompts / models are in the frame
```

**`load()` excludes error rows unless you ask for them.** A row with an `error_kind`
carries no judgement, and letting one into an accuracy denominator is how a failed call
becomes a published number: an empty completion used to be recorded as `FAIL` at score
0.0 with an empty error field, which is indistinguishable from a verdict of FAIL. Rows
are append-only; nothing rewrites an existing file.

## The stress test


For a fast visual demonstration, separate from the scored experiment, `make scenario`
seeds two realistic Prometheus metrics: an unstable-tail latency
(`http_request_duration_seconds`, same median as control but roughly 20× the variance)
and an emerging memory leak (`process_resident_memory_bytes`, flat and then a late
ramp). It runs all judges through Kayenta and prints a contrast table and Referee
deep-links. The statistical judge passes the clearly-bad canary; the AI judges catch
it. It is the same blind spot the scored experiment measures across many seeded
instances.

## Reading a single result


Every judge returns the same `CanaryJudgeResult`:

- Verdict, PASS or FAIL: Kayenta's `didPassThresholds`, the overall score against the
  configured thresholds. PASS means promote, FAIL means roll back.
- Score, 0–100, higher is healthier. By default, promote if the score is at least 75.
- Per-metric classification: `Pass` (comparable), `High` (worse), `Low` (lower than
  control), `Nodata`, or `Error`.
- The reported judge and model, for example `NetflixACAJudge-v1.0`,
  `RemoteJudge-v1.0 (ai:summary:qwen-llm)`, or
  `RemoteJudge-v1.0 (hybrid:gated: NetflixACAJudge + ai:summary:qwen-llm)`. This is how
  you confirm which judge and model produced a result.

> Small local models sometimes get a per-metric direction wrong even when the overall
> verdict is right. The experiment scores on the overall verdict and still records
> per-metric correctness. A larger model tightens the per-metric labels.

## Viewing a result in Referee

Referee is the browser view, and it is the manual inspection path: it draws the
per-metric control-versus-experiment graphs the judge saw, so you can look at
what your judge is judging rather than only at the verdict it returned. When a
verdict surprises you, this is where you find out why.

Every analysis is a real Kayenta execution, so they all render in Referee's SCAPE
Report Viewer and can be compared side by side:

```bash
make demo-dummy            # or: make scenario; both record execution ids
make referee               # opens the default report
make referee JUDGE=hybrid  # a specific recorded report (keys come from `make results`)
make referee EXEC=<id>     # any execution id
```

Referee serves its SPA under `/dashboard` and reverse-proxies Kayenta under
`/kayenta/*`. You can also drive analyses by hand through Retrospective Analysis (metric
source Prometheus, account `vm`, control scope `Control`, experiment scope `Experiment`).

## Configuration reference


Nothing you would want to change is hardcoded; it lives in a handful of files.

### `.env` (and `.env.example`)

Compose reads `.env` automatically. It holds pinned image tags (never `latest`), the
MinIO/object-store credentials, host ports, the judge defaults (`JUDGE_MODE`, default
`dummy`; `JUDGE_MODEL`; `JUDGE_TEMPERATURE` 0; `JUDGE_TOP_P` 1.0; `JUDGE_MAX_TOKENS`
1024; `JUDGE_SEED` 42), the local-model wiring (`OLLAMA_BASE_URL`), and optional cloud
credentials. `.env` is gitignored; copy `.env.example` to start.

### `kayenta/config/kayenta.yml`

Mounted into Kayenta, this replaces the image default entirely. The notable keys: the
`minio-store` object-store account (`kayenta.aws` plus `kayenta.s3.enabled: true`); the
`vm` Prometheus metric account (`kayenta.prometheus`); `kayenta.remoteJudge.enabled:
true` with `endpoint.baseUrl: http://judge-service:5000` (camelCase); and
`kayenta.standaloneCanaryAnalysis.enabled: true`, which is off by default upstream.
After editing, `docker compose up -d kayenta`.

### Canary configs (`kayenta/canary-configs/*.json`)

A canary config tells Kayenta what to query, how to judge, and how to score:

```jsonc
{
  "name": "example",
  "judge": { "name": "NetflixACAJudge-v1.0", "judgeConfigurations": {} },
  "metrics": [{
    "name": "http_request_duration_seconds",
    "query": {
      "type": "prometheus", "serviceType": "prometheus",
      "metricName": "http_request_duration_seconds",
      // `sum without(Canary)` strips the discriminator label so control and experiment
      // pair into one Control/Experiment pair. Without it Kayenta sees two series and
      // returns Nodata.
      "customInlineTemplate": "PromQL:sum without(Canary) (http_request_duration_seconds{Canary=\"${scope}\"})"
    },
    "groups": ["latency"],
    "analysisConfigurations": { "canary": {
      "direction": "increase",
      "nanStrategy": "remove",
      "effectSize": { "measure": "meanRatio", "allowedIncrease": 1.05, "criticalIncrease": 1.25 },
      "outliers": { "strategy": "keep" },
      "critical": true, "mustHaveData": true
    }},
    "scopeName": "default"
  }],
  "classifier": {
    "groupWeights": { "latency": 100 },
    "scoreThresholds": { "pass": 75, "marginal": 50 }
  }
}
```

`judgeConfigurations` holds the per-analysis judge knobs, read by `judge-service` with
`.env` fallbacks:

- `mode`: `dummy` (no-op, default), `summary`, `raw`, `plot`, or `hybrid`.
- `model`: a registry alias from `models.yaml` (for the AI modes).
- `aiMode`: for `mode: hybrid`, one of `summary`/`raw`/`plot` to make the hybrid's AI
  half a real model (default: dummy).
- `hybridPolicy`: for `mode: hybrid`, `gated` (default), `or`, or `and`.

### `judge-service/models.yaml` and `litellm/config.yaml`

`models.yaml` maps each alias to its modality (`text` or `vision`), the only thing the
judge branches on. `litellm/config.yaml` maps each alias to a provider, model, and
credentials. Alias names have to match across the two files. Reload the gateway with
`docker compose up -d litellm`.

## Build notes and operational constraints


### Images

- Pulled and pinned, no build: `redis`, `minio`, `minio/mc`, `victoriametrics`,
  `armory/kayenta`, `litellm`. Tags live in `.env`.
- `judge-service` (`python:3.11-slim`): FastAPI, uvicorn, pydantic, requests, openai,
  httpx 0.27.2 (pinned), PyYAML, matplotlib, numpy. Rebuild with `make up` or
  `make build`.
- `referee`: builds the archived `Nike-Inc/referee` from source on `node:10`,
  downloading the architecture-matching `tini` so it runs natively on arm64 and amd64.

### Config confirmed against the Kayenta source

Where a common assumption differs from the source, the source wins:

- The remote-judge config key is `kayenta.remoteJudge.*` (camelCase). The service is
  POSTed at `<baseUrl>/judge` with `RemoteJudgeRequest = {canaryConfig, scoreThresholds,
  metricSetPairList}` and returns a `CanaryJudgeResult`.
- `kayenta.standaloneCanaryAnalysis.enabled` has to be `true` (it is off by default) to
  expose `POST /standalone_canary_analysis/`.
- The hybrid reuses the real judge through
  `POST /judges/judge?canaryConfigId&metricSetPairListId&passThreshold&marginalThreshold`
  on a stored `metricSetPairList`, with no re-fetch.
- The analysis scope shape is flat: `controlScope` and `experimentScope` are strings
  with sibling `startTimeIso` / `endTimeIso` / `step`.
- `effectSize` thresholds are mean ratios under the default `meanRatio` measure. A
  metric is `High` only on a significant location shift and a mean ratio over
  `allowedIncrease`.

### Runtime requirements

- The PromQL template has to strip the `Canary` label (`sum without(Canary) (...)`) or
  control and experiment do not pair, and every metric comes back `Nodata`.
- The remote-judge response has to nest `stats.count` in each metric's `controlMetadata`
  and `experimentMetadata`, or the standalone aggregation throws an NPE.
- Ollama has to bind `0.0.0.0` so the `litellm` container can reach it through
  `host.docker.internal`.
- Vision needs LiteLLM's `ollama/` provider, not `ollama_chat/`. The latter passes the
  OpenAI `content` array straight through and Ollama rejects it.
- `openai==1.51` is incompatible with `httpx>=0.28` (the removed `proxies` argument), so
  `httpx==0.27.2` is pinned.
- `models.yaml` has to be COPYed into the judge image, or the modality lookup silently
  falls back to `text` and `plot` mode sends no image.
- Referee's `tini` has to match the image architecture. The generic release asset is
  amd64-only and triggers a `rosetta error` on Apple Silicon.

## Repository layout

```
canaryLLM/
|- docker-compose.yml         # the 7-service stack (+ host.docker.internal for Ollama)
|- .env / .env.example        # pinned image tags, ports, creds, judge defaults, model tags
|- Makefile                   # every workflow (validate, demo, experiment, scenario, ...)
|- PROVENANCE.md              # the fabricated-verdict defect, and what changed because of it
|- kayenta/
|  |- config/kayenta.yml      # Kayenta config (mounted into the container)
|  \- canary-configs/         # one JSON per judge/representation
|- judge-service/             # the configurable remote judge (FastAPI)
|  |- models.yaml             # alias -> modality registry (text/vision)
|  |- sample_request.json     # a captured RemoteJudgeRequest (validate-ai / mock)
|  |- tests/test_mock.py      # model-stubbed unit test
|  \- app/
|     |- main.py              # /judge + /health; dispatches dummy|ai|hybrid by config
|     |- judge_config.py      # resolve mode/model from judgeConfigurations + env
|     |- models.py            # pydantic mirrors of the Kayenta request/response
|     |- dummy_judge.py       # the no-op stub verdict (model-free default)
|     |- representations.py   # summary / raw / plot builders (matplotlib for plot)
|     |- llm_judge.py         # the AI judge: prompt -> litellm -> parse -> map
|     |- prompt_registry.py   # loads prompts/ by id; unknown ids fail, never fall back
|     |- verdict_parser.py    # the tolerant JSON parser, pure so it can be replayed offline
|     |- hybrid.py            # hybrid mode: real statistical + AI, merged by a policy
|     |- hybrid_policy.py     # the pure merge policy (gated|or|and), shared with the runner
|     |- series_guards.py     # shape detectors (healed transient, equal-variance noise)
|     |- ensemble_judge.py    # the statistical-ensemble judge
|     \- kayenta_callback.py  # hybrid: call Kayenta back for the genuine default judge
|- prompts/                   # one rubric per file; v1-frozen-2026-06 is the default
|- litellm/config.yaml        # gateway: alias -> provider/model (local + cloud examples)
|- referee/Dockerfile         # builds Nike-Inc/referee from source
|- tools/                     # host-run Python (in .venv)
|  |- eval_dataset.py         # the seeded datasets + config template
|  |- repo_env.py             # loads .env so a bare `python tools/...` matches `make`
|  |- results_schema.py       # the long-format result schema and its only loader
|  |- seed_eval_dataset.py    # load the dataset into VM + manifest (make seed-eval)
|  |- judge_clients.py        # host clients: genuine NetflixACAJudge + AI judge on exact pairs
|  |- run_experiment.py       # the scored experiment (make experiment) -> results/
|  |- run_multiseed_corrected.py  # the multi-seed sweep, written through the schema
|  |- test_failed_call_is_not_scored.py  # the regression test for the defect above
|  |- figure_*.py             # the figure generators, each parameterised by input path
|  |- render_figures.py       # matplotlib figures, run inside the judge-service container
|  \- ...                     # CIs, ensemble, frontier, ROC, McNemar, replay
|- example-results/           # a small committed run, so `make demo` works from a clone
\- data/                      # last_run.json, ai-logs/, eval_manifest.json, eval-configs/
```

`results/` is where your own runs land, and it is gitignored. Nothing in it is
published; regenerate it with `make experiment`.

## Scope and limitations


What the harness produces is the numbers, tables, and figures for the judge comparison,
in `results/`.

Some limits are there by design:

- The evaluated models are laptop-scale, roughly 2–14B parameters. Larger or cloud
  models reduce the AI's benign-family false positives and tighten its per-metric
  labels.
- The dataset is synthetic: parametric and seeded. It is built to embody the known
  statistical blind spots realistically, not to match the prompt. The prompt is frozen
  and never tuned against it.
- Small vision models occasionally misread dense or precise charts
  (`cross_metric_marginal`, `subtle_regression`). The visually salient families
  (variance, tail, drift) suit them better.

### A defect that shaped this harness

An earlier version of the judge turned any unusable model response into a
structurally valid result carrying `verdict = FAIL` at score 0.0 with an empty
error field, which downstream is indistinguishable from a genuine failing
judgement. In the published run 22 rows were failed calls recorded as verdicts,
and because the affected families are FAIL-truth, 21 of them scored as *correct*
and flattered the configurations they damaged.

The harness no longer does this. A call that produces no usable answer emits an
error row carrying an `error_kind` and no verdict, and `results_schema.load()`
excludes such rows from every denominator.
`tools/test_failed_call_is_not_scored.py` checks it on this code rather than
asserting it, by stubbing the AI judge to fail in each of the five ways the
harness can observe.

[`PROVENANCE.md`](PROVENANCE.md) has the full account: what the defect was, how it
was found, how many rows it touched in each artefact, and which published results
are pre- and post-correction. `make replay-logs` re-derives the evidence from the
archived payloads.

If you want to extend it, the seams are all inside the service:

1. `judge-service/app/llm_judge.py`: `judge_ai()`, the frozen prompt, and the mapping.
2. `judge-service/app/representations.py`: the `summary`/`raw`/`plot` builders.
3. `judge-service/app/hybrid_policy.py`: `decide_policy()`, the merge.
4. `judge-service/app/dummy_judge.py`: the no-op fallback, kept as the default.

### Platform notes

- `armory/kayenta`, `redis`, `minio`, `victoriametrics`, and `litellm` resolve on both
  arm64 and amd64. Referee builds and runs natively on arm64 (tested on Apple Silicon);
  if a future toolchain change breaks the build, add `platform: linux/amd64` to the
  `referee` service.
- Ollama runs on the host because the Metal GPU is not reachable from Docker on macOS.
  Everything else runs in Compose on the `canary-net` bridge.
- A clean `make clean && make validate` brings the stack up and passes the model-free
  regression with every service healthy.

## License


Released under the Apache License 2.0; see [`LICENSE`](LICENSE). The evaluated
models are used under their own licenses (Apache-2.0 or MIT; each is named with
its licence in `judge-service/models.yaml`). Kayenta, Referee, and the other
components keep their upstream licenses.

For how to cite this testbed, see [`CITATION.cff`](CITATION.cff). For the study it
was built for and where its data lives, see
[the study section](#the-study-and-its-data).
