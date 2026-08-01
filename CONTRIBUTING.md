# Contributing

Thanks for taking a look. This is a research harness, so most useful contributions are
one of: a new judge representation, a new hybrid policy, another evaluation family, a
bug fix in the pipeline, or a port to another metric store or provider.

## Getting set up

You need Docker (with Compose v2) and Python 3.11 on the host. Ollama is only needed if
you want to run the real AI judges.

```bash
cp .env.example .env
make validate        # brings the stack up and runs the model-free regression
```

`make validate` is the gate to pass before and after any change. It needs no model. If
it goes green, the stack builds and the three judges (statistical, dummy-AI, hybrid)
all return a verdict.

The host tools run in a `.venv` that `make` creates from `tools/requirements.txt`. You
do not need to manage it by hand.

## Running the tests

```bash
make test-judge-mock   # AI judge with the model stubbed; checks parsing and mapping
make validate          # full model-free end-to-end regression
```

If you have models pulled through Ollama:

```bash
make validate-ai       # sweeps the models present locally
make experiment-quick  # a small scored run
```

## Where things live

- `judge-service/app/` is the service. The AI judge (`llm_judge.py`), the
  representations (`representations.py`), the hybrid policy (`hybrid_policy.py`), and the
  dummy fallback (`dummy_judge.py`) are the parts you are most likely to touch.
- `prompts/` holds the rubrics, one file per variant.
- `tools/` is the host-run experiment and seeding code. `results_schema.py` defines the
  long-format result row and is the only thing that reads it — figure generators select
  through it rather than parsing raw files.
- `kayenta/canary-configs/` holds one JSON per judge and representation.

## Checking you have not changed the numbers

Most changes here are meant to be behaviour-preserving under the default configuration.
Two checks prove it, and neither needs a cloud credential:

```bash
make venv                                             # once: the host tools need `requests`
make up                                               # test-judge-mock runs inside the service

make replay-logs                                      # re-parse every archived response; verdicts and scores must not move
.venv/bin/python tools/test_failed_call_is_not_scored.py  # a failed call must still produce no verdict
make test-judge-mock                                  # representation, parse and mapping, model stubbed
make demo                                             # the committed example data must still render
```

Run the test through `.venv/bin/python`, not a bare `python`: it imports
`judge_clients`, which imports `requests`, which lives in the virtualenv.

`replay-logs` is the sensitive one. It re-runs the tolerant parser over every archived
model response and compares the verdict, the score and the rationale string. A verdict
is one of two values and a score one of a hundred, so both can coincide across a whole
run by chance; a sixty-word rationale reproducing character for character cannot.

The row-by-row check against the published run lived here too, as `make verify-repro`.
It compared a fresh run against `reference-results/`, which is archived with the paper's
data rather than shipped with the testbed, so both the target and the script moved with
it.

The README's [scope section](README.md#scope-and-limitations) lists the four main
extension seams and where each one is.

## A few conventions

- The rubric lives in `prompts/`, one file per variant, and
  `prompts/v1-frozen-2026-06.prompt` is the one behind every published number. **Add a
  file; do not edit that one.** Selection is by id (`JUDGE_PROMPT_ID`, or
  `judge.judgeConfigurations.prompt_id`), the id and a hash of the text are recorded on
  every result row, and an unknown id fails the call rather than falling back. Run
  `make prompts` to list them.
- Anything that reads a setting from the environment must import `tools/repo_env.py`
  first, so a bare `python tools/…` loads `.env` the way `make` does. Three entry points
  have silently run with the wrong `EVAL_N` because they did not.
- A judge that could not produce a judgement must emit an **error row**, never a verdict.
  `judgeMetadata.ok` carries this out of the service and `results_schema.load()` drops
  such rows by default. An empty completion recorded as `FAIL` at score 0.0 is
  indistinguishable from a real failing verdict, and that has already put fabricated rows
  into published results.
- `hybrid_policy.py` is kept pure and stdlib-only, because both the service and the
  experiment runner import it. Do not add framework dependencies to it.
- The dummy judge is the model-free default and the regression path. Keep it working
  without a model.
- Pin every tag. Container images are pinned in `.env`, and every Ollama model in
  `litellm/config.yaml` is named by a version tag, never a floating one. If you add a
  model, use a version tag and record its manifest digest in `.env.example` beside the
  others, so a reader can check what they pulled against what was scored.
- If you add a model, register its alias in both `judge-service/models.yaml` (modality)
  and `litellm/config.yaml` (provider), and use an open-weight, permissively-licensed
  model if you want it in the default evaluated set.

## Opening a change

Run `make validate` first. In the pull request, say what you changed, how you tested it,
and, if you touched anything that affects scoring (a representation, a policy, the
prompt, the dataset), include a before/after from `results/summary.md` so the effect is
visible.
