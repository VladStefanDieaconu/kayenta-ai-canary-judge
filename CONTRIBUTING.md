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
- `tools/` is the host-run experiment and seeding code.
- `kayenta/canary-configs/` holds one JSON per judge and representation.

The README's [scope section](README.md#scope-and-limitations) lists the four main
extension seams and where each one is.

## A few conventions

- The judge prompt is frozen (`PROMPT_VERSION` in `llm_judge.py`). If you change the
  prompt text, bump the version and re-run the full sweep, otherwise the published
  numbers no longer describe the code.
- `hybrid_policy.py` is kept pure and stdlib-only, because both the service and the
  experiment runner import it. Do not add framework dependencies to it.
- The dummy judge is the model-free default and the regression path. Keep it working
  without a model.
- Keep image tags pinned in `.env`; no `:latest`.
- If you add a model, register its alias in both `judge-service/models.yaml` (modality)
  and `litellm/config.yaml` (provider), and use an open-weight, permissively-licensed
  model if you want it in the default evaluated set.

## Opening a change

Run `make validate` first. In the pull request, say what you changed, how you tested it,
and, if you touched anything that affects scoring (a representation, a policy, the
prompt, the dataset), include a before/after from `results/summary.md` so the effect is
visible.
