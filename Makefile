# kayenta-ai-canary-judge: convenience targets.
# Host tools (seeder + pipeline) run in a local venv created from tools/requirements.txt.

SHELL := /bin/bash
COMPOSE := docker compose
VENV := .venv
PY := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

# Captured before the -include below, which would otherwise append .env to
# MAKEFILE_LIST and make `help` scan it too.
THIS_MAKEFILE := $(lastword $(MAKEFILE_LIST))

# Read .env so the host tools see the same ports Compose publishes. Without
# this the tools fall back to their hardcoded defaults and a changed *_PORT
# moves the container while every host tool keeps talking to the old port.
# `-include` so a fresh clone with no .env still works off the defaults below.
-include .env
export

KAYENTA_PORT ?= 8090
REFEREE_PORT ?= 3001
VICTORIAMETRICS_PORT ?= 8428
KAYENTA_URL ?= http://localhost:$(KAYENTA_PORT)
VM_URL ?= http://localhost:$(VICTORIAMETRICS_PORT)
REFEREE_URL ?= http://localhost:$(REFEREE_PORT)

.DEFAULT_GOAL := help

.PHONY: help up ensure-up down logs ps build venv wait-kayenta \
        seed seed-scenario seed-eval demo-dummy pipeline judge scenario results referee \
        validate validate-ai test-judge-mock experiment experiment-quick clean \
        analysis analysis-live figures demo require-results require-stack

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(THIS_MAKEFILE) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

up: ## Start the stack, forcing a rebuild of local images (use after code changes)
	$(COMPOSE) up -d --build

ensure-up: ## Start the stack without forcing a rebuild (builds only missing images)
	$(COMPOSE) up -d

down: ## Stop the stack (keep volumes)
	$(COMPOSE) down

logs: ## Tail logs for all services
	$(COMPOSE) logs -f

ps: ## Show service status
	$(COMPOSE) ps

build: ## Build the locally-built images (judge-service, referee)
	$(COMPOSE) build

venv: $(VENV)/.installed ## Create the host Python venv for the tools

# --system-site-packages lets the venv inherit a host-installed `requests`, so the
# tools work on a machine where the package index is gated or offline but the
# dependency is already present system-wide.
#
# The install itself is allowed to fail (the `-` prefix) for exactly that case.
# What must not happen is failing quietly: a venv with no `requests` looks ready
# and then every tool dies with "No module named 'requests'" somewhere further in,
# which is a much worse error than the one that caused it. So the import is
# checked, and the target refuses to mark itself installed if it is not there.
$(VENV)/.installed: tools/requirements.txt
	python3 -m venv --system-site-packages $(VENV)
	-$(PIP) install --no-deps -r tools/requirements.txt
	@$(PY) -c "import requests" 2>/dev/null || { \
	  echo ""; \
	  echo "ERROR: the host tools need 'requests' and it is not importable."; \
	  echo ""; \
	  echo "  The install step above could not reach a package index, and this"; \
	  echo "  machine has no system-wide 'requests' for the venv to inherit."; \
	  echo ""; \
	  echo "  Fix it either way:"; \
	  echo "    $(PIP) install requests==2.32.3     # if you can reach an index"; \
	  echo "    pip3 install --user requests==2.32.3 # then re-run: make venv"; \
	  echo ""; \
	  exit 1; }
	@touch $(VENV)/.installed

wait-kayenta: ## Poll Kayenta /health until UP (timeout ~4 min)
	@echo "Waiting for Kayenta health at :$(KAYENTA_PORT) ..."
	@for i in $$(seq 1 48); do \
		if curl -fsS http://localhost:$(KAYENTA_PORT)/health >/dev/null 2>&1; then \
			status=$$(curl -fsS http://localhost:$(KAYENTA_PORT)/health | tr -d ' '); \
			case "$$status" in *'"status":"UP"'*) echo "Kayenta is UP"; exit 0;; esac; \
		fi; \
		sleep 5; \
	done; \
	echo "Kayenta did not become healthy in time" >&2; exit 1

seed: venv ## Seed dummy Canary=Control/Experiment data into VictoriaMetrics
	$(PY) tools/seed_dummy_data.py

seed-scenario: venv ## Seed the Mann-Whitney blind-spot scenario dataset
	$(PY) tools/seed_scenario_data.py

# Run the Mann-Whitney blind-spot scenario across all judges (default vs AI) and
# print a contrast table + Referee links. Demonstrates the default judge missing
# a bad canary the AI judges catch. Needs the AI models pulled (Ollama).
scenario: venv ## Run the MW blind-spot scenario: default vs AI judges (+ Referee links)
	$(PY) tools/run_scenario.py

results: venv ## List the latest recorded Kayenta results (verdict, score, Referee links)
	$(PY) tools/list_results.py

# Baseline demo: default (statistical) + DUMMY-AI (no-op, no model) + hybrid, over
# the dummy dataset. This is the model-free regression path; it does NOT use an LLM.
# (The real LLM/VLM judge demo is `make scenario`.)
demo-dummy: venv ## Baseline demo: default + dummy-AI + hybrid (NO model, regression only)
	$(PY) tools/run_pipeline.py

pipeline: demo-dummy ## (alias of demo-dummy, kept for back-compat)

# Run ONE baseline judge end-to-end (seed + analyze + record id + print Referee link).
#   make judge JUDGE=default | dummy | hybrid
judge: venv ## Run a single baseline judge (JUDGE=default|dummy|hybrid)
	$(PY) tools/run_pipeline.py --only $(JUDGE)

# Open Referee showing the rendered canary result. Defaults to the latest
# default-judge run recorded in data/last_run.json.
#   make referee                       -> default report
#   make referee JUDGE=dummy|hybrid    -> a baseline-demo report
#   make referee JUDGE=scenario-summary (or scenario-default|raw|plot) -> a scenario report
#   make referee EXEC=<id>             -> a specific execution id
JUDGE ?= default
referee: ## Open Referee on a recorded report (JUDGE=<key> or EXEC=<id>; `make results` lists keys)
	@base="$(REFEREE_URL)"; \
	path="/dashboard/reports/standalone_canary_analysis/"; \
	id="$(EXEC)"; \
	if [ -z "$$id" ] && [ -f data/last_run.json ]; then \
		id=$$($(PY) -c "import json;print(json.load(open('data/last_run.json')).get('$(JUDGE)',''))" 2>/dev/null); \
	fi; \
	if [ -z "$$id" ]; then \
		echo "No execution id found (run 'make demo-dummy' or 'make scenario' first). Opening dashboard."; \
		url="$$base/dashboard"; \
	else \
		url="$$base$$path$$id"; \
		echo "Opening Referee report ($(JUDGE)) for execution $$id"; \
	fi; \
	echo "$$url"; \
	open "$$url" 2>/dev/null || xdg-open "$$url" 2>/dev/null || true

# The functional test: bring everything up, wait for health, seed, run all
# three judges. Propagates a non-zero exit from the pipeline.
# Uses ensure-up (not up): no forced rebuild, so repeat runs are fast and don't
# re-export images when nothing changed. If you edited judge-service/ or referee/,
# run `make up` (or `make build`) first to rebuild.
validate: ensure-up venv wait-kayenta seed pipeline ## Full end-to-end functional test
	@echo "validate: OK"

# Per-model AI sweep: validates only models present locally (Ollama), marks the
# rest pending. Does NOT gate on a model being up beyond the ones pulled.
validate-ai: venv ## Sweep configured AI models present locally (LLM + VLM)
	$(PY) tools/validate_ai.py

# --- Labelled dataset + scored experiment -----------------------------------
seed-eval: venv ## Seed the labelled evaluation dataset into VM + write manifest (--probe shows the statistical judge per family)
	$(PY) tools/seed_eval_dataset.py --probe

# Full scored experiment: every judge x every scenario x every present model.
# Skips absent models. Override size/seed/models via EVAL_N / EVAL_SEED / EVAL_MODELS.
experiment: ensure-up venv wait-kayenta ## Run the full scored experiment -> results/
	$(PY) tools/run_experiment.py

experiment-quick: ensure-up venv wait-kayenta ## Smoke run of the experiment (n=1, two default models)
	$(PY) tools/run_experiment.py --quick

# Mock-first test: stubs the model endpoint, asserts a valid CanaryJudgeResult.
# Runs inside the judge-service container (no model required).
test-judge-mock: ## Run the AI judge mock test (no model needed)
	$(COMPOSE) exec -T judge-service python - < judge-service/tests/test_mock.py

clean: ## Stop the stack and remove volumes
	$(COMPOSE) down -v

# --- Post-hoc analysis and figures ------------------------------------------
#
# Two groups, split by whether the script re-derives tables from results already
# on disk or has to call Kayenta again.
#
#   OFFLINE (make analysis)     read results/ and results/agg/, write results/agg/.
#                               No container, no model, no network.
#   LIVE    (make analysis-live) call judge_clients, so Kayenta (and for some,
#                               judge-service + Ollama or Bedrock) must be up.
#
# The offline set is ordered by what each script reads and writes:
#   bounded_confidence_intervals  needs the multi-seed long_results.csv (from the
#                                 live set) and WRITES agg/metrics_with_ci.csv
#                                 with Wilson intervals plus a bootstrap
#                                 cross-check. It must run before anything that
#                                 reads that file, which is ensemble_multiseed_ci.py.
#   pooled_mcnemar                needs agg/long_results.csv, and pairs against
#                                 agg/ensemble_multiseed_rows.csv when present
#                                 (it degrades to skipping the ensemble row).
#   baselines_and_mcc             needs results/summary.csv only. Independent.
#   roc_pr_curves                 needs results/results_raw.csv and
#                                 agg/frontier_long_*.csv. Independent.
#   run_fp_guard_experiment       needs results/results_raw.csv and
#                                 results/summary.csv. Independent.

ANALYSIS_SCRIPTS := \
	tools/bounded_confidence_intervals.py \
	tools/pooled_mcnemar.py \
	tools/baselines_and_mcc.py \
	tools/roc_pr_curves.py \
	tools/run_fp_guard_experiment.py

# Add a new generator here; nothing else changes. Each takes --results and --out
# and writes a PNG, so a host that has matplotlib can run any of them directly.
FIGURE_SCRIPTS := \
	tools/figure_distributions.py \
	tools/figure_blindspot_series.py \
	tools/figure_approach_bars.py \
	tools/figure_threshold_summaries.py \
	tools/figure_multiseed_intervals.py

# render_figures.py predates the others and takes --spec as well, so it is invoked
# separately rather than bent into the loop above.
# The locally-built judge-service image, named without an explicit tag so docker
# resolves it to whatever `make build` produced. Nothing upstream is referenced
# here, so there is no external tag to pin.
FIGURE_IMAGE := canaryllm-judge-service

require-results:
	@if [ ! -f results/results_raw.csv ]; then \
		echo "results/results_raw.csv is missing. Run 'make experiment' first, or copy" >&2; \
		echo "the bundled example run into results/ (see example-results/README.md)." >&2; \
		exit 1; \
	fi

require-stack:
	@if ! curl -fsS http://localhost:$(KAYENTA_PORT)/health >/dev/null 2>&1; then \
		echo "Kayenta is not reachable on :$(KAYENTA_PORT). This target calls the genuine" >&2; \
		echo "judge and will not start the stack for you. Run 'make up' first." >&2; \
		exit 1; \
	fi

analysis: venv require-results ## Re-derive every table from results/ (offline: no stack, no model)
	@for s in $(ANALYSIS_SCRIPTS); do echo "--- $$s"; $(PY) $$s || exit 1; done
	@echo "analysis: OK -> results/agg/"

# Everything here calls judge_clients and therefore needs Kayenta up. The two
# frontier targets additionally need AWS_BEARER_TOKEN_BEDROCK; the multiseed
# sweep additionally needs Ollama and the local models pulled. Ordered so each
# script's inputs exist when it runs: bounded_confidence_intervals is invoked
# mid-chain because ensemble_multiseed_ci.py reads the file it rewrites.
analysis-live: venv require-stack ## Re-run the analyses that call Kayenta (NEEDS the stack up; frontier steps need Bedrock creds)
	$(PY) tools/run_multiseed_corrected.py --out results/n5-corrected
	$(PY) tools/run_ensemble_experiment.py
	$(PY) tools/run_frontier_experiment.py
	$(PY) tools/bounded_confidence_intervals.py
	$(PY) tools/ensemble_multiseed_ci.py
	$(PY) tools/reproducibility_test_retest.py
	@echo "analysis-live: OK -> results/agg/"

# The generators need matplotlib, which is in the judge-service image and not in
# the host venv (PyPI is gated on the reference machine). They run in a one-off
# container over a bind mount of the repository, so the stack does not have to be
# up: only the image has to exist. A host that has matplotlib can skip all this
# and run any generator directly.
figures: require-results ## Render every figure into results/figures/ (needs the judge-service image built; no stack, no model)
	@if ! docker image inspect $(FIGURE_IMAGE) >/dev/null 2>&1; then \
		echo "$(FIGURE_IMAGE) is not built. Figures render inside that image because" >&2; \
		echo "matplotlib is not in the host venv. Run 'make build' first." >&2; \
		exit 1; \
	fi
	@mkdir -p results/figures
	@for s in $(FIGURE_SCRIPTS); do \
		echo "--- $$s"; \
		docker run --rm -v "$(PWD)":/work -w /work $(FIGURE_IMAGE) \
			python $$s --results results --out results/figures || exit 1; \
	done
	@echo "--- tools/render_figures.py"
	@docker run --rm -v "$(PWD)":/work -w /work $(FIGURE_IMAGE) \
		python tools/render_figures.py --spec data/ai-logs/_exp_figdata.json \
			--results results --out results/figures
	@echo "figures: OK -> results/figures/"

# The end-to-end check a fresh clone can run: committed data in, a rendered
# figure out, no stack and no model. example-results/ is a small slice of a real
# run kept for exactly this, and is documented in the README as example data
# rather than as the study's results.
#
# figures-published used to live here. It rendered every manuscript figure from
# reference-results/n20 and results/corrected, and both of those are now archived
# with the paper's data instead of shipped with the testbed, so the target has
# nothing to read. Rendering the paper's figures is done from the archive; see
# the README section on where the paper's data lives.
demo: ## Render one figure from the committed example data (no stack, no model, no network)
	@if ! docker image inspect $(FIGURE_IMAGE) >/dev/null 2>&1; then \
		echo "$(FIGURE_IMAGE) is not built. Figures render inside that image because" >&2; \
		echo "matplotlib is not in the host venv. Run 'make build' first." >&2; \
		exit 1; \
	fi
	@mkdir -p results/figures-demo
	docker run --rm -v "$(PWD)":/work -w /work $(FIGURE_IMAGE) \
		python tools/figure_approach_bars.py \
			--results example-results --out results/figures-demo
	@echo "demo: OK -> results/figures-demo/"

# reproduce, verify-repro and verify-repro-quick were removed here. All three
# compared a fresh run against reference-results/, the published run for the
# paper, which is archived with the paper's data rather than shipped with the
# testbed. The scripts moved with it. Comparing your own runs against each other
# needs neither.

replay-logs: venv ## Replay archived data/ai-logs payloads through the current parser (no model, no stack)
	$(PY) tools/replay_ai_logs.py

prompts: venv ## List the available rubrics with their ids and text hashes
	@$(PY) -c "import sys; sys.path.insert(0,'judge-service/app'); import prompt_registry as p; \
	[print(f'  {i:26s} {p.load(i).hash}  {p.load(i).description}') for i in p.available()]; \
	print(f'\n  default: {p.DEFAULT_PROMPT_ID}')"
