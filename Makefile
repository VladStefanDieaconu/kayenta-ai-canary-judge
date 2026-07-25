# kayenta-ai-canary-judge: convenience targets.
# Host tools (seeder + pipeline) run in a local venv created from tools/requirements.txt.

SHELL := /bin/bash
COMPOSE := docker compose
VENV := .venv
PY := $(VENV)/bin/python
PIP := $(VENV)/bin/pip
KAYENTA_PORT ?= 8090

.DEFAULT_GOAL := help

.PHONY: help up ensure-up down logs ps build venv wait-kayenta \
        seed seed-scenario seed-eval demo-dummy pipeline judge scenario results referee \
        validate validate-ai test-judge-mock experiment experiment-quick clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

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

# --system-site-packages lets the venv inherit a host-installed `requests`,
# so the tools work even where access to a package index is gated/offline.
$(VENV)/.installed: tools/requirements.txt
	python3 -m venv --system-site-packages $(VENV)
	-$(PIP) install --no-deps -r tools/requirements.txt
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
	@base="http://localhost:$${REFEREE_PORT:-3001}"; \
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
