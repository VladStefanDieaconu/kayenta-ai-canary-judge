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
        analysis analysis-live figures demo require-results require-stack \
        own-data own-data-quick own-data-check own-data-template require-own-data

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

# --- Your own canary windows -------------------------------------------------
#
# The same judges, the same statistical configuration and the same scoring code
# as above, over windows you exported from your own metric store. The format and
# every rule the loader enforces are in docs/own-data.md; nothing here needs a
# file in this repository to be edited.
#
# OWN_DATA is read from .env too (the -include above), so putting
# OWN_DATA=./windows.csv there makes every target below work with no argument.
#
# The four below are the loader's documented switches (docs/own-data.md), one
# variable each, so every flag the documentation names is reachable from the
# documented command. Empty means "not passed", which is the loader's default:
#   STAT_CONFIG=tolerances.json   per-service statistical overrides
#   MIN_SAMPLES=8                 lower the 10-samples-a-side floor
#   ALLOW_GAPS=1                  accept windows above the 20% missing limit
#   KEEP_IDS=1                    do not rename (your names reach the model)
OWN_DATA ?=
STAT_CONFIG ?=
MIN_SAMPLES ?=
ALLOW_GAPS ?=
KEEP_IDS ?=
OUT ?= own-data-template.csv

# The switches every target that reads your file passes through, so one edit
# reaches all of them and none can drift out of step with another.
OWN_DATA_FLAGS = $(if $(STAT_CONFIG),--stat-config $(STAT_CONFIG),) \
	$(if $(MIN_SAMPLES),--min-samples $(MIN_SAMPLES),) \
	$(if $(ALLOW_GAPS),--allow-gaps,) \
	$(if $(KEEP_IDS),--keep-ids,)

require-own-data:
	@if [ -z "$(OWN_DATA)" ]; then \
		echo "OWN_DATA is not set. Point it at your exported windows, either on the" >&2; \
		echo "command line ('make own-data OWN_DATA=./windows.csv') or as OWN_DATA=./windows.csv" >&2; \
		echo "in .env. 'make own-data-template' writes a valid example to copy the shape" >&2; \
		echo "from, and docs/own-data.md specifies the format." >&2; \
		exit 1; \
	fi
	@if [ ! -f "$(OWN_DATA)" ]; then \
		echo "OWN_DATA=$(OWN_DATA) does not exist." >&2; \
		exit 1; \
	fi

own-data-template: venv ## Write a valid example input file in the own-data format (OUT=<file>)
	$(PY) tools/own_data.py --export $(OUT)

# Needs neither the stack nor a model, and is the target to run first: it prints
# exactly what will be judged before a single model call is paid for.
own-data-check: venv require-own-data ## Validate your own windows and print what the harness will see (OWN_DATA=<file>)
	$(PY) tools/own_data.py --check $(OWN_DATA) $(OWN_DATA_FLAGS)

own-data: ensure-up venv wait-kayenta require-own-data ## Run the scored experiment over your own windows -> results/own-data/ (OWN_DATA=<file>)
	$(PY) tools/run_experiment.py --own-data $(OWN_DATA) $(OWN_DATA_FLAGS)

own-data-quick: ensure-up venv wait-kayenta require-own-data ## Smoke run over the first two of your own windows (OWN_DATA=<file>)
	$(PY) tools/run_experiment.py --own-data $(OWN_DATA) --quick $(OWN_DATA_FLAGS)

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
# RESULTS is the run the offline set reads. It defaults to results/, which is
# where `make experiment` writes, and `make analysis RESULTS=results/own-data`
# re-derives the same tables over your own windows -- the point of the flag, and
# the answer to a `make own-data` that used to be followed by an `analysis`
# target telling you to go and run the seeded experiment instead.
#
# Every recipe below quotes it, because `tools/run_experiment.py --out "a b"`
# accepts a directory whose name has a space in it and there is no reason
# `make analysis RESULTS="a b"` should not. Unquoted it word-split before the
# script saw it: `require-results` passed (it quoted), and then
# `baselines_and_mcc.py --results a b` died on `unrecognized arguments: b`.
# `figures` was worse -- `mkdir -p a b/figures` made a stray `a/` first.
RESULTS ?= results

# Is $(RESULTS) the repository's own results/ directory? A shell fragment rather
# than a make comparison, because the answer has to be about the directory and
# not about how it was spelled: `results`, `./results`, `results/` and an
# absolute path are one directory, and a literal compare against the string
# "results" made `make analysis RESULTS=./results` skip the three seeded-only
# scripts on the seeded run itself and say nothing. Sets $$seeded_run to 1 or 0.
# Both sides must resolve, so an unreadable $(RESULTS) is never mistaken for it.
IS_SEEDED_RUN = seeded_run=0; \
	_this=$$(cd "$(RESULTS)" 2>/dev/null && pwd -P || true); \
	_seeded=$$(cd "$(CURDIR)/results" 2>/dev/null && pwd -P || true); \
	if [ -n "$$_this" ] && [ "$$_this" = "$$_seeded" ]; then seeded_run=1; fi

# The figure spec tools/run_experiment.py writes for the run it just scored. The
# seeded run leaves it where it has always been, under data/ai-logs -- the
# directory the judge-service container and the host share, and where the study's
# own spec lives. Every other run writes it beside its own results, so
# `make figures RESULTS=<dir>` draws <dir>'s numbers. run_experiment.py picks the
# same path by the same rule, so the two cannot disagree; when it hardcoded
# data/ai-logs here, `make figures RESULTS=results/own-data` redrew three of the
# article's figures from someone else's windows.
FIGSPEC = $$(if [ $$seeded_run = 1 ]; then echo data/ai-logs/_exp_figdata.json; \
	else echo "$(RESULTS)/_exp_figdata.json"; fi)

# The offline set splits by whether a script can read a run other than the
# seeded one. Those that can take --results and are pointed at $(RESULTS):
#   baselines_and_mcc             needs <results>/summary.csv and
#                                 <results>/results_raw.csv (for the class
#                                 balance the trivial baselines are computed
#                                 from). Independent.
#   roc_pr_curves                 needs <results>/results_raw.csv and
#                                 agg/frontier_long_*.csv. Independent.
#
# Those that cannot are run only for the seeded run, and skipped with a reason
# otherwise. Running them bare against results/ while the scoreboard beside them
# came from somewhere else is how the study's numbers end up in someone's own
# report:
#   bounded_confidence_intervals  needs the multi-seed long_results.csv (from the
#                                 live set) and WRITES agg/metrics_with_ci.csv
#                                 with Wilson intervals plus a bootstrap
#                                 cross-check. It must run before anything that
#                                 reads that file, which is ensemble_multiseed_ci.py.
#                                 Its part B additionally checks the published
#                                 statistical accuracy (0.544) and reads the nine
#                                 seeded families at 20 instances each.
#   pooled_mcnemar                needs agg/long_results.csv, and pairs against
#                                 agg/ensemble_multiseed_rows.csv when present
#                                 (it degrades to skipping the ensemble row).
#                                 Both come from the five-seed sweep.
#   run_fp_guard_experiment       needs <results>/results_raw.csv and
#                                 <results>/summary.csv, but its guard signals are
#                                 regenerated from the seeded scenarios and keyed
#                                 on their ids, and the two families the guards
#                                 target are families of that dataset. It refuses
#                                 a results directory whose scenarios are not
#                                 those, so it is here rather than above.

ANALYSIS_SCRIPTS := \
	tools/baselines_and_mcc.py \
	tools/roc_pr_curves.py

SEEDED_ANALYSIS_SCRIPTS := \
	tools/bounded_confidence_intervals.py \
	tools/pooled_mcnemar.py \
	tools/run_fp_guard_experiment.py

# Add a new generator here; nothing else changes. Each takes --results and --out
# and writes a PNG, so a host that has matplotlib can run any of them directly.
#
# Split the same way the offline analyses are, and for the same reason. Every one
# of them ACCEPTS --results, so running the whole list against any directory
# looked like it worked; four of them do not READ it, and the target was offered
# in the README as `make figures RESULTS=results/own-data`. That run wrote
# fig02_distributions.png (clean_mean_shift_000 and variance_increase_000 at seed
# 20260621) and fig03_blindspot_series.png into the user's directory and then
# died in figure_multiseed_intervals.py, which needs a five-seed sweep an
# own-data run cannot produce -- leaving two of the study's figures behind as the
# only output.
#
# Those that read $(RESULTS):
RESULTS_FIGURE_SCRIPTS := \
	tools/figure_approach_bars.py \
	tools/figure_threshold_summaries.py

# Those that do not, and so are run only for the seeded run:
#   figure_distributions      draws two seeded instances from eval_dataset.py at
#                             --seed; --results is accepted and ignored.
#   figure_blindspot_series   the same, for two blind-spot families.
#   figure_multiseed_intervals needs agg/metrics_with_ci.csv from the five-seed
#                             sweep (`make analysis-live`).
#   figure_diagrams           four schematics that read nothing at all, and are
#                             the article's Figures 1, 4, 5 and 6.
SEEDED_FIGURE_SCRIPTS := \
	tools/figure_distributions.py \
	tools/figure_blindspot_series.py \
	tools/figure_multiseed_intervals.py \
	tools/figure_diagrams.py

# The seeded run keeps running all six, in the order it always has.
FIGURE_SCRIPTS := \
	tools/figure_distributions.py \
	tools/figure_blindspot_series.py \
	tools/figure_approach_bars.py \
	tools/figure_threshold_summaries.py \
	tools/figure_multiseed_intervals.py \
	tools/figure_diagrams.py

# Does $(RESULTS) hold a single labelled window? Sets $$labelled to 1 or 0.
#
# A run in which nothing is labelled is a documented mode (docs/own-data.md), and
# every figure this target draws is a picture of accuracy, of which such a run has
# none. Run against one, `figures` neither completed nor refused: it wrote an empty
# fig10_approach_bars.png with all five columns SKIPPED, then exited 1 out of
# figure_threshold_summaries.py with "agg/roc_pr_auc.csv not found; run
# tools/roc_pr_curves.py first" -- advice that cannot be followed, because
# roc_pr_curves.py correctly and deliberately writes nothing for an unlabelled run.
# render_figures.py would have gone the same way: run_experiment.py writes no figure
# spec when there is nothing to score.
#
# The header is read for the `truth` column rather than its position being assumed,
# and the scan stops at the first labelled row. Only the first three fields are
# touched, which are identifiers; the quoted free text is further right.
HAS_LABELLED_RUN = labelled=$$(awk -F, 	'NR == 1 { for (i = 1; i <= NF; i++) if ($$i == "truth") col = i; next } 	 col && $$col != "" { found = 1; exit } 	 END { print (found ? 1 : 0) }' "$(RESULTS)/results_raw.csv")

# render_figures.py predates the others and takes --spec as well, so it is invoked
# separately rather than bent into the loop above.
# The locally-built judge-service image, named without an explicit tag so docker
# resolves it to whatever `make build` produced. Nothing upstream is referenced
# here, so there is no external tag to pin.
FIGURE_IMAGE := canaryllm-judge-service

require-results:
	@if [ ! -f "$(RESULTS)/results_raw.csv" ]; then \
		echo "$(RESULTS)/results_raw.csv is missing. Run 'make experiment' first, or" >&2; \
		echo "'make own-data OWN_DATA=<file>' and then 'make analysis RESULTS=results/own-data'," >&2; \
		echo "or copy the bundled example run in (see example-results/README.md)." >&2; \
		exit 1; \
	fi

require-stack:
	@if ! curl -fsS http://localhost:$(KAYENTA_PORT)/health >/dev/null 2>&1; then \
		echo "Kayenta is not reachable on :$(KAYENTA_PORT). This target calls the genuine" >&2; \
		echo "judge and will not start the stack for you. Run 'make up' first." >&2; \
		exit 1; \
	fi

analysis: venv require-results ## Re-derive every table from RESULTS= (default results/; RESULTS=results/own-data for your own run; offline)
	@for s in $(ANALYSIS_SCRIPTS); do echo "--- $$s"; $(PY) $$s --results "$(RESULTS)" || exit 1; done
	@$(IS_SEEDED_RUN); \
	if [ $$seeded_run = 1 ]; then \
		for s in $(SEEDED_ANALYSIS_SCRIPTS); do echo "--- $$s"; $(PY) $$s || exit 1; done; \
	else \
		echo "--- not run for RESULTS=$(RESULTS):"; \
		for s in $(SEEDED_ANALYSIS_SCRIPTS); do echo "      $$s"; done; \
		echo "    These three are built for the seeded run and cannot read another one:"; \
		echo "    two need the five-seed sweep ('make analysis-live'), which an own-data"; \
		echo "    run does not produce, and the guard ablation is defined against the"; \
		echo "    seeded families. Run bare they would read results/ and report the"; \
		echo "    study's numbers beside yours."; \
	fi
	@echo "analysis: OK -> $(RESULTS)/agg/"

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
figures: require-results ## Render every figure into RESULTS=/figures/ (default results/; needs the judge-service image built; no stack, no model)
	@if ! docker image inspect $(FIGURE_IMAGE) >/dev/null 2>&1; then \
		echo "$(FIGURE_IMAGE) is not built. Figures render inside that image because" >&2; \
		echo "matplotlib is not in the host venv. Run 'make build' first." >&2; \
		exit 1; \
	fi
	@$(HAS_LABELLED_RUN); \
	if [ "$$labelled" = "0" ]; then \
		echo "figures: nothing to draw for RESULTS=$(RESULTS)."; \
		echo "    No window in $(RESULTS)/results_raw.csv carries a label. Every figure this"; \
		echo "    target draws is a picture of accuracy, and this run has none: the judges ran"; \
		echo "    and every verdict, score and rationale is in results_raw.csv. Running without"; \
		echo "    labels is a documented mode (docs/own-data.md), not an error, so this is a"; \
		echo "    skip and not a failure -- the same treatment 'make analysis' gives it."; \
		echo "    --- not run for RESULTS=$(RESULTS):"; \
		echo "          tools/figure_approach_bars.py       every column would be SKIPPED; it"; \
		echo "                                              wrote an empty figure"; \
		echo "          tools/figure_threshold_summaries.py needs agg/roc_pr_auc.csv, which"; \
		echo "                                              tools/roc_pr_curves.py deliberately"; \
		echo "                                              does not write for an unlabelled run"; \
		echo "          tools/render_figures.py             needs the figure spec, which"; \
		echo "                                              tools/run_experiment.py does not"; \
		echo "                                              write when there is nothing to score"; \
		echo "    Label the windows you remember -- including the ones you promoted and did not"; \
		echo "    regret -- and run it again."; \
		exit 0; \
	fi; \
	mkdir -p "$(RESULTS)/figures"; \
	$(IS_SEEDED_RUN); \
	if [ $$seeded_run = 1 ]; then scripts="$(FIGURE_SCRIPTS)"; \
	else scripts="$(RESULTS_FIGURE_SCRIPTS)"; fi; \
	for s in $$scripts; do \
		echo "--- $$s"; \
		docker run --rm -v "$(PWD)":/work -w /work $(FIGURE_IMAGE) \
			python $$s --results "$(RESULTS)" --out "$(RESULTS)/figures" || exit 1; \
	done; \
	echo "--- tools/render_figures.py"; \
	docker run --rm -v "$(PWD)":/work -w /work $(FIGURE_IMAGE) \
		python tools/render_figures.py --spec "$(FIGSPEC)" \
			--results "$(RESULTS)" --out "$(RESULTS)/figures" || exit 1; \
	if [ $$seeded_run = 0 ]; then \
		echo "--- not run for RESULTS=$(RESULTS):"; \
		for s in $(SEEDED_FIGURE_SCRIPTS); do echo "      $$s"; done; \
		echo "    These four do not read a results directory. Two draw their series"; \
		echo "    from eval_dataset.py at the published seed, one needs the five-seed"; \
		echo "    sweep an own-data run does not produce, and one is four schematics"; \
		echo "    that are the same picture whatever ran. Run against $(RESULTS) they"; \
		echo "    would write the study's figures into your directory under the"; \
		echo "    study's filenames, which is how the article's numbers end up in"; \
		echo "    someone else's report."; \
	fi; \
	echo "figures: OK -> $(RESULTS)/figures/"

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
