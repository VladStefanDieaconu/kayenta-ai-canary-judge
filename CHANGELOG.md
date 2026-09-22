# Changelog

Version 1.0.0 stays tagged and archived. Each entry says what changed, whether it
can change a verdict, and what a reader of 1.0.0 could have been misled on.

## 1.0.1 (unreleased)

**What stays the same.**
- The scored scoring path (`tools/run_experiment.py`,
  `tools/run_frontier_experiment.py`, `tools/run_multiseed_corrected.py`) gives the
  same verdict for every call that succeeds.
- `original-180` and `generalisation-60` generate the same scenarios, seeds, metric
  names, series, pairs and manifest entries as 1.0.0.
- The three rubric hashes are unchanged.
- Every change that could move a published verdict is off by default, or lives in
  its own sensitivity script.

### Changed defaults in the judge service (no published number was produced by the service)

- **The hybrid's AI verdict comes from the AI score** (`hybridAiVerdict=score`). The
  offline harness that scored every published hybrid number passes `classify(score)`
  and never reads the model's `overallVerdict` label. 1.0.0 of the service passed the
  label, so the served gated hybrid was a different decision function from the scored
  one: it followed the model's label, its keyword escalation was live, and its
  boundary was `<= 50`. `hybridAiVerdict=label` restores 1.0.0.
  *Misleading in 1.0.0:* `hybrid_policy.py` and the README said sharing
  `decide_policy()` guarantees that the scored hybrid is what the service returns.
- **A failed AI half is not a verdict.** 1.0.0 passed the Fail/0.0 placeholder of a
  failed model call into the policy, which overrides a clean statistical PASS. Now
  the hybrid returns the statistical verdict alone, with `judgeMetadata.ok=false` and
  `error_kind=ai_half_failed`. `onAiError=fail` returns an explicit Fail at score 0.
- **`prompt_id` and `seed` reach the hybrid's AI half.** 1.0.0 rebuilt the AI
  settings from mode and model only, so a hybrid request ran under the default rubric
  and seed whatever the config said.
- **The canary-config name and description no longer carry the family or truth
  label** (`eval-<suffix>`, `eval dataset scenario <suffix>`). No judge reads either
  field, and the service logged them verbatim. `build_canary_config(...,
  legacy_description=True)` reproduces the 1.0.0 strings.
- **New datasets get opaque metric-name tokens.** A `DATASETS` entry without
  `metric_suffix` gets a random token per scenario, from an RNG seeded by a recorded
  token seed and independent of family and order; it refuses to build without that
  seed. `original-180` and `generalisation-60` keep their `_gNNNN` suffix, because
  renaming would re-key every archived result.
  *Misleading in 1.0.0:* the README and `eval_dataset.py` called the gid "opaque" and
  said the ground truth never leaks. The gid is sequential in family order and, with
  the metric kinds, identifies the family.

### Added, recorded but not enforced by default

- **`judgeMetadata.score_check`** records why `overallScore` is not a finite number in
  [0, 100]. The score is still clamped. `JUDGE_STRICT_SCORE=1` turns a failed check
  into a `parse_failure` error. Off by default: it can change archived verdicts,
  because 1.0.0 clamped an out-of-range score and derived a fallback score when
  `overallScore` was missing, and both were scored.
- **`tools/sensitivity_strict_score.py`** says which archived AI rows strict scoring
  could move. From the result frames alone, a row whose stored score is not 0, 1, 10,
  40, 70 or 100 cannot move; a row at one of those values is undetermined, and each
  configuration's accuracy range is reported. With `--ai-logs` it re-parses every
  logged completion and lists those that fail the check. It never edits a row.
- **`judgeMetadata.truncated`** is set when finish_reason is `length` or completion
  tokens reached `max_tokens`. `JUDGE_TRUNCATION_IS_ERROR=1` turns it into a
  `truncated` error. Off by default: in the deposited five-seed sweep, one scored
  `deepseek-r1-llm` verdict came from a completion at exactly the 1024-token cap.
- **`tools/sensitivity_truncation.py`** lists every archived AI row at the cap, and
  each configuration's accuracy with and without those rows. It never edits a row.
- `judge_clients` passes `truncated` and `score_check` through, and adds
  `kayenta_reachable()`.

### Fixed in the harness

- **A failed statistical call is an error, not a verdict.**
  - `run_experiment.run_scenario` and `run_frontier_experiment.run_scenario` scored
    the FAIL/0.0 object of a failed Kayenta call as the statistical verdict, and
    derived all three hybrids from it. Both were correct on every FAIL-truth
    scenario. The call is now an error record, and no hybrid is derived from it.
    None of the archived statistical rows carries an error, so no published number
    moves.
  - `run_ensemble_experiment.materialise` now raises `StatisticalJudgeUnavailable`
    instead of carrying the placeholder into the ensemble.
  - `tools/test_failed_call_is_not_scored.py` covers the case.
- **`tools/validate_new_families.py` stops when Kayenta is unreachable.**
  *Misleading in 1.0.0:* its docstring said it was free to re-run with nothing
  running. Offline it exited 0 and reported statistical accuracy 1.0 on all three
  held-out families, where the deposited run has 1.0, 0.05 and 0.0. It also moved the
  tuned ensemble's `partial_recovery` accuracy from 1.0 to 0.95.
- `tools/roc_pr_curves.py --raw` and `tools/figure_approach_bars.py --summary` read
  the corrected per-scenario files from the data deposit, while `agg/` stays where it
  is. The defaults are unchanged.
- **`tools/bounded_confidence_intervals.py` writes the single-seed interval table for
  a corrected run.** 1.0.0 stopped with a `ValueError` on a family cell with no scored
  row (blank in `family_matrix.csv`), and took n = 20 for every family cell. It now
  writes a no-data cell as a row with n = 0 and empty rate and intervals, and counts
  per-family k and n from `results_raw.csv`. New `--results`, `--out-dir` and
  `--part {all,multiseed,headline}` options; the defaults are unchanged.
  *Misleading in 1.0.0:* the post-correction single-seed interval table could not be
  produced at all. On complete 20-instance inputs the three tables are byte-identical
  to those of 1.0.0.

### Documentation corrections

- **Prompt descriptions.** `prompts/v2-operational-2026-07.prompt` and
  `v3-temporal-2026-07.prompt` called the variants "Pre-registered". No dated
  registration exists, so the word is removed. The v2 commentary said one sentence
  was appended; it is two. Descriptions and commentary are not part of the prompt
  hash, so the hashes do not change.
- **`judge-service/models.yaml`.**
  - It said modality is the only thing the judge branches on. The judge also reads
    `json_mode` and `max_tokens`, and the file is baked into the image.
  - The gpt-oss comment cited a 21% empty-completion rate from a run that is not in
    the data deposit, and "1085 archived Opus and Qwen responses, longest ~252
    tokens". The deposit holds 1,890 such rows, and the longest is 363.
  - It said 1024 never binds for any other alias. It did for `deepseek-r1-llm`.
  - DeepSeek-R1-Distill's base model and licence are added.
- **`.env.example`.** "Every published number ran at 1024" was wrong: gpt-oss ran at
  4096.
- **`tools/eval_dataset.py` comments.**
  - The template comment said the statistical judge's shortfall is "real, not a
    config artifact" because the rank test is "median-based". It tests a location
    shift (Hodges-Lehmann). For `cross_metric_marginal` the miss follows from the
    1.05 effect-size gate.
  - `late_transient` is healthy until an onset at 72-78% of the window (sample 22-24),
    not "the first three quarters".
  - `sustained_excursion`'s run placement does not hold the fitted slope near zero.
  - `gradual_drift` and `cross_metric_marginal` are identifiable by metric kind.
- **Judge-service code comments.**
  - `llm_judge.py` said a failed call returns an "Error" classification. The overall
    classification is Fail at score 0; only the per-metric classifications are Error.
  - `ensemble_judge.py` cited "15/15" `healed_transient` firings. The archived sets
    give 30/30 (tuning) and 20/20 (reported).
  - `ensemble_tuned.json` said "eight values". There are seven.
- **`kayenta/canary-configs/`.** Added `README.md`: these are demo configs, and the
  benchmark used `STAT_CANARY_TEMPLATE` (1.05 / 1.25 / remove), not
  `scenario-statistical.json` (1.1 / 1.5 / replace). That file's description no
  longer says Mann-Whitney "only tests the median".
- **README.**
  - The data deposit DOI is now given.
  - "Because the affected families are FAIL-truth" is corrected: 21 of the 22 rows
    fell in FAIL-truth families.
  - `gated` is no longer called the recommended policy that "delivers
    statistical-like precision with AI-like recall". In the deposited 180-scenario
    local-model runs, it returned the same verdict as `or` on every scenario.

### Tests added

- `judge-service/tests/test_hybrid_release.py`: verdict source, AI-half failure,
  forwarding of `prompt_id` and `seed`, score and truncation checks.
- `tools/test_dataset_naming.py`: legacy names unchanged, opaque tokens, config text.
