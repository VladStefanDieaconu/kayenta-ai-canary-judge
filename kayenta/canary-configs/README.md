# Checked-in canary configs: demos, not the benchmark configuration

The JSON files in this directory drive the demo targets (`make validate`,
`make scenario`, `make demo-dummy`, `make judge`). **None of them produced a
benchmark result.** The scored experiment builds one canary config per scenario
in code, from `STAT_CANARY_TEMPLATE` in `tools/eval_dataset.py`, and records that
template in every manifest (`statistical_template`).

The statistical demo configs differ from that template:

| setting | `scenario-statistical.json` (demo) | `STAT_CANARY_TEMPLATE` (benchmark) |
|---|---|---|
| `direction` | `increase` | `increase` |
| `effectSize.allowedIncrease` | 1.1 | 1.05 |
| `effectSize.criticalIncrease` | 1.5 | 1.25 |
| `effectSize.allowedDecrease` / `criticalDecrease` | not set | 0.95 / 0.75 (inert under `direction=increase`) |
| `nanStrategy` | `replace` | `remove` |
| `outliers.strategy` | `keep` | `keep` |
| `critical` / `mustHaveData` | not set | `true` / `true` |
| `scoreThresholds` | pass 75 / marginal 50 | pass 75 / marginal 50 |

`statistical.json` sets only `direction` per metric and leaves every other field at
Kayenta's default.

When citing the statistical judge's configuration, cite `STAT_CANARY_TEMPLATE` (or
the manifest), not a file in this directory. The Mann-Whitney confidence level and
the effect-size tolerance are not fields of either; they are Kayenta defaults.
