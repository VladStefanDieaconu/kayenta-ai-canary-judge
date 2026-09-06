# VictoriaMetrics (metric store)

Single-node `victoriametrics/victoria-metrics` (vmsingle), published on host port
**8428**. Kayenta's Prometheus metric source talks to it unchanged at
`http://victoriametrics:8428` (in-network).

## Why VictoriaMetrics instead of plain Prometheus
- Speaks the Prometheus query API (`/api/v1/query`, `/api/v1/query_range`), so
  Kayenta's Prometheus provider works against it with no changes.
- **Accepts back-dated samples**, unlike plain Prometheus. We rely on this to load
  dummy series at chosen timestamps now, and realistic synthetic series later.

## How data is loaded
There is no scrape config. `tools/seed_dummy_data.py` pushes the dummy series via
the Prometheus text import endpoint:

```
POST http://localhost:8428/api/v1/import/prometheus
```

Each line is `metric{labels} value timestamp_ms`. The dummy dataset carries a
single distinguishing label `Canary` with values `Control` (baseline) and
`Experiment` (canary), plus the metric name, nothing else, so the canary-config
template `metric{Canary="${scope}"}` selects exactly one series per scope.

## Reading data back
```
GET http://localhost:8428/api/v1/query_range?query=dummy_latency{Canary="Control"}&start=...&end=...&step=60
```

VM is started with `-search.latencyOffset=1s` so freshly-imported recent samples
are queryable almost immediately (the default 30s offset would hide just-written
points from range queries over a "now" window).

Data persists in the `vm-data` named volume across restarts.
