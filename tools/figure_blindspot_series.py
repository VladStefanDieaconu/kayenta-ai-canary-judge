#!/usr/bin/env python3
"""Figure 3: two blind-spot scenarios as time series, control against experiment.

Panel (a) is a `variance_increase` instance: the medians coincide and the spread
is several times wider. Panel (b) is a `gradual_drift` instance: flat for most of
the window, then a ramp in the last fifth. Both carry the benchmark label FAIL
and both are passed by the rank test, which sees ranks and not time.

Series come from tools/eval_dataset.py at the given seed, so the figure plots the
same arrays the seeder loads into VictoriaMetrics and the judges score.

Needs matplotlib. On a host without it, run inside the judge-service image:

    docker run --rm -v "$PWD":/work -w /work canaryllm-judge-service \
        python tools/figure_blindspot_series.py

Usage:
  python tools/figure_blindspot_series.py [--seed 20260621] [--instance 0]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "tools"))
import eval_dataset as ed  # noqa: E402

CONTROL_COLOUR = "#1f77b4"
EXPERIMENT_COLOUR = "#d62728"

PANELS = [
    ("variance_increase", "(a) variance_increase: same mean, wider spread"),
    ("gradual_drift", "(b) gradual_drift: flat then late upward ramp"),
]


def instance(family: str, seed: int, n_per_family: int, index: int):
    """Selection is by scenario id, not by Scenario.gid, which is a global index
    across all families rather than a per-family one. `n_per_family` matters:
    each instance's RNG seed is derived from its global index, so the same family
    instance carries different values at n=12 and n=20.
    """
    wanted = f"{family}_{index:03d}"
    for sc in ed.build_scenarios(seed, n_per_family):
        if sc.id == wanted:
            series = ed.series_for_scenario(sc)
            m = sc.metrics[0]
            return series[m.name]["control"], series[m.name]["experiment"], sc, m
    raise SystemExit(f"no scenario {wanted} at seed {seed}, n={n_per_family}")


def ramp_onset(n: int, sc):
    """Sample index where gradual_drift's ramp begins, taken from the generator's
    own recorded `ramp_start_fraction` with the same max(1, int(...)) arithmetic
    it used. Returns None when the family records no ramp."""
    for p in (sc.params or {}).values():
        start = p.get("ramp_start_fraction")
        if start is not None:
            return max(1, int(n * float(start)))
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seed", type=int, default=ed.DEFAULT_MASTER_SEED)
    ap.add_argument("--n", type=int, default=20,
                    help="instances per family; each instance's RNG seed is "
                         "derived from its global index, so this changes the "
                         "values (default 20, the published run)")
    ap.add_argument("--instance", type=int, default=0,
                    help="which instance of each family to plot (default 0)")
    ap.add_argument("--results", default=str(REPO_ROOT / "results"),
                    help="accepted for a uniform generator interface; these two "
                         "figures are generated from the seed, not from results/")
    ap.add_argument("--out", default=str(REPO_ROOT / "results" / "figures"))
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args()

    fig, axes = plt.subplots(2, 1, figsize=(7.030, 4.527), dpi=args.dpi)
    reported = []
    for ax, (family, title) in zip(axes, PANELS):
        control, experiment, sc, metric = instance(family, args.seed, args.n, args.instance)
        x = list(range(len(control)))
        ax.plot(x, control, "o-", color=CONTROL_COLOUR, ms=3, lw=1.4, label="control")
        ax.plot(x, experiment, "o-", color=EXPERIMENT_COLOUR, ms=3, lw=1.4,
                label="experiment")
        onset = ramp_onset(len(control), sc)
        if onset is not None and family == "gradual_drift":
            ax.axvline(onset, color="grey", ls="--", lw=1.0)
            ax.annotate("ramp onset", xy=(onset, ax.get_ylim()[1]),
                        xytext=(3, -10), textcoords="offset points",
                        fontsize=8, color="grey", va="top")
        ax.set_title(f"{title}  |  rank test: PASS, label: {sc.truth}", fontsize=10)
        ax.set_ylabel(f"{metric.kind} ({metric.unit})")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left", fontsize=8)
        reported.append((sc.id, metric.name, len(control)))
    axes[-1].set_xlabel("sample index (time)")
    fig.tight_layout()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "fig03_blindspot_series.png"
    fig.savefig(path)
    plt.close(fig)

    print(f"[figure-03] seed={args.seed} n_per_family={args.n} instance={args.instance}")
    for sid, mname, n in reported:
        print(f"  {sid}: metric={mname}, {n} samples")
    print(f"  wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
