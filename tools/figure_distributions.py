#!/usr/bin/env python3
"""Figure 2: location shift versus equal-median shape change, as two density panels.

Both panels are drawn from the shipped dataset generator (tools/eval_dataset.py),
not from hand-chosen constants, so the picture describes the benchmark a reader
actually runs. The left panel takes one `clean_mean_shift` instance, where the
experiment median moves and the rank test detects it. The right panel takes one
`variance_increase` instance, where the medians coincide and only the spread
changes, which a rank test on ranks cannot see.

Each curve is the normal density implied by the generated array's own mean and
standard deviation. `eval_dataset._noisy` draws Gaussian samples, so that density
is the distribution the sample came from rather than a smoothing choice.

Needs matplotlib. On a host without it, run inside the judge-service image:

    docker run --rm -v "$PWD":/work -w /work canaryllm-judge-service \
        python tools/figure_distributions.py

Usage:
  python tools/figure_distributions.py [--seed 20260621] [--out results/figures]
"""

from __future__ import annotations

import argparse
import math
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


def instance(family: str, seed: int, n_per_family: int, index: int = 0):
    """The (control, experiment) arrays of one instance of `family`.

    Selection is by scenario id, not by Scenario.gid, which is a global index
    across all families rather than a per-family one. `n_per_family` matters:
    each instance's RNG seed is derived from its global index, so the same
    family instance carries different values at n=12 and n=20.
    """
    wanted = f"{family}_{index:03d}"
    for sc in ed.build_scenarios(seed, n_per_family):
        if sc.id == wanted:
            series = ed.series_for_scenario(sc)
            name = sc.metrics[0].name
            return series[name]["control"], series[name]["experiment"], sc
    raise SystemExit(f"no scenario {wanted} at seed {seed}, n={n_per_family}")


def moments(xs):
    n = len(xs)
    mean = sum(xs) / n
    sd = (sum((x - mean) ** 2 for x in xs) / max(1, n - 1)) ** 0.5
    return mean, sd


def normal_density(xs, mean, sd):
    k = 1.0 / (sd * math.sqrt(2 * math.pi))
    return [k * math.exp(-0.5 * ((x - mean) / sd) ** 2) for x in xs]


def panel(ax, control, experiment, title, unit):
    c_mean, c_sd = moments(control)
    e_mean, e_sd = moments(experiment)
    lo = min(c_mean - 4 * c_sd, e_mean - 4 * e_sd)
    hi = max(c_mean + 4 * c_sd, e_mean + 4 * e_sd)
    grid = [lo + (hi - lo) * i / 400.0 for i in range(401)]

    ax.plot(grid, normal_density(grid, c_mean, c_sd), color=CONTROL_COLOUR,
            lw=2, label="control")
    ax.plot(grid, normal_density(grid, e_mean, e_sd), color=EXPERIMENT_COLOUR,
            lw=2, label="experiment")
    ax.fill_between(grid, normal_density(grid, e_mean, e_sd),
                    color=EXPERIMENT_COLOUR, alpha=0.12)
    ax.set_title(title)
    ax.set_xlabel(f"metric value ({unit})")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    return c_mean, c_sd, e_mean, e_sd


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seed", type=int, default=ed.DEFAULT_MASTER_SEED,
                    help="dataset master seed (default: the published seed)")
    ap.add_argument("--n", type=int, default=20,
                    help="instances per family; each instance's RNG seed is "
                         "derived from its global index, so this changes the "
                         "values (default 20, the published run)")
    ap.add_argument("--results", default=str(REPO_ROOT / "results"),
                    help="accepted for a uniform generator interface; these two "
                         "figures are generated from the seed, not from results/")
    ap.add_argument("--out", default=str(REPO_ROOT / "results" / "figures"),
                    help="directory to write the PNG into")
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args()

    shift_c, shift_e, shift_sc = instance("clean_mean_shift", args.seed, args.n)
    var_c, var_e, var_sc = instance("variance_increase", args.seed, args.n)

    fig, axes = plt.subplots(1, 2, figsize=(6.957, 2.630), dpi=args.dpi)
    a = panel(axes[0], shift_c, shift_e,
              "Location shift (detected)", shift_sc.metrics[0].unit)
    b = panel(axes[1], var_c, var_e,
              "Shape change: same mean (missed)", var_sc.metrics[0].unit)
    axes[0].set_ylabel("density")
    fig.tight_layout()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "fig02_distributions.png"
    fig.savefig(path)
    plt.close(fig)

    print(f"[figure-02] seed={args.seed} n_per_family={args.n}")
    print(f"  clean_mean_shift  {shift_sc.id}: control mean={a[0]:.4f} sd={a[1]:.4f} "
          f"| experiment mean={a[2]:.4f} sd={a[3]:.4f} "
          f"| mean ratio {a[2] / a[0]:.3f}, sd ratio {a[3] / a[1]:.2f}")
    print(f"  variance_increase {var_sc.id}: control mean={b[0]:.4f} sd={b[1]:.4f} "
          f"| experiment mean={b[2]:.4f} sd={b[3]:.4f} "
          f"| mean ratio {b[2] / b[0]:.3f}, sd ratio {b[3] / b[1]:.2f}")
    print(f"  wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
