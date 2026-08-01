#!/usr/bin/env python3
"""Figure 14: multi-seed accuracy with 95% intervals, as a point-and-interval plot.

One row per configuration: the tuned statistical ensemble, the best local
language-model judge, every gated hybrid, and the statistical judge. The
statistical judge's interval is drawn as a shaded band across the plot so the
separation, or the lack of it, is readable directly.

Three interval families are available through --ci, because they answer different
questions and can disagree about whether a configuration clears the statistical
judge:

  wilson     (default) Wilson score interval on the pooled successes/trials.
             Bounded to [0, 1] by construction. This is the primary method.
  bootstrap  Seeded percentile bootstrap over the per-seed accuracies. Bounded
             because the resampled values are. Reported as the cross-check.
  wald       mean +/- t(n-1, 0.975) * sd / sqrt(n) over the per-seed accuracies.
             Superseded: it can fall outside [0, 1] on a rate near 0 or 1.
             Computed here from agg/long_results.csv, since no released table
             carries it, and offered only so the older figure can be reproduced.

Configurations whose per-seed accuracy is bit-identical across every seed are
flagged `identical_across_seeds` in metrics_with_ci.csv. Under Wilson their
interval is not zero-width, so the flat rate would otherwise become invisible.
They are drawn with an open circle rather than a filled one, with its own legend
entry, so a structural zero-variance result stays legible whichever family is
selected.

Needs matplotlib. On a host without it, run inside the judge-service image:

    docker run --rm -v "$PWD":/work -w /work canaryllm-judge-service \
        python tools/figure_multiseed_intervals.py

Usage:
  python tools/figure_multiseed_intervals.py [--ci wilson|bootstrap|wald]
                                             [--results results] [--out results/figures]
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "tools"))
import ci_utils as ci  # noqa: E402

ENSEMBLE_COLOUR = "#2c7a37"
BEST_LOCAL_COLOUR = "#d1791b"
HYBRID_COLOUR = "#1f4e9c"
STAT_COLOUR = "#000000"
BAND_COLOUR = "#c0392b"

# Student t, two-sided 95%, by degrees of freedom. Same table as
# the retired multi-seed runner's _T_TABLE, so --ci wald reproduces the
# interval that script wrote before it was superseded.
_T_TABLE = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
            7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}


def read_csv(path: Path):
    if not path.exists():
        return []
    with path.open() as f:
        return list(csv.DictReader(f))


def pretty(judge: str, model: str) -> str:
    m = (model or "").replace("-llm", "").replace("-vlm", "")
    names = {"deepseek-r1": "DeepSeek-R1", "mistral-nemo": "Mistral-Nemo",
             "phi4": "Phi-4", "qwen": "Qwen2.5", "olmo2": "OLMo 2",
             "moondream": "Moondream", "minicpm-v": "MiniCPM-V",
             "granite-vision": "Granite Vision"}
    m = names.get(m, m)
    if judge == "ensemble:tuned":
        return "ensemble (tuned, no model calls)"
    if judge == "statistical":
        return "statistical judge"
    if judge.startswith("ai:"):
        return f"best local LLM ({judge} / {m})"
    return f"hybrid:gated ({m})"


def per_seed_accuracies(long_rows, judge, model):
    by_seed = defaultdict(list)
    for r in long_rows:
        if r["judge"] == judge and r["model"] == model:
            by_seed[int(r["seed"])].append(r)
    out = []
    for seed in sorted(by_seed):
        c = ci.confusion_counts(by_seed[seed])
        out.append(ci.metrics_from_conf(c)["accuracy"])
    return out


def wald(values):
    n = len(values)
    if n < 2:
        return (values[0], values[0]) if values else (0.0, 0.0)
    mean = sum(values) / n
    sd = (sum((v - mean) ** 2 for v in values) / (n - 1)) ** 0.5
    half = _T_TABLE.get(n - 1, 1.96) * sd / (n ** 0.5)
    return (mean - half, mean + half)


def collect(results: Path, ci_kind: str):
    """(label, accuracy, low, high, kind, identical) per configuration."""
    agg = results / "agg"
    metrics = read_csv(agg / "metrics_with_ci.csv")
    ens = read_csv(agg / "ensemble_multiseed_ci.csv")
    long_rows = read_csv(agg / "long_results.csv") if ci_kind == "wald" else []

    def estimate(row, judge, model):
        """(point, low, high) for one configuration.

        Wilson and the bootstrap are both centred on the pooled accuracy that
        metrics_with_ci.csv records, so the point estimate is that column. The
        Wald interval is centred on the MEAN OF THE PER-SEED accuracies, which is
        not the same number whenever the seeds carry unequal scenario counts, so
        it has to bring its own point estimate or the whisker comes out
        asymmetric around the wrong centre.
        """
        if ci_kind == "wilson":
            return (float(row["accuracy_mean"]),
                    float(row["accuracy_ci_low"]), float(row["accuracy_ci_high"]))
        if ci_kind == "bootstrap":
            return (float(row["accuracy_mean"]),
                    float(row["accuracy_boot_low"]), float(row["accuracy_boot_high"]))
        vals = per_seed_accuracies(long_rows, judge, model)
        if not vals:
            raise SystemExit(
                f"--ci wald needs per-seed rows for {judge}/{model} in "
                f"{agg / 'long_results.csv'}")
        lo, hi = wald(vals)
        return sum(vals) / len(vals), lo, hi

    rows = []
    for r in ens:
        if r["judge"] != "ensemble:tuned":
            continue
        # The ensemble table carries no per-seed rows in long_results.csv, so a
        # Wald interval is not derivable for it; fall back to its Wilson columns
        # and say so, rather than dropping the row from the figure.
        lo, hi = ((float(r["accuracy_boot_low"]), float(r["accuracy_boot_high"]))
                  if ci_kind == "bootstrap"
                  else (float(r["accuracy_ci_low"]), float(r["accuracy_ci_high"])))
        rows.append((pretty(r["judge"], ""), float(r["accuracy_mean"]), lo, hi,
                     "ensemble", "accuracy" in (r.get("identical_across_seeds") or "")))

    ai = [r for r in metrics if r["judge"].startswith("ai:")]
    if ai:
        best = max(ai, key=lambda r: float(r["accuracy_mean"]))
        acc, lo, hi = estimate(best, best["judge"], best["model"])
        rows.append((pretty(best["judge"], best["model"]), acc, lo, hi, "best_local",
                     "accuracy" in (best.get("identical_across_seeds") or "")))

    for r in metrics:
        if r["judge"] != "hybrid:gated":
            continue
        acc, lo, hi = estimate(r, r["judge"], r["model"])
        rows.append((pretty(r["judge"], r["model"]), acc, lo, hi, "hybrid",
                     "accuracy" in (r.get("identical_across_seeds") or "")))

    stat = next((r for r in metrics if r["judge"] == "statistical"), None)
    if stat is None:
        raise SystemExit("no statistical row in metrics_with_ci.csv")
    acc, lo, hi = estimate(stat, "statistical", stat["model"])
    stat_row = (pretty("statistical", ""), acc, lo, hi,
                "statistical", "accuracy" in (stat.get("identical_across_seeds") or ""))

    rows.sort(key=lambda t: t[1], reverse=True)
    return rows + [stat_row]


def _pooled_n(results: Path):
    """n_pooled from metrics_with_ci.csv, so the axis label states the sample size
    rather than hardcoding the published run's."""
    rows = read_csv(results / "agg" / "metrics_with_ci.csv")
    vals = {r.get("n_pooled") for r in rows if r.get("n_pooled")}
    return vals.pop() if len(vals) == 1 else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ci", choices=("wilson", "bootstrap", "wald"), default="wilson",
                    help="interval family (default wilson, the primary method)")
    ap.add_argument("--results", default=str(REPO_ROOT / "results"))
    ap.add_argument("--out", default=None,
                    help="output directory (default: <results>/figures)")
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args()

    results = Path(args.results)
    rows = collect(results, args.ci)
    stat = rows[-1]

    colours = {"ensemble": ENSEMBLE_COLOUR, "best_local": BEST_LOCAL_COLOUR,
               "hybrid": HYBRID_COLOUR, "statistical": STAT_COLOUR}

    fig, ax = plt.subplots(figsize=(6.000, 3.808), dpi=args.dpi)
    ys = list(range(len(rows)))[::-1]

    ax.axvspan(stat[2], stat[3], color=BAND_COLOUR, alpha=0.10, lw=0, zorder=0)
    ax.axvline(stat[3], color=BAND_COLOUR, ls="--", lw=1.2, zorder=1)

    for y, (label, acc, lo, hi, kind, identical) in zip(ys, rows):
        colour = colours[kind]
        ax.errorbar(acc, y, xerr=[[acc - lo], [hi - acc]], fmt="none",
                    ecolor=colour, elinewidth=2.0, capsize=4, capthick=2.0, zorder=3)
        ax.plot([acc], [y], marker="o", ms=7, zorder=4,
                color="white" if identical else colour,
                markeredgecolor=colour, markeredgewidth=2.0)

    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in rows], fontsize=8)
    n_pooled = _pooled_n(results)
    ax.set_xlabel("accuracy" + (f", pooled over {n_pooled} observations" if n_pooled else ""),
                  fontsize=9)
    ax.grid(True, axis="x", alpha=0.3)
    ax.set_axisbelow(True)

    # Headroom on the right so the legend has somewhere to sit that is not on top
    # of a bar.
    data_lo = min(r[2] for r in rows)
    data_hi = max(r[3] for r in rows)
    span = data_hi - data_lo
    ax.set_xlim(data_lo - 0.04 * span, data_hi + 0.06 * span)

    # Band annotation inside the plot area with a leader, never on the axis where
    # it would collide with the tick labels. The band's own column is empty on
    # every row but the statistical one, so the text goes there.
    ax.annotate(f"statistical judge 95% CI\n[{stat[2]:.3f}, {stat[3]:.3f}]",
                xy=(stat[3], ys[1]), xycoords="data",
                xytext=(stat[2], ys[0] + 0.15),
                textcoords="data", fontsize=7, color=BAND_COLOUR,
                ha="left", va="center",
                arrowprops=dict(arrowstyle="-", color=BAND_COLOUR, lw=0.7,
                                shrinkA=2, shrinkB=2))

    handles = [
        plt.Line2D([], [], color=ENSEMBLE_COLOUR, marker="o", lw=2,
                   label="tuned statistical ensemble"),
        plt.Line2D([], [], color=BEST_LOCAL_COLOUR, marker="o", lw=2,
                   label="best local language-model judge"),
        plt.Line2D([], [], color=HYBRID_COLOUR, marker="o", lw=2,
                   label="gated hybrid"),
        plt.Line2D([], [], color=STAT_COLOUR, marker="o", lw=2,
                   label="statistical judge"),
        plt.Line2D([], [], color="grey", marker="o", lw=0, mfc="white",
                   mec="grey", mew=2, label="identical across all seeds"),
        plt.Rectangle((0, 0), 1, 1, color=BAND_COLOUR, alpha=0.10,
                      label="statistical judge CI"),
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=6, framealpha=0.95,
              borderpad=0.4, labelspacing=0.35, handlelength=1.6)
    fig.tight_layout()

    out_dir = Path(args.out) if args.out else results / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"fig14_multiseed_intervals_{args.ci}.png"
    fig.savefig(path)
    plt.close(fig)

    print(f"[figure-14] interval family: {args.ci}")
    overlap = []
    for label, acc, lo, hi, kind, identical in rows:
        flag = " (identical across seeds)" if identical else ""
        mark = ""
        if kind != "statistical" and lo < stat[3]:
            mark = "  OVERLAPS the statistical judge"
            overlap.append(label)
        print(f"  {label:38} acc={acc:.3f}  [{lo:.3f}, {hi:.3f}]{flag}{mark}")
    print(f"  statistical upper bound = {stat[3]:.3f}; "
          f"{len(overlap)} configuration(s) overlap it")
    print(f"  wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
