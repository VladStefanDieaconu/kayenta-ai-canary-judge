#!/usr/bin/env python3
"""Figure 12: one instance of each added family, drawn the way the judge sees it.

The point of the figure is that the three families fail in three different
places in the window, and that the third does not differ from its control in any
whole-window statistic at all. The sustained_excursion panel therefore carries
its equal-marginals annotation directly, because a reader who does not notice
that the two series are permutations of each other will read it as noise.

Usage:
  python tools/figure_new_families.py [--n 20] [--seed 20260621]
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "tools"))

import container_plot  # noqa: E402
import eval_dataset as ed  # noqa: E402

PLOT = r"""
fams = spec["families"]
fig, axes = plt.subplots(len(fams), 1, figsize=(9.5, 3.5 * len(fams)), squeeze=False)
for i, f in enumerate(fams):
    ax = axes[i][0]
    x = list(range(len(f["control"])))
    ax.plot(x, f["control"], color="#1f77b4", marker="o", markersize=3, lw=1.4, label="Control")
    ax.plot(x, f["experiment"], color="#d62728", marker="o", markersize=3, lw=1.4, label="Canary")
    # The title goes above via set_title, which matplotlib places correctly, and
    # the note goes below the x-axis. Stacking both above the axes by hand needs
    # the note's rendered height in axes coordinates, which is not known before
    # layout, and every estimate of it overlapped the title.
    ax.set_title(f["family"] + "   -   benchmark label: " + f["truth"],
                 loc="left", fontsize=12.5, fontweight="bold", pad=8)
    ax.text(0.0, -0.30, f["note"], transform=ax.transAxes, ha="left", va="top",
            fontsize=9, color="#444444", linespacing=1.4)
    ax.set_xlabel("sample index (1 min steps)")
    ax.set_ylabel(f["unit"])
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9, framealpha=0.9)
    ax.margins(y=0.12)
fig.suptitle("generalisation-60: three held-out scenario families", fontsize=14, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.975])
fig.subplots_adjust(hspace=0.95)
save(fig, spec.get("outfile", "fig12_new_families.png"), dpi=spec.get("dpi", 300))
"""


def main() -> int:
    ap = argparse.ArgumentParser(description="Draw one instance of each new family")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=ed.DEFAULT_MASTER_SEED)
    ap.add_argument("--index", type=int, default=0, help="which instance of each family")
    ap.add_argument("--dpi", type=int, default=300,
                    help="output resolution; the figure size is fixed, so this "
                         "scales the pixels without changing the aspect ratio")
    ap.add_argument("--outfile", default="fig12_new_families.png")
    ap.add_argument("--out", default=None,
                    help="output directory (default: results/figures)")
    args = ap.parse_args()

    scenarios = ed.build_scenarios(args.seed, args.n, "generalisation-60")
    chosen = {}
    for sc in scenarios:
        if sc.id.endswith(f"_{args.index:03d}"):
            chosen[sc.family] = sc

    panels = []
    for family in ed.GENERALISATION_FAMILIES:
        sc = chosen[family]
        series = ed.series_for_scenario(sc)
        name, d = next(iter(series.items()))
        c, e = d["control"], d["experiment"]
        p = sc.params[name]
        late = max(1, len(e) // 4)
        end_ratio = st.mean(e[-late:]) / st.mean(c)

        if family == "partial_recovery":
            note = (f"degrades to {p['degradation_factor']:.2f}x, recovers, but settles "
                    f"{(p['residual_ratio']-1)*100:.0f}% above control\n"
                    f"end-of-window ratio {end_ratio:.2f} -- it did NOT return to control levels")
        elif family == "late_transient":
            note = (f"healthy for the first {p['onset_fraction']*100:.0f}% of the window, then a step to "
                    f"{p['step_magnitude']:.2f}x\nstill degraded at the last sample "
                    f"(end-of-window ratio {end_ratio:.2f})")
        else:
            note = (f"canary is a PERMUTATION of the control: mean, stddev, min, max and every\n"
                    f"percentile are identical. {p['n_elevated']} elevated samples on both sides -- "
                    f"isolated on the\ncontrol, gathered into {p['n_runs']} runs of "
                    f"{p['run_length']} on the canary. No rubric names this shape.")

        panels.append({
            "family": family, "truth": sc.truth, "scenario": sc.id,
            "control": [round(v, 5) for v in c], "experiment": [round(v, 5) for v in e],
            "unit": "latency (s)", "note": note,
        })

    info = container_plot.render(PLOT, {"families": panels, "dpi": args.dpi,
                                        "outfile": args.outfile}, out_dir=args.out)
    print(container_plot.report(info))
    print(json.dumps(info, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
