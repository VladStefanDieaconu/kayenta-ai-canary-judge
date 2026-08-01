#!/usr/bin/env python3
"""The prompt ablation, per family, per model: frozen vs v2 vs v3.

The figure has to answer two questions at a glance -- did the variant repair the
healed transient, and what did it cost elsewhere -- so it shows both directly.
Each model gets a row of grouped bars over the families, the healed-transient
column is shaded because it is the case the variants were written for, and every
family whose accuracy moved is annotated with the signed change against the
frozen rubric. A reader should not have to subtract two bar heights by eye to
find the cost.

Selection runs through results_schema.load(); this file does not know which run
wrote which row.

Usage:
  python tools/figure_prompt_comparison.py [--dataset original-180]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "tools"))

import container_plot  # noqa: E402
import eval_dataset as ed  # noqa: E402
import results_schema as rs  # noqa: E402

FROZEN = "v1-frozen-2026-06"
PROMPT_ORDER = [FROZEN, "v2-operational-2026-07", "v3-temporal-2026-07"]
# Display names and a palette for the rubrics this study ablated. Neither
# decides which rubrics appear -- that comes from the frame -- so another
# rubric set renders under its own ids in the same colours.
LABEL = {FROZEN: "v1 frozen", "v2-operational-2026-07": "v2 operational",
         "v3-temporal-2026-07": "v3 temporal"}
COLOUR = {FROZEN: "#4c72b0", "v2-operational-2026-07": "#dd8452",
          "v3-temporal-2026-07": "#55a868"}
PALETTE = ["#4c72b0", "#dd8452", "#55a868", "#c44e52", "#8172b3", "#937860"]


def prompts_in_frame(dataset, representation, experiments):
    """Rubric ids present, the study's three first and any others after.

    PROMPT_ORDER is this study's rubric set. Filtering to it alone means a frame
    written with different rubric ids produces an empty figure and no
    explanation, which is the one thing a figure script must not do to someone
    else's data.
    """
    found = {r["prompt_id"] for r in rs.load(experiments, dataset_id=dataset,
                                             judge="ai", representation=representation)
             if r["prompt_id"]}
    known = [p for p in PROMPT_ORDER if p in found]
    return known + sorted(found - set(PROMPT_ORDER))

PLOT = r"""
models = spec["models"]
families = spec["families"]
prompts = spec["prompts"]
labels = spec["labels"]
colours = spec["colours"]
highlight = set(spec["highlight"])
truth = spec["truth"]

nrows = len(models)
fig, axes = plt.subplots(nrows, 1, figsize=(max(11, 1.25 * len(families)), 3.5 * nrows),
                         squeeze=False, sharex=True)
x = np.arange(len(families))
width = 0.8 / max(1, len(prompts))

for mi, m in enumerate(models):
    ax = axes[mi][0]
    for fi, fam in enumerate(families):
        if fam in highlight:
            ax.axvspan(fi - 0.5, fi + 0.5, color="#ffe9b8", zorder=0)
    for pi, p in enumerate(prompts):
        vals = [m["family"].get(p, {}).get(f) for f in families]
        pos = x - 0.4 + width * (pi + 0.5)
        ax.bar(pos, [0 if v is None else v for v in vals], width * 0.92,
               label=labels[p], color=colours[p], zorder=3,
               edgecolor="white", linewidth=0.6)
    # signed change vs the frozen rubric, printed only where it is non-zero
    for fi, fam in enumerate(families):
        base = m["family"].get(spec["frozen"], {}).get(fam)
        if base is None:
            continue
        for pi, p in enumerate(prompts):
            if p == spec["frozen"]:
                continue
            v = m["family"].get(p, {}).get(fam)
            if v is None or abs(v - base) < 1e-9:
                continue
            d = v - base
            pos = x[fi] - 0.4 + width * (pi + 0.5)
            ax.annotate(("+" if d > 0 else "") + f"{d:.2f}",
                        xy=(pos, v), xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=7.6, zorder=4,
                        color="#12693a" if d > 0 else "#a01722", fontweight="bold")
    ax.set_ylim(0, 1.19)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_ylabel("accuracy")
    ax.grid(True, axis="y", alpha=0.28, zorder=0)
    ax.set_axisbelow(True)
    # n is printed per prompt, not once for the panel: an arm scored on fewer
    # scenarios than another must not borrow the other's denominator in the label.
    head = "   ".join(f"{labels[p]} {m['overall'][p]:.3f} (n={m['n_by_prompt'][p]})"
                      for p in prompts if p in m["overall"])
    ax.set_title(f"{m['model']}      overall accuracy:  {head}",
                 loc="left", fontsize=11.5, fontweight="bold")
    if mi == 0:
        ax.legend(loc="upper right", ncol=len(prompts), fontsize=9, framealpha=0.95)

ax = axes[-1][0]
ax.set_xticks(x)
ax.set_xticklabels([f + ("\n(truth PASS)" if truth[f] == "PASS" else "") for f in families],
                   rotation=32, ha="right", fontsize=9)
for tick, f in zip(ax.get_xticklabels(), families):
    if f in highlight:
        tick.set_fontweight("bold")

fig.suptitle(spec["title"], fontsize=13.5, fontweight="bold")
fig.text(0.5, 0.005, spec["footer"], ha="center", fontsize=8.8, color="#444444")
fig.tight_layout(rect=[0, 0.022, 1, 0.985])
save(fig, spec["outfile"], dpi=200)
"""


def build(dataset: str, representation: str, outfile: str, title: str,
          highlight: List[str], footer: str, min_rubrics: int = 2,
          experiments: Optional[Path] = None,
          requested_models: Optional[List[str]] = None) -> Dict[str, Any]:
    models = sorted({r["model"] for r in rs.load(experiments, dataset_id=dataset, judge="ai")})
    if requested_models:
        missing = [m for m in requested_models if m not in models]
        if missing:
            raise SystemExit(
                f"[figure] requested model(s) not in the frame: {', '.join(missing)}\n"
                f"         the frame contains: {', '.join(models) or '(nothing)'}\n"
                f"         dataset={dataset}")
        models = [m for m in requested_models]
    prompts = prompts_in_frame(dataset, representation, experiments)
    families = sorted({r["family"] for r in rs.load(experiments, dataset_id=dataset, judge="ai")},
                      key=ed.family_order)
    truth = {}
    for r in rs.load(experiments, dataset_id=dataset, judge="ai"):
        truth[r["family"]] = r["truth_label"]

    out_models = []
    for model in models:
        fam: Dict[str, Dict[str, float]] = {}
        overall: Dict[str, float] = {}
        n_by_prompt: Dict[str, int] = {}
        n = 0
        for p in prompts:
            rows = rs.load(experiments, dataset_id=dataset, judge="ai",
                           representation=representation, model=model, prompt_id=p)
            if not rows:
                continue
            n = max(n, len(rows))
            n_by_prompt[p] = len(rows)
            overall[p] = sum(r["correct"] for r in rows) / len(rows)
            per: Dict[str, float] = {}
            for f in families:
                sub = [r for r in rows if r["family"] == f]
                if sub:
                    per[f] = sum(r["correct"] for r in sub) / len(sub)
            fam[p] = per
        # A model that ran only one rubric was not ablated, and a panel with a
        # single bar per family reads as a result about that model rather than
        # as a missing arm. The local models entered this frame after the figure
        # was first drawn, under the frozen rubric only; without this guard the
        # figure silently grew from three panels to seven.
        if len(overall) >= min_rubrics:
            out_models.append({"model": model, "family": fam, "overall": overall,
                               "n": n, "n_by_prompt": n_by_prompt})

    # Labels and colours cover whatever rubrics the frame actually holds.
    labels = {p: LABEL.get(p, p) for p in prompts}
    colours = {p: COLOUR.get(p, PALETTE[i % len(PALETTE)]) for i, p in enumerate(prompts)}
    return {"models": out_models, "families": families, "prompts": prompts,
            "labels": labels, "colours": colours, "frozen": prompts[0] if prompts else FROZEN,
            "highlight": highlight, "truth": truth, "title": title,
            "footer": footer, "outfile": outfile}


def main() -> int:
    ap = argparse.ArgumentParser(description="Prompt comparison figure")
    ap.add_argument("--dataset", default="original-180")
    ap.add_argument("--representation", default="raw")
    ap.add_argument("--outfile", default="")
    ap.add_argument("--min-rubrics", type=int, default=2,
                    help="drop models that ran fewer rubrics than this "
                         "(default 2: an ablation figure shows ablated models)")
    ap.add_argument("--models", default="",
                    help="comma-separated aliases to draw (default: every model in the frame)")
    ap.add_argument("--out", default=None,
                    help="output directory (default: results/figures)")
    ap.add_argument("--experiments", default=None,
                    help="directory of long-format result CSVs "
                         "(default: results/agg/experiments)")
    args = ap.parse_args()
    experiments = Path(args.experiments) if args.experiments else None

    if args.dataset == "original-180":
        outfile = args.outfile or "prompt_comparison_original180.png"
        title = "Prompt ablation on the original 180 scenarios (ai:raw, three hosted models)"
        highlight = ["healed_transient"]
        footer = ("Shaded column: healed_transient, the false-failure the variants were written to repair. "
                  "Annotations give the signed change against the frozen rubric; families with no annotation did not move.")
    else:
        outfile = args.outfile or "prompt_comparison_generalisation60.png"
        title = "Generalisation: the three held-out families (ai:raw, three hosted models)"
        highlight = ["sustained_excursion"]
        footer = ("Shaded column: sustained_excursion, the held-out family no rubric names. "
                  "All three families are labelled FAIL, so accuracy here is recall.")

    spec = build(args.dataset, args.representation, outfile, title, highlight,
                 footer, args.min_rubrics, experiments,
                 [m.strip() for m in args.models.split(",") if m.strip()])
    if not spec["prompts"]:
        print(f"[figure] no rubric ids in the frame for dataset={args.dataset} "
              f"representation={args.representation}", file=sys.stderr)
        return 1
    if not spec["models"]:
        print(f"[figure] no model in the frame ran at least {args.min_rubrics} rubrics "
              f"for dataset={args.dataset}; rubrics present: "
              f"{', '.join(spec['prompts'])}", file=sys.stderr)
        return 1
    info = container_plot.render(PLOT, spec, out_dir=args.out)
    print(container_plot.report(info))
    print(json.dumps(info, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
