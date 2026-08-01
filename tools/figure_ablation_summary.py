#!/usr/bin/env python3
"""Section 7.9 in one panel: the three rubrics against the three hosted models.

Section 7.9 carries the newest result in the paper and has no figure. Tables 15
and 16 hold it as eighteen numbers, and the finding -- that the same sentence
repairs one model completely, one partially and one not at all -- is a shape
rather than a value, so it belongs in a figure.

Three panels, left to right:
  (a) overall accuracy over all nine families, per model per rubric;
  (b) the healed-transient family alone, which is what the variants were
      written for. That family's benchmark label is PASS, so the rate is the
      proportion correctly passed -- specificity, not recall;
  (c) the paired comparison of each variant against the frozen rubric: how many
      verdicts it fixed and how many it broke, over the same 180 scenarios.

`figure_prompt_comparison.py` already draws the full nine-family breakdown at
three panels of nine grouped bars. This is the summary a reader meets first.

Selection runs through results_schema.load(); the plotting code receives values.

Usage:
  python tools/figure_ablation_summary.py
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
import results_schema as rs  # noqa: E402

FROZEN = "v1-frozen-2026-06"
PROMPT_ORDER = [FROZEN, "v2-operational-2026-07", "v3-temporal-2026-07"]
# Same labels and palette as figure_prompt_comparison.py, so the two figures
# read as one pair.
LABEL = {FROZEN: "v1 frozen", "v2-operational-2026-07": "v2 operational",
         "v3-temporal-2026-07": "v3 temporal"}
COLOUR = {FROZEN: "#4c72b0", "v2-operational-2026-07": "#dd8452",
          "v3-temporal-2026-07": "#55a868"}
DISPLAY = {"claude-opus-4-8-vlm": "Claude Opus 4.8",
           "gpt-oss-120b-llm": "gpt-oss-120b",
           "qwen3-vl-235b-vlm": "Qwen3-VL 235B"}
MODELS = list(DISPLAY)
TARGET_FAMILY = "healed_transient"

PLOT = r"""
models = spec["models"]
prompts = spec["prompts"]
labels = spec["labels"]
colours = spec["colours"]

fig, axes = plt.subplots(1, 3, figsize=(15.0, 5.6),
                         gridspec_kw={"width_ratios": [1.0, 1.0, 1.05]})
x = np.arange(len(models))
width = 0.8 / len(prompts)


def grouped(ax, key, title, ylabel):
    for pi, p in enumerate(prompts):
        vals = [m[key].get(p) for m in models]
        pos = x - 0.4 + width * (pi + 0.5)
        ax.bar(pos, [0 if v is None else v for v in vals], width * 0.9,
               label=labels[p], color=colours[p], zorder=3,
               edgecolor="white", linewidth=0.7)
        for xi, v in zip(pos, vals):
            if v is None:
                continue
            ax.annotate(f"{v:.3f}", xy=(xi, v), xytext=(0, 3),
                        textcoords="offset points", ha="center", va="bottom",
                        fontsize=8.2, zorder=4)
    ax.set_ylim(0, 1.16)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels([m["label"] for m in models], fontsize=9.5)
    ax.grid(True, axis="y", alpha=0.28, zorder=0)
    ax.set_axisbelow(True)
    ax.set_title(title, loc="left", fontsize=11.5, fontweight="bold")


grouped(axes[0], "accuracy", "(a)  overall accuracy, nine families",
        "accuracy   (n = 180)")
axes[0].legend(loc="upper center", ncol=3, fontsize=8.6, framealpha=0.95,
               bbox_to_anchor=(0.5, 1.0))

grouped(axes[1], "target", "(b)  healed_transient only",
        "correctly passed   (n = 20)")

# (c) the paired comparison. Fixed above the axis, broken below, so the sign of
# the net effect is the visible thing rather than something to compute.
ax = axes[2]
variants = spec["variants"]
vw = 0.8 / len(variants)
for vi, v in enumerate(variants):
    pos = x - 0.4 + vw * (vi + 0.5)
    fixed = [m["paired"][v]["fixed"] for m in models]
    broken = [-m["paired"][v]["broken"] for m in models]
    ax.bar(pos, fixed, vw * 0.9, color=colours[v], zorder=3,
           edgecolor="white", linewidth=0.7, label=labels[v])
    ax.bar(pos, broken, vw * 0.9, color=colours[v], zorder=3, alpha=0.45,
           hatch="///", edgecolor="white", linewidth=0.7)
    for xi, f, b, m in zip(pos, fixed, broken, models):
        if f:
            ax.annotate(f"+{f}", xy=(xi, f), xytext=(0, 3),
                        textcoords="offset points", ha="center", va="bottom",
                        fontsize=8.6, fontweight="bold", color="#12693a", zorder=4)
        if b:
            ax.annotate(f"{b}", xy=(xi, b), xytext=(0, -4),
                        textcoords="offset points", ha="center", va="top",
                        fontsize=8.6, fontweight="bold", color="#a01722", zorder=4)
        star = m["paired"][v]["stars"]
        if star:
            ax.annotate(star, xy=(xi, max(f, 0) + 2.6), ha="center", va="bottom",
                        fontsize=10.5, zorder=4, color="#333333")
ax.axhline(0, color="#333333", lw=1.0, zorder=4)
ax.set_ylim(-9, 26)
ax.set_ylabel("verdicts fixed  (above)   /   broken  (below)")
ax.set_xticks(x)
ax.set_xticklabels([m["label"] for m in models], fontsize=9.5)
ax.grid(True, axis="y", alpha=0.28, zorder=0)
ax.set_axisbelow(True)
ax.set_title("(c)  paired against the frozen rubric", loc="left",
             fontsize=11.5, fontweight="bold")

fig.suptitle(spec["title"], fontsize=13.5, fontweight="bold")
fig.text(0.5, 0.012, spec["footer"], ha="center", fontsize=8.6, color="#444444")
fig.tight_layout(rect=[0, 0.135, 1, 0.955])
save(fig, spec["outfile"], dpi=spec.get("dpi", 200))
"""


def stars(p: float) -> str:
    """Holm-corrected significance, as marked in Table 16."""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def mcnemar_p(fixed: int, broken: int) -> float:
    import math
    nd = fixed + broken
    if nd == 0:
        return 1.0
    k = min(fixed, broken)
    return min(1.0, 2.0 * sum(math.comb(nd, i) * 0.5 ** nd for i in range(k + 1)))


def build(dataset: str, representation: str, outfile: str,
          experiments: Optional[Path] = None) -> Dict[str, Any]:
    variants = [p for p in PROMPT_ORDER if p != FROZEN]
    out_models: List[Dict[str, Any]] = []
    raw_p: Dict[tuple, float] = {}

    per_model_rows = {}
    for model in MODELS:
        per_prompt = {}
        for p in PROMPT_ORDER:
            per_prompt[p] = rs.load(experiments, dataset_id=dataset, judge="ai",
                                    representation=representation,
                                    model=model, prompt_id=p)
        per_model_rows[model] = per_prompt
        for v in variants:
            a = {r["scenario_id"]: r for r in per_prompt[FROZEN]}
            b = {r["scenario_id"]: r for r in per_prompt[v]}
            shared = set(a) & set(b)
            fixed = sum(1 for s in shared if a[s]["correct"] != 1 and b[s]["correct"] == 1)
            broken = sum(1 for s in shared if a[s]["correct"] == 1 and b[s]["correct"] != 1)
            raw_p[(model, v)] = mcnemar_p(fixed, broken)

    # Holm across all six comparisons, exactly as Table 16 does it.
    order = sorted(raw_p, key=lambda k: raw_p[k])
    holm: Dict[tuple, float] = {}
    running = 0.0
    for i, k in enumerate(order):
        running = max(running, min(1.0, raw_p[k] * (len(order) - i)))
        holm[k] = running

    for model in MODELS:
        per_prompt = per_model_rows[model]
        acc, target, n_by, paired = {}, {}, {}, {}
        for p in PROMPT_ORDER:
            rows = per_prompt[p]
            if not rows:
                continue
            n_by[p] = len(rows)
            acc[p] = sum(r["correct"] for r in rows) / len(rows)
            sub = [r for r in rows if r["family"] == TARGET_FAMILY]
            target[p] = (sum(r["correct"] for r in sub) / len(sub)) if sub else None
        for v in variants:
            a = {r["scenario_id"]: r for r in per_prompt[FROZEN]}
            b = {r["scenario_id"]: r for r in per_prompt[v]}
            shared = set(a) & set(b)
            paired[v] = {
                "fixed": sum(1 for s in shared if a[s]["correct"] != 1 and b[s]["correct"] == 1),
                "broken": sum(1 for s in shared if a[s]["correct"] == 1 and b[s]["correct"] != 1),
                "n_paired": len(shared),
                "p_exact": raw_p[(model, v)],
                "p_holm": holm[(model, v)],
                "stars": stars(holm[(model, v)]),
            }
        out_models.append({"model": model, "label": DISPLAY[model],
                           "accuracy": acc, "target": target,
                           "n_by_prompt": n_by, "paired": paired})

    return {
        "models": out_models, "prompts": PROMPT_ORDER, "variants": variants,
        "labels": LABEL, "colours": COLOUR, "outfile": outfile,
        "title": "One added instruction, three hosted models: a complete repair, "
                 "a partial one, and none",
        # Hard-wrapped: the footer is drawn in figure coordinates and
        # bbox_inches="tight" grows the canvas to contain it, so one long line
        # silently stretches the figure to five times its intended width.
        "footer": (
            "Frozen rubric v1-frozen-2026-06 against variants v2 and v3, "
            "raw-series representation, 180 scenarios, temperature 0.\n"
            "Panel (b): healed_transient carries the benchmark label PASS, so "
            "the rate shown is the proportion correctly passed (specificity), "
            "not recall.\n"
            "Panel (c): stars are the Holm-corrected two-sided exact McNemar "
            "test over all six comparisons  -  *** p<0.001, ** p<0.01, "
            "* p<0.05."),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="original-180")
    ap.add_argument("--representation", default="raw")
    ap.add_argument("--outfile", default="ablation_summary.png")
    ap.add_argument("--dpi", type=int, default=200,
                    help="output resolution; the figure size is fixed, so this "
                         "scales the pixels without changing the aspect ratio")
    ap.add_argument("--experiments", default=None,
                    help="directory of long-format result CSVs "
                         "(default: results/agg/experiments)")
    args = ap.parse_args()

    spec = build(args.dataset, args.representation, args.outfile,
                 Path(args.experiments) if args.experiments else None)
    spec["dpi"] = args.dpi
    if not any(m["accuracy"] for m in spec["models"]):
        print("[figure] no ablation rows in the frame", file=sys.stderr)
        return 1
    info = container_plot.render(PLOT, spec)
    print(container_plot.report(info))
    print(json.dumps({m["label"]: {"accuracy": m["accuracy"], "target": m["target"],
                                   "paired": m["paired"]}
                      for m in spec["models"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
