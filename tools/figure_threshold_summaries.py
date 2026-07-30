#!/usr/bin/env python3
"""Figure 11: threshold-independent summaries for the representative configurations.

Three bars per configuration from results/agg/roc_pr_auc.csv: AUC-ROC, average
precision, and the trapezoidal precision-recall area. Average precision is the
primary precision-recall summary; the trapezoid is shown alongside because the
two diverge for a judge that emits few distinct scores, and the divergence
flatters the judge with the fewest.

The trapezoid bars are hatched and drawn in a lighter shade so the two
precision-recall summaries are not read as interchangeable.

Requires the `average_precision` column, which tools/roc_pr_curves.py writes.
An older results file without that column is reported and the run stops rather
than silently plotting the trapezoid twice.

Needs matplotlib. On a host without it, run inside the judge-service image:

    docker run --rm -v "$PWD":/work -w /work canaryllm-judge-service \
        python tools/figure_threshold_summaries.py

Usage:
  python tools/figure_threshold_summaries.py [--results results] [--out results/figures]
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent

ROC_COLOUR = "#4c72b0"
AP_COLOUR = "#55a868"
TRAPZ_COLOUR = "#a8d5b5"


def short(judge: str, model: str) -> str:
    m = model.replace("-llm", "").replace("-vlm", "").replace("claude-", "")
    rep = judge.split(":")[1] if ":" in judge else judge
    if judge == "statistical":
        return "statistical*"
    if judge.startswith("hybrid"):
        return f"hybrid:gated\n{m}"
    return f"{m}\n{rep}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results", default=str(REPO_ROOT / "results"))
    ap.add_argument("--out", default=None,
                    help="output directory (default: <results>/figures)")
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args()

    results = Path(args.results)
    src = results / "agg" / "roc_pr_auc.csv"
    if not src.exists():
        raise SystemExit(f"{src} not found; run tools/roc_pr_curves.py first")
    with src.open() as f:
        rows = [r for r in csv.DictReader(f) if r.get("auc_roc")]
    if not rows:
        raise SystemExit(f"{src} has no scored rows")
    if "average_precision" not in rows[0]:
        raise SystemExit(
            f"{src} has no average_precision column. Re-run tools/roc_pr_curves.py; "
            "the trapezoid alone is not the primary summary this figure reports.")

    labels = [short(r["judge"], r["model"]) for r in rows]
    roc = [float(r["auc_roc"]) for r in rows]
    apv = [float(r["average_precision"]) for r in rows]
    trapz = [float(r["auc_pr"]) for r in rows]

    x = list(range(len(rows)))
    w = 0.26
    fig, ax = plt.subplots(figsize=(6.500, 3.048), dpi=args.dpi)
    b1 = ax.bar([xi - w for xi in x], roc, w, label="AUC-ROC", color=ROC_COLOUR)
    b2 = ax.bar(x, apv, w, label="Average precision (primary PR summary)",
                color=AP_COLOUR)
    b3 = ax.bar([xi + w for xi in x], trapz, w,
                label="AUC-PR, trapezoid (for comparison)",
                color=TRAPZ_COLOUR, hatch="//", edgecolor=AP_COLOUR)
    for bars in (b1, b2, b3):
        ax.bar_label(bars, fmt="%.3f", fontsize=6, padding=1)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("area under curve")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.20), ncol=3,
              frameon=False, fontsize=7)
    fig.tight_layout()

    out_dir = Path(args.out) if args.out else results / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "fig11_threshold_summaries.png"
    fig.savefig(path)
    plt.close(fig)

    print("[figure-11] configurations plotted:")
    for r in rows:
        print(f"  {r['judge']:14}/{r['model']:22} auc_roc={float(r['auc_roc']):.4f} "
              f"ap={float(r['average_precision']):.4f} "
              f"trapezoid={float(r['auc_pr']):.4f}")
    print(f"  wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
