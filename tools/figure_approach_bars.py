#!/usr/bin/env python3
"""Figure 10: the strongest configuration of each approach, as grouped bars.

Five configurations on accuracy, precision and recall: the statistical judge, the
best local model, the frontier model's best representation, the tuned statistical
ensemble, and the gated hybrid on the best local model.

Reads three tables and picks the strongest row of each approach by accuracy
rather than naming configurations, so a run with a different model set still
produces the figure:
  results/summary.csv                    local judges and hybrids
  results/agg/frontier_representation.csv frontier run, when present
  results/agg/ensemble_scoreboard.csv     tuned ensemble, when present

The frontier and ensemble panels are skipped with a note when their table is
absent, since neither is reachable without cloud credentials or a full run.

Needs matplotlib. On a host without it, run inside the judge-service image:

    docker run --rm -v "$PWD":/work -w /work canaryllm-judge-service \
        python tools/figure_approach_bars.py

Usage:
  python tools/figure_approach_bars.py [--results results] [--out results/figures]
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
METRICS = ["accuracy", "precision", "recall"]
COLOURS = {"accuracy": "#4c72b0", "precision": "#dd8452", "recall": "#55a868"}


def read_csv(path: Path):
    if not path.exists():
        return []
    with path.open() as f:
        return list(csv.DictReader(f))


def strongest(rows, predicate, label_of):
    """The row satisfying `predicate` with the highest accuracy, or None."""
    candidates = [r for r in rows if predicate(r)]
    if not candidates:
        return None
    best = max(candidates, key=lambda r: float(r["accuracy"]))
    return label_of(best), best


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results", default=str(REPO_ROOT / "results"),
                    help="directory holding summary.csv and agg/")
    ap.add_argument("--out", default=None,
                    help="output directory (default: <results>/figures)")
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args()

    results = Path(args.results)
    agg = results / "agg"
    out_dir = Path(args.out) if args.out else results / "figures"

    summary = read_csv(results / "summary.csv")
    # Frontier tables are written to a FIXED filename by every
    # run_frontier_experiment.py invocation, so agg/frontier_representation.csv only
    # ever holds whichever model ran last. Each run is preserved under
    # frontier_representation_<alias>.csv; read all of those so every frontier model
    # is represented, and fall back to the bare file when no suffixed copy exists.
    frontier = []
    suffixed = sorted(agg.glob("frontier_representation_*.csv"))
    for path in suffixed:
        frontier.extend(read_csv(path))
    if not frontier:
        frontier = read_csv(agg / "frontier_representation.csv")
    ensemble = read_csv(agg / "ensemble_scoreboard.csv")

    columns, skipped = [], []

    stat = next((r for r in summary if r["judge"] == "statistical"), None)
    if stat:
        columns.append(("statistical", stat))
    else:
        skipped.append("statistical (no row in summary.csv)")

    pick = strongest(summary, lambda r: r["judge"].startswith("ai:"),
                     lambda r: f"best local\n({r['model'].replace('-llm', '').replace('-vlm', '')} "
                               f"· {r['judge'].split(':')[1]})")
    best_local_row = pick[1] if pick else None
    columns.append(pick) if pick else skipped.append("best local model")

    # One column per frontier MODEL, each at its own strongest representation, so a
    # multi-vendor comparison is visible rather than a single overall winner.
    frontier_models = sorted({r["model"] for r in frontier if r["judge"].startswith("ai:")})
    if frontier_models:
        for fm in frontier_models:
            pick = strongest(frontier,
                             lambda r, fm=fm: r["judge"].startswith("ai:") and r["model"] == fm,
                             lambda r: f"frontier\n({r['model'].replace('-vlm', '').replace('-llm', '').replace('claude-', '')} "
                                       f"· {r['judge'].split(':')[1]})")
            columns.append(pick) if pick else skipped.append(f"frontier {fm}")
    else:
        skipped.append("frontier (no agg/frontier_representation*.csv; needs Bedrock credentials)")

    pick = strongest(ensemble, lambda r: r["judge"] == "ensemble:tuned",
                     lambda r: "ensemble\n(tuned)")
    columns.append(pick) if pick else skipped.append(
        "tuned ensemble (agg/ensemble_scoreboard.csv absent)")

    # The gated hybrid is taken on the SAME model as the best local judge rather
    # than on whichever model scores highest under the gate. Pairing them is what
    # makes the column readable as "what the gate adds to the best local model";
    # the highest-accuracy gated hybrid can sit on a different model and then the
    # two columns are not comparable.
    best_local_model = best_local_row["model"] if best_local_row is not None else None
    pick = strongest(summary,
                     lambda r: r["judge"] == "hybrid:gated" and r["model"] == best_local_model,
                     lambda r: f"gated hybrid\n({r['model'].replace('-llm', '').replace('-vlm', '')})")
    if pick is None:
        pick = strongest(summary, lambda r: r["judge"] == "hybrid:gated",
                         lambda r: f"gated hybrid\n({r['model'].replace('-llm', '').replace('-vlm', '')})")
    columns.append(pick) if pick else skipped.append("gated hybrid")

    labels = [c[0] for c in columns]
    x = list(range(len(labels)))
    width = 0.26

    fig, ax = plt.subplots(figsize=(6.600, 2.987), dpi=args.dpi)
    for i, metric in enumerate(METRICS):
        vals = [float(c[1][metric]) for c in columns]
        ax.bar([xi + (i - 1) * width for xi in x], vals, width,
               label=metric.capitalize(), color=COLOURS[metric])
    ax.axhline(1.0, color="grey", ls=":", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("score")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.18), ncol=3, frameon=False)
    fig.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "fig10_approach_bars.png"
    fig.savefig(path)
    plt.close(fig)

    print("[figure-10] configurations plotted:")
    for label, row in columns:
        flat = label.replace("\n", " ")
        print(f"  {flat:34} acc={float(row['accuracy']):.3f} "
              f"prec={float(row['precision']):.3f} rec={float(row['recall']):.3f}")
    for s in skipped:
        print(f"  SKIPPED: {s}")
    print(f"  wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
