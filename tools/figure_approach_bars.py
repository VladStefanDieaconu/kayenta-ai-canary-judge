#!/usr/bin/env python3
"""Figure 10: the strongest configuration of each approach, as grouped bars.

Five configurations on accuracy, precision and recall: the statistical judge, the
best local model, the frontier model's best representation, the tuned statistical
ensemble, and the gated hybrid on the best local model.

Reads three tables and picks the strongest row of each approach by accuracy
rather than naming configurations, so a run with a different model set still
produces the figure:
  results/summary.csv                    local judges and hybrids
  results/agg/frontier_representation*.csv the hosted runs, filtered to the
                                         three models in FRONTIER_MODELS
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

# The three hosted models, named rather than globbed.
#
# run_frontier_experiment.py writes agg/frontier_representation__<tag>.csv for
# whatever it was pointed at, and it was later pointed at local models for the
# defect re-runs and the reproduction checks. A glob over that filename pattern
# therefore grew from three files to nine, and this figure silently went from
# five columns to eleven with DeepSeek-R1, Moondream, Phi-4 and Qwen2.5 labelled
# "frontier" -- which is not what the caption says and not true of the models.
# The set of hosted models is a fact about the study, not something to infer
# from a directory listing, so it is written down. --frontier-models overrides.
FRONTIER_MODELS = ("claude-opus-4-8-vlm", "gpt-oss-120b-llm", "qwen3-vl-235b-vlm")


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
    ap.add_argument("--frontier-models", default="",
                    help="comma-separated model aliases to treat as frontier "
                         f"(default: {','.join(FRONTIER_MODELS)})")
    ap.add_argument("--frontier-columns", choices=("best", "all"), default="best",
                    help="one column for the strongest hosted configuration (best, "
                         "the published figure), or one per hosted model (all)")
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
    wanted = set(args.frontier_models.split(",")) if args.frontier_models else set(FRONTIER_MODELS)
    for path in sorted(agg.glob("frontier_representation_*.csv")):
        frontier.extend(r for r in read_csv(path) if r.get("model") in wanted)
    if not frontier:
        frontier = [r for r in read_csv(agg / "frontier_representation.csv")
                    if r.get("model") in wanted]
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

    # The published figure carries one frontier column, the strongest hosted
    # configuration, and its caption names that model. Splitting it per model
    # makes a multi-vendor comparison visible but no longer matches the caption,
    # so it is available under --frontier-columns all rather than by default.
    frontier_label = (lambda r: "frontier\n("
                      f"{r['model'].replace('-vlm', '').replace('-llm', '').replace('claude-', '')} "
                      f"· {r['judge'].split(':')[1]})")
    frontier_models = sorted({r["model"] for r in frontier if r["judge"].startswith("ai:")})
    if not frontier_models:
        skipped.append("frontier (no agg/frontier_representation*.csv; needs Bedrock credentials)")
    elif args.frontier_columns == "all":
        for fm in frontier_models:
            pick = strongest(frontier,
                             lambda r, fm=fm: r["judge"].startswith("ai:") and r["model"] == fm,
                             frontier_label)
            columns.append(pick) if pick else skipped.append(f"frontier {fm}")
    else:
        pick = strongest(frontier, lambda r: r["judge"].startswith("ai:"), frontier_label)
        columns.append(pick) if pick else skipped.append("frontier")

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
