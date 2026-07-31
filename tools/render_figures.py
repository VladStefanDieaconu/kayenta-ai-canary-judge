"""Render the experiment figures with matplotlib.

The host tools' venv has no matplotlib (PyPI is gated here), but the judge-service
container does (it renders the `plot` representation). So tools/run_experiment.py
runs this script inside that container:

    docker compose exec -T judge-service python - < tools/render_figures.py

It reads the figure spec from /app/data/ai-logs/_exp_figdata.json (the ai-logs
bind mount is the host<->container transport) and writes PNGs to
/app/data/ai-logs/_exp_figs/, which run_experiment.py then copies into
results/figures/.
"""

import argparse
import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

IN = "/app/data/ai-logs/_exp_figdata.json"
# The published figures were drawn at 120. MDPI asks for 300 in print, so the
# resolution is a flag rather than three literals -- but the default is the
# published value, so re-rendering without asking changes nothing.
DPI = 120
OUT = "/app/data/ai-logs/_exp_figs"

# The heatmap's extra rows, when their table is reachable. Each entry is
# (file under <results>/agg, row selector, label). Both tables are produced by
# analyses that run separately from the main sweep, so neither is guaranteed to
# exist; a missing one drops its row and is reported.
# frontier_family_matrix.csv is a FIXED filename overwritten by every
# run_frontier_experiment.py invocation, so it only ever holds the model that ran
# last. Each run is preserved as frontier_family_matrix_<alias>.csv; select from
# those so every frontier model gets a row instead of only the most recent one.
EXTRA_HEATMAP_ROWS = [
    ("ensemble_family_matrix.csv",
     lambda r: r.get("judge") == "ensemble:tuned",
     "ensemble:tuned"),
    ("frontier_family_matrix_claude-opus-4-8-vlm.csv",
     lambda r: r.get("judge") == "ai:raw" and "opus" in (r.get("model") or ""),
     "opus:raw"),
    ("frontier_family_matrix_gpt-oss-120b-llm.csv",
     lambda r: r.get("judge") == "ai:summary" and "gpt-oss" in (r.get("model") or ""),
     "gpt-oss:summary"),
    ("frontier_family_matrix_qwen3-vl-235b-vlm.csv",
     lambda r: r.get("judge") == "ai:plot" and "qwen3-vl" in (r.get("model") or ""),
     "qwen3-vl:plot"),
]

# hybrid:or is dropped from the heatmap by default. On this benchmark the
# fail-if-either union is verdict-identical to the gated policy on every
# scenario (Section 7.1), so its row duplicates hybrid:gated exactly and costs a
# row without adding information. --include-or restores it, which is the honest
# way to check that identity rather than taking the caption's word for it.
DROP_BY_DEFAULT = ("hybrid:or",)

# Long family names crowd the x axis at this figure width.
COLUMN_ABBREVIATIONS = {"noise_equivalent": "noise_equiv"}


def read_family_matrix(path, selector):
    """The nine per-family accuracies of the first row `selector` accepts."""
    if not os.path.exists(path):
        return None
    with open(path) as f:
        for row in csv.DictReader(f):
            if selector(row):
                return row
    return None


def heatmap_from_results(results):
    """Rebuild the heatmap block from <results>/family_matrix.csv.

    run_experiment.py writes the spec as a side effect of a scored run, so
    without this the heatmap can only be redrawn by re-running the whole
    experiment. family_matrix.csv holds the same per-(judge, model) rates, and
    averaging them over each judge's models reproduces the spec's `z` exactly.
    """
    path = os.path.join(results, "family_matrix.csv")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    cols = [c for c in rows[0] if c not in ("judge", "model")]
    order, by_judge = [], {}
    for r in rows:
        by_judge.setdefault(r["judge"], []).append(r)
        if r["judge"] not in order:
            order.append(r["judge"])
    z = [[sum(float(r[c]) for r in by_judge[j]) / len(by_judge[j]) for c in cols]
         for j in order]
    return {"rows": order, "cols": cols, "z": z}


def main() -> int:
    global IN, OUT, DPI
    ap = argparse.ArgumentParser(description="Render the experiment figures")
    ap.add_argument("--spec", default=IN, help="figure spec written by run_experiment.py")
    ap.add_argument("--out", default=OUT, help="directory to write PNGs into")
    ap.add_argument("--results", default=None,
                    help="results directory holding agg/, for the heatmap's "
                         "ensemble and frontier rows; omit to draw the main "
                         "sweep's judges only")
    ap.add_argument("--dpi", type=int, default=DPI,
                    help=f"figure resolution (default {DPI}, the published value)")
    ap.add_argument("--include-or", action="store_true",
                    help="keep the hybrid:or row, which is verdict-identical to "
                         "hybrid:gated on this benchmark (Section 7.1)")
    args = ap.parse_args()
    IN, OUT, DPI = args.spec, args.out, args.dpi

    spec = {}
    if os.path.exists(IN):
        with open(IN) as f:
            spec = json.load(f)
    elif not args.results:
        raise SystemExit(f"{IN} not found and --results not given; nothing to draw")
    os.makedirs(OUT, exist_ok=True)

    if "heatmap" not in spec and args.results:
        rebuilt = heatmap_from_results(args.results)
        if rebuilt is None:
            raise SystemExit(
                f"no heatmap in the spec and no family_matrix.csv under {args.results}")
        spec["heatmap"] = rebuilt
        print("[render-figures] heatmap rebuilt from family_matrix.csv "
              "(no spec from a scored run)")

    # 1) Grouped bar chart: accuracy / precision / recall / F1 per judge.
    # Only the spec carries these aggregates, so a results-only run skips it.
    bar = spec.get("bar")
    if bar:
        judges = list(bar.keys())
        metrics = ["accuracy", "precision", "recall", "f1"]
        x = np.arange(len(judges))
        w = 0.2
        fig, ax = plt.subplots(figsize=(max(8, 1.3 * len(judges)), 5), dpi=DPI)
        for i, mname in enumerate(metrics):
            vals = [bar[j].get(mname, 0.0) for j in judges]
            ax.bar(x + (i - 1.5) * w, vals, w, label=mname)
        ax.set_xticks(x)
        ax.set_xticklabels(judges, rotation=30, ha="right")
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("score")
        ax.set_title("Judge performance (FAIL = positive class), averaged over models")
        ax.legend(loc="lower right")
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(f"{OUT}/accuracy_bars.png")
        plt.close(fig)
    else:
        print("[render-figures] no bar block in the spec; skipping accuracy_bars.png")

    # 2) Family heatmap: judge x family accuracy.
    hm = spec["heatmap"]
    rows, cols = list(hm["rows"]), list(hm["cols"])
    z = [list(map(float, r)) for r in hm["z"]]

    if not args.include_or:
        keep = [i for i, name in enumerate(rows) if name not in DROP_BY_DEFAULT]
        dropped = [rows[i] for i in range(len(rows)) if i not in keep]
        rows = [rows[i] for i in keep]
        z = [z[i] for i in keep]
        for name in dropped:
            print(f"[render-figures] heatmap: dropped {name} "
                  f"(verdict-identical to hybrid:gated here; --include-or keeps it)")

    # The ensemble and frontier rows come from analyses outside the main sweep,
    # so they are appended below a separating line rather than mixed in.
    separator_after = len(rows) - 1
    if args.results:
        agg = os.path.join(args.results, "agg")
        for filename, selector, label in EXTRA_HEATMAP_ROWS:
            row = read_family_matrix(os.path.join(agg, filename), selector)
            if row is None:
                print(f"[render-figures] heatmap: no {label} row "
                      f"({filename} absent or has no matching row)")
                continue
            rows.append(label)
            z.append([float(row[c]) for c in cols])
    else:
        print("[render-figures] heatmap: --results not given, "
              "drawing the main sweep's judges only")

    z = np.array(z, dtype=float)
    labels = [COLUMN_ABBREVIATIONS.get(c, c) for c in cols]
    fig, ax = plt.subplots(figsize=(max(8, 1.1 * len(cols)), max(4, 0.6 * len(rows))), dpi=DPI)
    im = ax.imshow(z, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(labels, rotation=40, ha="right")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows)
    for i in range(len(rows)):
        for j in range(len(cols)):
            ax.text(j, i, f"{z[i, j]:.2f}", ha="center", va="center", fontsize=8,
                    color="black")
    if len(rows) > separator_after + 1:
        ax.axhline(separator_after + 0.5, color="black", lw=1.2)
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label="accuracy")
    fig.tight_layout()
    fig.savefig(f"{OUT}/family_heatmap.png")
    plt.close(fig)

    # 3) Confusion matrices for the headline judges.
    conf = spec.get("confusion", {})
    if conf:
        n = len(conf)
        fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.0), dpi=DPI, squeeze=False)
        for k, (name, c) in enumerate(conf.items()):
            ax = axes[0][k]
            # rows = actual [FAIL, PASS], cols = predicted [FAIL, PASS]
            m = np.array([[c["TP"], c["FN"]], [c["FP"], c["TN"]]], dtype=float)
            im = ax.imshow(m, cmap="Blues")
            ax.set_xticks([0, 1]); ax.set_xticklabels(["pred FAIL", "pred PASS"])
            ax.set_yticks([0, 1]); ax.set_yticklabels(["true FAIL", "true PASS"])
            for i in range(2):
                for j in range(2):
                    ax.text(j, i, int(m[i, j]), ha="center", va="center",
                            color="black", fontsize=11)
            ax.set_title(name, fontsize=9)
        fig.suptitle("Confusion matrices (FAIL = positive)")
        fig.tight_layout()
        fig.savefig(f"{OUT}/confusion_matrices.png")
        plt.close(fig)

    print(json.dumps({"ok": True, "out": OUT}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
