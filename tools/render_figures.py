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

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

IN = "/app/data/ai-logs/_exp_figdata.json"
OUT = "/app/data/ai-logs/_exp_figs"


def main() -> int:
    with open(IN) as f:
        spec = json.load(f)
    os.makedirs(OUT, exist_ok=True)

    # 1) Grouped bar chart: accuracy / precision / recall / F1 per judge.
    bar = spec["bar"]
    judges = list(bar.keys())
    metrics = ["accuracy", "precision", "recall", "f1"]
    x = np.arange(len(judges))
    w = 0.2
    fig, ax = plt.subplots(figsize=(max(8, 1.3 * len(judges)), 5), dpi=120)
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

    # 2) Family heatmap: judge x family accuracy.
    hm = spec["heatmap"]
    rows, cols, z = hm["rows"], hm["cols"], np.array(hm["z"], dtype=float)
    fig, ax = plt.subplots(figsize=(max(8, 1.1 * len(cols)), max(4, 0.6 * len(rows))), dpi=120)
    im = ax.imshow(z, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=40, ha="right")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows)
    for i in range(len(rows)):
        for j in range(len(cols)):
            ax.text(j, i, f"{z[i, j]:.2f}", ha="center", va="center", fontsize=8,
                    color="black")
    ax.set_title("Per-family accuracy (judge x family)")
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label="accuracy")
    fig.tight_layout()
    fig.savefig(f"{OUT}/family_heatmap.png")
    plt.close(fig)

    # 3) Confusion matrices for the headline judges.
    conf = spec.get("confusion", {})
    if conf:
        n = len(conf)
        fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.0), dpi=120, squeeze=False)
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
