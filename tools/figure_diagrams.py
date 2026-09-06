#!/usr/bin/env python3
"""Figures 1, 4, 5 and 6: the four schematic diagrams, drawn rather than pasted.

These four were the only manuscript figures with no generator. They were drawn
by hand in a diagramming tool, which is why they were the only ones that could
not be re-rendered at print resolution when the rest of the set was. Everything
here is laid out on a normalised 0-1 canvas, so a figure's aspect ratio and its
resolution are the only things that change between renders.

Usage:
    python tools/figure_diagrams.py --out results/figures --dpi 300
    python tools/figure_diagrams.py --out results/figures --only 4

Each figure keeps the aspect ratio of the version it replaces, so a manuscript
that already places these images does not have to be re-laid out.
"""

from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Polygon

# One palette for all four, matching the roles the manuscript uses: the
# statistical path is blue, the model path orange, an outcome green, the
# interface purple, infrastructure grey, and a failing verdict red.
BLUE   = ("#E8EEF9", "#3B6EB5")
ORANGE = ("#FBE7DC", "#D9743B")
GREEN  = ("#E4F1E4", "#1E8A2E")
PURPLE = ("#EDE9F7", "#7A5FBF")
GREY   = ("#F4F4F4", "#8A8A8A")
RED    = ("#FBE9E9", "#C0392B")
ARROW  = "#4A4A4A"
MUTED  = "#6E6E6E"

ASPECT = {1: 3.276, 4: 1.589, 5: 2.271, 6: 1.146}
WIDTH_IN = 7.0


def canvas(fig_no):
    h = WIDTH_IN / ASPECT[fig_no]
    fig, ax = plt.subplots(figsize=(WIDTH_IN, h))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_axis_off()
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    return fig, ax


def box(ax, x, y, w, h, text, style=BLUE, fs=8.0, weight="normal", radius=0.012):
    face, edge = style
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle=f"round,pad=0,rounding_size={radius}",
                                linewidth=1.3, facecolor=face, edgecolor=edge,
                                mutation_aspect=ASPECT_CURRENT[0]))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, weight=weight, color="#1A1A1A", linespacing=1.35)
    return (x, y, w, h)


def diamond(ax, cx, cy, w, h, text, fs=7.2):
    face, edge = GREY
    ax.add_patch(Polygon([(cx, cy + h / 2), (cx + w / 2, cy),
                          (cx, cy - h / 2), (cx - w / 2, cy)],
                         closed=True, facecolor=face, edgecolor=edge, linewidth=1.3))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fs,
            color="#1A1A1A", linespacing=1.3)


def arrow(ax, p0, p1, color=ARROW, style="-|>", dashed=False, lw=1.4, rad=0.0):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=11,
                                 linewidth=lw, color=color,
                                 linestyle="--" if dashed else "-",
                                 connectionstyle=f"arc3,rad={rad}",
                                 shrinkA=0, shrinkB=0))


def elbow(ax, x0, y0, x1, y1, color=ARROW, lw=1.4, first="h"):
    """Right-angled connector from (x0, y0) to (x1, y1), arrowhead at the end.

    Slanted connectors read badly at these aspect ratios because the canvas is
    not square, so every join here turns once instead.
    """
    if first == "h":
        mid = (x1, y0)
    else:
        mid = (x0, y1)
    ax.add_patch(FancyArrowPatch((x0, y0), mid, arrowstyle="-", linewidth=lw,
                                 color=color, shrinkA=0, shrinkB=0))
    arrow(ax, mid, (x1, y1), color=color, lw=lw)


def trunk(ax, x_out, ys, x_trunk, x_end, y_end, color=ARROW, lw=1.4):
    """Several sources on one vertical trunk, joined into a single arrow.

    ys are the source y positions at x_out; the trunk runs down x_trunk and
    leaves as one arrow into (x_end, y_end).
    """
    for y in ys:
        ax.add_patch(FancyArrowPatch((x_out, y), (x_trunk, y), arrowstyle="-",
                                     linewidth=lw, color=color, shrinkA=0, shrinkB=0))
    ax.add_patch(FancyArrowPatch((x_trunk, min(ys)), (x_trunk, max(ys)), arrowstyle="-",
                                 linewidth=lw, color=color, shrinkA=0, shrinkB=0))
    arrow(ax, (x_trunk, y_end), (x_end, y_end), color=color, lw=lw)


ASPECT_CURRENT = [1.0]


# --------------------------------------------------------------------- Fig 1
def figure1(out, dpi):
    ASPECT_CURRENT[0] = ASPECT[1]
    fig, ax = canvas(1)
    labels = ["Metric sources\n(Prometheus /\nVictoriaMetrics)",
              "Fetch phase\npaired control /\nexperiment series",
              "Judge phase\nclassify metrics,\nweighted score",
              "Canary score\n0 to 100",
              "Promote /\nMarginal / Fail"]
    styles = [BLUE, BLUE, BLUE, BLUE, GREEN]
    w, gap = 0.165, 0.032
    x0 = (1 - (5 * w + 4 * gap)) / 2
    for i, (lab, st) in enumerate(zip(labels, styles)):
        x = x0 + i * (w + gap)
        box(ax, x, 0.24, w, 0.52, lab, st, fs=8.0)
        if i < 4:
            arrow(ax, (x + w, 0.50), (x + w + gap, 0.50))
    fig.savefig(os.path.join(out, "fig01_kayenta_pipeline.png"), dpi=dpi,
                facecolor="white")
    plt.close(fig)


# --------------------------------------------------------------------- Fig 4
def figure4(out, dpi):
    ASPECT_CURRENT[0] = ASPECT[4]
    fig, ax = canvas(4)

    box(ax, 0.16, 0.845, 0.24, 0.115, "Referee (UI)", PURPLE, fs=8.2)
    arrow(ax, (0.28, 0.845), (0.28, 0.735))

    kay = box(ax, 0.055, 0.575, 0.40, 0.16,
              "Kayenta\n(ACA engine, standalone)", BLUE, fs=8.6, weight="bold")
    js = box(ax, 0.545, 0.575, 0.22, 0.16, "judge-service\n(Remote Judge)", ORANGE, fs=8.2)
    box(ax, 0.815, 0.575, 0.165, 0.16, "LiteLLM\ngateway", ORANGE, fs=8.2)
    arrow(ax, (0.765, 0.655), (0.815, 0.655))

    # request out, hybrid verdict back
    arrow(ax, (0.455, 0.688), (0.545, 0.688))
    arrow(ax, (0.545, 0.622), (0.455, 0.622), color=ORANGE[1], dashed=True)
    ax.text(0.50, 0.545, "hybrid callback", ha="center", va="top", fontsize=6.8,
            style="italic", color=ORANGE[1])

    arrow(ax, (0.8975, 0.575), (0.8975, 0.435))
    box(ax, 0.795, 0.275, 0.205, 0.16, "Ollama (host GPU)\nor hosted API", GREEN, fs=7.8)

    for cx in (0.115, 0.255, 0.395):
        arrow(ax, (cx, 0.575), (cx, 0.445), style="<|-|>")
    for x, w, lab in ((0.045, 0.145, "Redis"), (0.205, 0.145, "MinIO (S3)"),
                      (0.355, 0.145, "VictoriaMetrics")):
        box(ax, x, 0.285, w, 0.16, lab, GREY, fs=7.8)
    ax.text(0.2725, 0.255, "backing services", ha="center", va="top",
            fontsize=7.2, color=MUTED)

    ax.text(0.5, 0.10, "one analysis:   fetch  →  pair  →  judge  →  score  →  decision",
            ha="center", va="center", fontsize=8.6, color="#3A3A3A")
    fig.savefig(os.path.join(out, "fig04_architecture.png"), dpi=dpi, facecolor="white")
    plt.close(fig)


# --------------------------------------------------------------------- Fig 5
def figure5(out, dpi):
    ASPECT_CURRENT[0] = ASPECT[5]
    fig, ax = canvas(5)

    box(ax, 0.015, 0.370, 0.185, 0.26,
        "Paired control /\nexperiment series", BLUE, fs=8.0)
    rows = [(0.710, "Summary\nstatistics", "LLM"),
            (0.370, "Raw arrays\n(text)", "LLM"),
            (0.030, "Rendered plot (PNG)\n+ summary statistics", "VLM")]
    for y, rep, model in rows:
        box(ax, 0.315, y, 0.235, 0.26, rep, ORANGE, fs=7.8)
        box(ax, 0.625, y, 0.135, 0.26, model, GREEN, fs=8.6)
        arrow(ax, (0.55, y + 0.13), (0.625, y + 0.13))
        arrow(ax, (0.20, 0.50), (0.315, y + 0.13))
        arrow(ax, (0.76, y + 0.13), (0.845, 0.50))
    box(ax, 0.845, 0.350, 0.145, 0.30, "Structured\nverdict", PURPLE,
        fs=8.2, weight="bold")
    fig.savefig(os.path.join(out, "fig05_representations.png"), dpi=dpi, facecolor="white")
    plt.close(fig)


# --------------------------------------------------------------------- Fig 6
def figure6(out, dpi):
    ASPECT_CURRENT[0] = ASPECT[6]
    fig, ax = canvas(6)

    def gate_panel(title, cy, kind, note):
        ax.text(0.02, cy + 0.118, title, fontsize=9.0, weight="bold", color="#1A1A1A")
        top, bot = cy + 0.055, cy - 0.055
        box(ax, 0.030, top - 0.035, 0.160, 0.070, "statistical\nverdict", BLUE, fs=7.2)
        box(ax, 0.030, bot - 0.035, 0.160, 0.070, "AI\nverdict", ORANGE, fs=7.2)
        gx, gw, gh = 0.300, 0.130, 0.070
        if kind == "or":
            face, edge, lab, lc = ORANGE[0], ORANGE[1], "OR", ORANGE[1]
            pts = [(gx, cy + gh), (gx + 0.032, cy), (gx, cy - gh), (gx + gw, cy)]
            inset = 0.032 * (1 - 0.055 / gh)      # where the notch sits at y = cy +/- 0.055
            tx = gx + 0.052
        else:
            face, edge, lab, lc = BLUE[0], BLUE[1], "AND", BLUE[1]
            pts = [(gx, cy + gh), (gx + 0.055, cy + gh), (gx + gw, cy),
                   (gx + 0.055, cy - gh), (gx, cy - gh)]
            inset = 0.0
            tx = gx + 0.046
        ax.add_patch(Polygon(pts, closed=True, facecolor=face, edgecolor=edge, linewidth=1.4))
        ax.text(tx, cy, lab, ha="center", va="center", fontsize=8.4, weight="bold", color=lc)
        # straight in, level with each source box, onto the gate's own left edge
        arrow(ax, (0.190, top), (gx + inset, top))
        arrow(ax, (0.190, bot), (gx + inset, bot))
        arrow(ax, (gx + gw, cy), (0.520, cy))
        box(ax, 0.520, cy - 0.048, 0.200, 0.096, "promote / fail", GREEN, fs=7.6)
        ax.text(0.755, cy, note, fontsize=7.4, style="italic", color=MUTED,
                ha="left", va="center", linespacing=1.4)

    gate_panel("fail-if-either (OR)", 0.845, "or", "maximises\nrecall")
    gate_panel("both-must-fail (AND)", 0.545, "and", "maximises\nprecision")

    ax.text(0.02, 0.322, "gated (recommended)", fontsize=9.0, weight="bold", color="#1A1A1A")
    cy = 0.172
    d1, d2, d3 = 0.320, 0.550, 0.790
    dh, dw1, dw3 = 0.056, 0.078, 0.100
    box(ax, 0.020, 0.199, 0.150, 0.066, "statistical\nverdict", BLUE, fs=7.2)
    box(ax, 0.020, 0.079, 0.150, 0.066, "AI\nverdict", ORANGE, fs=7.2)
    # both inputs join one trunk and enter the first decision head-on
    trunk(ax, 0.170, (0.232, 0.112), 0.212, d1 - dw1, cy)

    diamond(ax, d1, cy, dw1 * 2, dh * 2, "statistical\nFAIL?", fs=6.8)
    diamond(ax, d2, cy, dw1 * 2, dh * 2, "clear\nPASS?", fs=6.8)
    diamond(ax, d3, cy, dw3 * 2, dh * 2, "AI FAIL and\n(score \u2264 50 or\nblind-spot)?", fs=6.2)

    for xa, xb, lab in ((d1 + dw1, d2 - dw1, "no"), (d2 + dw1, d3 - dw3, "yes")):
        arrow(ax, (xa, cy), (xb, cy))
        ax.text((xa + xb) / 2, cy + 0.015, lab, ha="center", fontsize=6.6, color=MUTED)
    arrow(ax, (d3 + dw3, cy), (0.900, cy))
    ax.text(0.878, cy + 0.015, "no", ha="center", fontsize=6.6, color=MUTED)
    box(ax, 0.900, cy - 0.042, 0.090, 0.084, "PASS", GREEN, fs=8.0, weight="bold")

    for dx in (d1, d3):
        arrow(ax, (dx, cy - dh), (dx, 0.076))
        ax.text(dx + 0.012, cy - dh - 0.012, "yes", fontsize=6.6, color=MUTED, va="top")
        box(ax, dx - 0.075, 0.012, 0.150, 0.064, "FAIL", RED, fs=8.0, weight="bold")

    arrow(ax, (d2, cy + dh), (d2, 0.248))
    ax.text(d2 - 0.012, cy + dh + 0.022, "no", fontsize=6.6, color=MUTED, ha="right")
    box(ax, d2 - 0.100, 0.248, 0.200, 0.066, "defer to\nAI verdict", ORANGE, fs=7.2)

    fig.savefig(os.path.join(out, "fig06_hybrid_policies.png"), dpi=dpi, facecolor="white")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="results/figures")
    ap.add_argument("--results", default=None,
                    help="accepted for symmetry with the other generators and ignored: "
                         "these four figures are schematics and read no results file")
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--only", type=int, choices=[1, 4, 5, 6], default=None)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for n, fn in ((1, figure1), (4, figure4), (5, figure5), (6, figure6)):
        if a.only in (None, n):
            fn(a.out, a.dpi)
            print(f"wrote Figure {n}")


if __name__ == "__main__":
    main()
