#!/usr/bin/env python3
"""Figure 5: one seeded instance of each of the nine kind-control families and the five held-out benign families.

Drawn in the style of Figure 2 of the manuscript. Figure 2 was drawn by write_a6 of
analysis/area-A-headline/analyze_A.py of the result data deposit (an SVG at 6.9 in, rendered to PNG at 600 dpi); no
tool in tools/ draws it. write_a6 lays out exactly nine panels in a 3 x 3 grid, so it cannot draw fourteen
unchanged. This script keeps its geometry, colours, markers, line widths, text sizes, title lines and tick rule
(analyze_A._nice_ticks, loaded read only) and changes four things:
  * five rows of three panels instead of three rows (fourteen panels);
  * a plot height of 53 pt instead of 84 pt and 16 pt below each plot instead of 22, so that the figure is at most
    1.20 times as high as wide (the height the manuscript's figure template holds together with its caption);
  * the explanatory note sits in the fifteenth (empty) cell instead of under the grid;
  * the text font is DejaVu Sans, the sans-serif font of the judge-service image (the published PNGs were rendered
    from SVG with a Helvetica-type face).

Panels: the first window (_000) of each family, the series of the protocol's series files (the windows the judges
saw); the scenario's first metric (latency for the three-metric families, as Figure 2 does for
cross_metric_marginal); memory in MiB. Panel titles: family; label and the statistical judge's verdict, read from
the run records (identical in every invocation); canary/control ratio of the sample SD (n - 1) and of the mean of
the plotted series. The PNG is drawn by matplotlib inside the running judge-service container (render_in_service).

Run (from the repository root, with COMPOSE_PROJECT_NAME and COMPOSE_FILE of the stack in the environment):
  python -I -B analysis/r1/figures/figure5_controls.py --runs <runs> --deposit <data-archive> --out <new-dir>
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import r1_common as rc  # noqa: E402
import render_in_service as ris  # noqa: E402

PT = 72.0
W_IN = 6.9
LEGEND_H, HEAD, PH, BELOW = 22.0, 44.0, 53.0, 16.0
CTL, CAN = "#0072B2", "#D55E00"
DPI = 600
UNIT = {"s": ("s", 1.0), "bytes": ("MiB", 1.0 / (1024 * 1024)), "err/s": ("err/s", 1.0)}

PLOT = r"""
from matplotlib.patches import Rectangle
W, H = spec["W"], spec["H"]
fig = plt.figure(figsize=(spec["W_in"], spec["H_in"]))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W)
ax.set_ylim(H, 0)
ax.axis("off")
plt.rcParams["font.family"] = "DejaVu Sans"
for p in spec["prims"]:
    t = p[0]
    if t == "text":
        _, x, y, s, size, weight, ha = p
        ax.text(x, y, s, fontsize=size, fontweight=weight, ha=ha, va="baseline", family="DejaVu Sans")
    elif t == "line":
        _, x1, y1, x2, y2, col, lw, dash = p
        ax.plot([x1, x2], [y1, y2], color=col, lw=lw, solid_capstyle="butt",
                linestyle=(0, tuple(dash)) if dash else "-")
    elif t == "poly":
        _, xs, ys, col, lw = p
        ax.plot(xs, ys, color=col, lw=lw, solid_joinstyle="miter")
    elif t == "mark":
        _, kind, xs, ys, col, r = p
        ax.plot(xs, ys, linestyle="none", marker="o" if kind == "c" else "s", markersize=2 * r,
                markerfacecolor=col, markeredgewidth=0)
    elif t == "rect":
        _, x, y, w, h, edge, lw, fill = p
        ax.add_patch(Rectangle((x, y), w, h, fill=fill is not None, facecolor=fill or "none",
                               edgecolor=edge or "none", lw=lw))
    elif t == "wrap":
        # greedy word wrap on the rendered width of DejaVu Sans at this size (points)
        from matplotlib.textpath import TextPath
        from matplotlib.font_manager import FontProperties
        _, x, y, s, size, max_w, step, max_lines = p
        fp = FontProperties(family="DejaVu Sans", size=size)
        lines, cur = [], ""
        for w_ in s.split(" "):
            cand = w_ if not cur else cur + " " + w_
            if cur and TextPath((0, 0), cand, size=size, prop=fp).get_extents().width > max_w:
                lines.append(cur)
                cur = w_
            else:
                cur = cand
        if cur:
            lines.append(cur)
        if len(lines) > max_lines:
            raise SystemExit("note needs %d lines, %d fit" % (len(lines), max_lines))
        for i, ln in enumerate(lines):
            ax.text(x, y + i * step, ln, fontsize=size, ha="left", va="baseline", family="DejaVu Sans")
        print("NOTE_LINES %d" % len(lines))
emit(fig, spec["outfile"], spec["dpi"])
"""


def panels_spec(recs, series):
    letters = "abcdefghijklmn"
    fams = [(rc.KC, f) for f in rc.KC_FAMILIES] + [(rc.HB, f) for f in rc.HB_FAMILIES]
    out = []
    for i, (ds, f) in enumerate(fams):
        sid = f"{f}_000"
        w = series[ds][sid]
        mt = w["metrics"][0]
        unit, scale = UNIT[mt["unit"]]
        ctl = [v * scale for v in mt["control"]]
        can = [v * scale for v in mt["canary"]]
        verdict = recs.stat_verdicts(ds)[sid]
        out.append({"letter": letters[i], "dataset": ds, "family": f, "scenario": sid, "truth": w["truth"],
                    "verdict": verdict, "kind": mt["kind"], "unit": unit, "metric": mt["name"],
                    "n_metrics": len(w["metrics"]), "control": ctl, "canary": can,
                    "sd_ratio": statistics.stdev(can) / statistics.stdev(ctl),
                    "mean_ratio": statistics.fmean(can) / statistics.fmean(ctl)})
    return out


def primitives(panels, nice_ticks, wrap):
    W = W_IN * PT
    pw = W / 3
    cell_h = HEAD + PH + BELOW
    rows = math.ceil(len(panels) / 3)
    H = LEGEND_H + rows * cell_h + 4
    H_in = math.ceil(H / PT * 100) / 100
    P = []

    def marker(kind, xs, ys, col, r=1.1):
        P.append(["mark", kind, xs, ys, col, r])
    lx = W / 2 - 110
    for kind, col, text in (("c", CTL, "control"), ("s", CAN, "canary (experiment)")):
        P.append(["line", lx, 10.0, lx + 18, 10.0, col, 0.9, None])
        marker(kind, [lx + 9], [10.0], col, 1.6)
        P.append(["text", lx + 23, 12.8, text, 8, "normal", "left"])
        lx += 23 + len(text) * 4.4 + 24
    for idx, p in enumerate(panels):
        r_, c_ = divmod(idx, 3)
        x0, y0 = c_ * pw, LEGEND_H + r_ * cell_h
        cx = x0 + pw / 2
        pl, pr = x0 + 34.0, x0 + pw - 10.0
        ptop, pbot = y0 + HEAD, y0 + HEAD + PH
        P.append(["text", cx, y0 + 9, f"({p['letter']}) {p['family']}", 8, "bold", "center"])
        P.append(["text", cx, y0 + 18.6, f"label {p['truth']}; statistical judge: {p['verdict']}", 8, "normal", "center"])
        P.append(["text", cx, y0 + 28.2, f"SD ratio {p['sd_ratio']:.2f}; mean ratio {p['mean_ratio']:.3f}", 8, "normal", "center"])
        P.append(["text", x0 + 4, y0 + 38.6, f"{p['kind']} ({p['unit']})", 8, "normal", "left"])
        vals = p["control"] + p["canary"]
        vlo, vhi = min(vals), max(vals)
        pad = 0.06 * (vhi - vlo)
        ylo, yhi = vlo - pad, vhi + pad
        n = len(p["control"])

        def X(i):
            return pl + i / (n - 1) * (pr - pl)

        def Y(v):
            return pbot - (v - ylo) / (yhi - ylo) * (pbot - ptop)
        for tv, tl in nice_ticks(ylo, yhi, 4):
            yy = Y(tv)
            P.append(["line", pl, yy, pr, yy, "#dddddd", 0.4, None])
            P.append(["line", pl - 2, yy, pl, yy, "#000000", 0.5, None])
            P.append(["text", pl - 3.5, yy + 2.8, tl, 8, "normal", "right"])
        for i in (0, 10, 20, 30):
            xx = X(i)
            P.append(["line", xx, pbot, xx, pbot + 2, "#000000", 0.5, None])
            P.append(["text", xx, pbot + 10.5, str(i), 8, "normal", "center"])
        for ser, col, mk in ((p["control"], CTL, "c"), (p["canary"], CAN, "s")):
            xs = [X(i) for i in range(n)]
            ys = [Y(v) for v in ser]
            P.append(["poly", xs, ys, col, 0.8])
            marker(mk, xs, ys, col)
        P.append(["rect", pl, ptop, pr - pl, PH, "#000000", 0.5, None])
    # note in the empty cell (bottom right)
    idx = len(panels)
    r_, c_ = divmod(idx, 3)
    x0, y0 = c_ * pw, LEGEND_H + r_ * cell_h
    note = ("x axis: sample index, one sample per minute. (a) to (i): kind-controls-180; (j) to (n): "
            "heldout-benign-100. Titles: family; label and the statistical judge's verdict; canary/control ratios "
            "of the sample SD and of the mean of the plotted series. (d) and (i) show latency, the first of their "
            "three metrics.")
    max_lines = int((cell_h - 12) // 10.5) + 1
    P.append(["wrap", x0 + 8, y0 + 12, note, 8, pw - 14, 10.5, max_lines])
    return P, W, H, H_in, [note]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--deposit", required=True, help="data deposit 1.1.1 (analyze_A._nice_ticks and _wrap, read only)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--container", default="fi-r1-judge-service", help="name the judge-service container must have")
    ap.add_argument("--outfile", default="Figure_5.png")
    a = ap.parse_args(argv)
    runs = Path(a.runs).resolve()
    out = Path(a.out)
    env = {"math": math}
    rc.pc.load_functions(Path(a.deposit) / "analysis" / "area-A-headline" / "analyze_A.py", ["_nice_ticks", "_wrap"], env)
    recs = rc.Records(runs / "records", rc.load_protocol(), runs.parent)
    series = rc.load_all_series()
    panels = panels_spec(recs, series)
    prims, W, H, H_in, note_lines = primitives(panels, env["_nice_ticks"], env["_wrap"])
    if H_in > 1.20 * W_IN:
        raise SystemExit(f"figure height {H_in} in exceeds 1.20 times its width")
    spec = {"W": W, "H": H, "W_in": W_IN, "H_in": H_in, "dpi": DPI, "outfile": a.outfile, "prims": prims}
    info = ris.render(PLOT, spec, out, a.container)
    data = {"figure": a.outfile, "size": info, "width_in": W_IN, "height_in": H_in, "dpi": DPI,
            "panels": [{k: (round(v, 6) if isinstance(v, float) else v) for k, v in p.items() if k not in ("control", "canary")}
                       for p in panels],
            "series_files": [f"protocol/series_{rc.KC}.jsonl", f"protocol/series_{rc.HB}.jsonl"],
            "verdicts_from": "runs/records (judge=statistical; identical in every invocation)",
            "note_lines": note_lines}
    rc.write_json(out / (Path(a.outfile).stem + "_data.json"), data)
    facts = []
    for p in panels:
        facts.append({"key": f"fig5.{p['family']}", "value": f"({p['letter']}) {p['scenario']}: label {p['truth']}; statistical judge {p['verdict']}; "
                      f"SD ratio {p['sd_ratio']:.2f}; mean ratio {p['mean_ratio']:.3f} ({p['kind']}, {p['unit']})",
                      "description": f"Figure 5 panel ({p['letter']}): {p['family']}, first window, first metric",
                      "source_file": f"protocol/series_{p['dataset']}.jsonl; runs/records (statistical rows)",
                      "filter_or_computation": "statistics.stdev and statistics.fmean of the plotted canary over the plotted control",
                      "script": "analysis/r1/figures/figure5_controls.py"})
    facts.append({"key": "fig5.png", "value": f"{info[0]['width_px']} x {info[0]['height_px']} px at {info[0]['dpi']} dpi",
                  "description": "Figure 5 PNG size", "source_file": "analysis/figures/" + a.outfile,
                  "filter_or_computation": "tools/container_plot.png_size", "script": "analysis/r1/figures/figure5_controls.py"})
    rc.write_json(out / (Path(a.outfile).stem + "_facts.json"), facts)
    print(json.dumps(info, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
