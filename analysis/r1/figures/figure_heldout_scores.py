#!/usr/bin/env python3
"""Supplementary figure: hosted raw-series scores on the held-out benign and regression windows, by rubric.

Drawn in the style of Figure 4 of the manuscript. Figure 4 was drawn by figure_scores of
analysis/area-C-frontier-ablation-heldout/analyze_C.py of the result data deposit (an SVG at 6.9 in, rendered to
PNG at 600 dpi); no tool in tools/ draws it, and figure_scores reads the nine families of the 180-window set only.
This script keeps its geometry (6.9 in wide; three models by three rubrics; one row per family with a bar per
five-point score bin, bar height = windows per bin with row height = 20; thresholds 75, 80 and 85 as solid, dashed
and dotted lines; benign rows above a grey separator), its colours (vermillion for the family the repair targets,
blue for the other benign families, green for regressions) and text sizes, and changes three things:
  * the rows are the five held-out benign families of heldout-benign-100 (run H3) and the three held-out regression
    families of generalisation-60 (run H4), all from the November 2026 session; the two recovered-transient
    families take the vermillion of healed_transient, the three exchangeable families the blue of no_change and
    noise_equivalent;
  * the left margin is 128 pt instead of 98 pt and the right margin 14 pt instead of 6 pt, so that the longer
    family names and the last tick label fit; the legend entries are spaced for the wider font;
  * the text font is DejaVu Sans, the sans-serif font of the judge-service image.

Bins: [a, a + 5) and [95, 100] (analyze_C.score_bin, loaded read only). PASS iff score >= threshold.

Run (from the repository root, with COMPOSE_PROJECT_NAME and COMPOSE_FILE of the stack in the environment):
  python -I -B analysis/r1/figures/figure_heldout_scores.py --runs <runs> --deposit <data-archive> --out <new-dir>
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import r1_common as rc  # noqa: E402
import render_in_service as ris  # noqa: E402
import figure5_controls as f5  # noqa: E402  (the drawing code; same primitives)

W = 496.8
LM, RM, CG = 128.0, 14.0, 20.0
PITCH, BARMAX = 12.5, 10.5
COL = {"rt": "#D55E00", "exch": "#0072B2", "fail": "#009E73"}
DASH = {75.0: None, 80.0: [3, 2], 85.0: [1, 1.6]}
DPI = 600


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--deposit", required=True, help="data deposit 1.1.1 (analyze_C.score_bin, read only)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--container", default="fi-r1-judge-service")
    ap.add_argument("--outfile", default="Figure_S18.png")
    a = ap.parse_args(argv)
    runs = Path(a.runs).resolve()
    out = Path(a.out)
    env = {}
    rc.pc.load_functions(Path(a.deposit) / "analysis" / "area-C-frontier-ablation-heldout" / "analyze_C.py", ["score_bin"], env)
    score_bin = env["score_bin"]
    recs = rc.Records(runs / "records", rc.load_protocol(), runs.parent)
    series = rc.load_all_series()
    fams = rc.HB_RT + rc.HB_EXCH + rc.G60_FAMILIES
    colour = {**{f: COL["rt"] for f in rc.HB_RT}, **{f: COL["exch"] for f in rc.HB_EXCH}, **{f: COL["fail"] for f in rc.G60_FAMILIES}}
    ds_of = {**{f: rc.HB for f in rc.HB_FAMILIES}, **{f: rc.G60 for f in rc.G60_FAMILIES}}
    pw = (W - LM - RM - 2 * CG) / 3
    ph = PITCH * len(fams)
    rowh = ph + 34.0
    P = []
    for x, key, txt in ((8.0, "rt", "recovered transients (benign)"), (178.0, "exch", "exchangeable families (benign)"),
                        (355.0, "fail", "regression families")):
        P.append(["rect", x, 5.5, 7, 7, None, 0, COL[key]])
        P.append(["text", x + 10, 12, txt, 8, "normal", "left"])
    P.append(["text", 8, 27, "pass threshold:", 8, "normal", "left"])
    for T, x in ((75.0, 82.0), (80.0, 127.0), (85.0, 172.0)):
        P.append(["line", x, 24.2, x + 18, 24.2, "#000000", 1.0, DASH[T]])
        P.append(["text", x + 21, 27, f"{T:g}", 8, "normal", "left"])
    P.append(["text", 227, 27, "bar height = windows per 5-point bin (row height = 20)", 8, "normal", "left"])
    bins_out = []
    for mi, m in enumerate(rc.HOSTED):
        yb = 40.0 + mi * rowh
        top = yb + 14.0
        P.append(["text", 8, yb + 10, rc.PAPER_MODEL[m], 9, "bold", "left"])
        for pi, rub in enumerate(rc.RUBRICS):
            x0 = LM + pi * (pw + CG)
            x1 = x0 + pw
            bw = pw / 20
            P.append(["text", x0, yb + 10, f"rubric {rc.RUBRIC_SHORT[rub]}", 8.5, "normal", "left"])
            for fi, fam in enumerate(fams):
                ybase = top + (fi + 1) * PITCH
                P.append(["line", x0, ybase, x1, ybase, "#d9d9d9", 0.4, None])
                if pi == 0:
                    P.append(["text", LM - 4, ybase - 3, fam, 8, "normal", "right"])
                sc = recs.ai_scores(ds_of[fam], m, "raw", rub)
                vals = [sc[s] for s in sorted(sc) if series[ds_of[fam]][s]["family"] == fam]
                c = Counter(score_bin(float(v)) for v in vals)
                bins_out.append({"model": m, "rubric": rc.RUBRIC_SHORT[rub], "family": fam, "n": len(vals),
                                 **{f"bin_{5 * i:02d}": c.get(i, 0) for i in range(20)}})
                for i in sorted(c):
                    h = max(1.0, c[i] / 20 * BARMAX)
                    P.append(["rect", x0 + i * bw + 0.5, ybase - h, bw - 1.0, h, None, 0, colour[fam]])
            ysep = top + len(rc.HB_FAMILIES) * PITCH
            P.append(["line", x0, ysep, x1, ysep, "#808080", 0.6, None])
            for T in (75.0, 80.0, 85.0):
                xt = x0 + T / 100 * pw
                P.append(["line", xt, top, xt, top + ph, "#000000", 0.7, DASH[T]])
            P.append(["rect", x0, top, pw, ph, "#000000", 0.6, None])
            for t in (0, 25, 50, 75, 100):
                xt = x0 + t / 100 * pw
                P.append(["line", xt, top + ph, xt, top + ph + 3, "#000000", 0.6, None])
                P.append(["text", xt, top + ph + 11, str(t), 8, "normal", "center"])
    ylab = 40.0 + 2 * rowh + 14.0 + ph + 22.5
    P.append(["text", LM + (W - LM - RM) / 2, ylab, "model score (0-100); bins [a, a+5), last [95, 100]; PASS iff score ≥ threshold",
              8.5, "normal", "center"])
    P.append(["wrap", 8, ylab + 15, "Raw series, November 2026 session: heldout-benign-100 (run H3) and generalisation-60 (run H4), "
              "20 windows per family. Bins holding one window are drawn at least 1 pt tall.", 8, W - 16, 10.5, 2])
    H = ylab + 15 + 10.5 + 10
    H_in = math.ceil(H / 72.0 * 100) / 100
    spec = {"W": W, "H": H_in * 72.0, "W_in": 6.9, "H_in": H_in, "dpi": DPI, "outfile": a.outfile, "prims": P}
    info = ris.render(f5.PLOT, spec, out, a.container)
    rc.write_csv(out / (Path(a.outfile).stem + "_bins.csv"), bins_out)
    rc.write_json(out / (Path(a.outfile).stem + "_data.json"),
                  {"figure": a.outfile, "size": info, "width_in": 6.9, "height_in": H_in, "dpi": DPI,
                   "rows": fams, "records": "runs/records/H3-*/ and H4-*/ (judge=ai, representation=raw)",
                   "bins": "analyze_C.score_bin: min(int(score // 5), 19)"})
    facts = [{"key": "figS18.png", "value": f"{info[0]['width_px']} x {info[0]['height_px']} px at {info[0]['dpi']} dpi",
              "description": "supplementary figure of held-out raw-series scores by rubric: PNG size",
              "source_file": "analysis/figures/" + a.outfile, "filter_or_computation": "tools/container_plot.png_size",
              "script": "analysis/r1/figures/figure_heldout_scores.py"}]
    rc.write_json(out / (Path(a.outfile).stem + "_facts.json"), facts)
    print(json.dumps(info, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
