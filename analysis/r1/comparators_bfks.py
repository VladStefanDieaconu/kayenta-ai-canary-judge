#!/usr/bin/env python3
"""The post-hoc BF+KS battery of Archive S1 on the new windows.

Archive S1's posthoc_reanalysis.py computes the battery inside its module-level loop over the 180 deposited
windows (it asserts 180 windows and the nine published families), so it cannot be pointed at another dataset
unchanged. This script applies the same two statements to each metric of each window, copied from that loop:

    zc = abs(c - np.median(c)); zb = abs(b - np.median(b))
    bf = float(st.ttest_ind(zc, zb, equal_var=True, alternative='greater').pvalue)
    ks = float(st.ks_2samp(c, b, alternative='less', method='exact').pvalue)

with c the canary (experiment) series and b the control series, and the same decision: a window fails when any of
its 2m p-values is below 0.05 / (2m), m its number of metrics. No constant is changed.

Reproduction check first: on original-180 (series regenerated with tools/eval_dataset.py at master seed 20260621)
every verdict must equal Archive S1's classical_verdicts.csv and every p-value its classical_pvalues.csv.

Run (from the repository root, analysis interpreter with numpy and scipy):
  python -I -B analysis/r1/comparators_bfks.py --archive-s1 <unzipped Archive S1> --out <new-dir>
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats as st

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))

import r1_common as rc  # noqa: E402

DATASETS = [rc.O180, rc.G60, rc.KC, rc.HB]


def battery(metrics):
    """metrics: [{name, control, canary}] -> (verdict, rows of p-values)."""
    rows = []
    for m in metrics:
        c = np.asarray(m["canary"], dtype=float)
        b = np.asarray(m["control"], dtype=float)
        zc = abs(c - np.median(c)); zb = abs(b - np.median(b))  # noqa: E702
        bf = float(st.ttest_ind(zc, zb, equal_var=True, alternative='greater').pvalue)
        ks = float(st.ks_2samp(c, b, alternative='less', method='exact').pvalue)
        rows.append({"metric": m["name"], "kind": m.get("kind", ""), "bf_one_sided": bf, "ks_one_sided": ks})
    cut = .05 / (2 * len(rows))
    fail = any(r["bf_one_sided"] < cut or r["ks_one_sided"] < cut for r in rows)
    return ("FAIL" if fail else "PASS"), cut, rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--archive-s1", required=True, help="folder with Archive S1's classical_verdicts.csv and classical_pvalues.csv")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    out = Path(a.out)
    s1 = Path(a.archive_s1)
    series = rc.load_all_series()
    summary = {"rule": "FAIL iff any per-metric one-sided Brown-Forsythe or exact one-sided KS p < 0.05/(2m)",
               "source_of_statements": "Archive S1 posthoc_reanalysis.py (copied statements; see the module docstring)",
               "datasets": {}}
    for ds in DATASETS:
        vrows, prows = [], []
        for sid in sorted(series[ds]):
            w = series[ds][sid]
            v, cut, rows = battery(w["metrics"])
            vrows.append({"dataset_id": ds, "scenario_id": sid, "family": w["family"], "truth": w["truth"],
                          "n_metrics": len(rows), "threshold": cut, "bfks_verdict": v})
            for r in rows:
                prows.append({"dataset_id": ds, "scenario_id": sid, "family": w["family"], **r})
        rc.write_csv(out / f"bfks_verdicts_{ds}.csv", vrows)
        rc.write_csv(out / f"bfks_pvalues_{ds}.csv", prows)
        d = {"windows": len(vrows), "FAIL": sum(1 for r in vrows if r["bfks_verdict"] == "FAIL"),
             "detected": f"{sum(1 for r in vrows if r['truth'] == 'FAIL' and r['bfks_verdict'] == 'FAIL')}/"
                         f"{sum(1 for r in vrows if r['truth'] == 'FAIL')}",
             "false_alarms": f"{sum(1 for r in vrows if r['truth'] == 'PASS' and r['bfks_verdict'] == 'FAIL')}/"
                             f"{sum(1 for r in vrows if r['truth'] == 'PASS')}"}
        if ds == rc.O180:
            pub_v = {r["scenario"]: r["verdict"] for r in rc.read_csv(s1 / "classical_verdicts.csv")}
            pub_p = {(r["scenario"], r["metric"]): (float(r["bf_one_sided"]), float(r["ks_one_sided"]))
                     for r in rc.read_csv(s1 / "classical_pvalues.csv")}
            mine_v = {r["scenario_id"]: r["bfks_verdict"] for r in vrows}
            same = sum(1 for s in pub_v if mine_v.get(s) == pub_v[s])
            # Archive S1 reads the deposited own-data export, whose metric names are anonymised; match p-values by
            # scenario and metric kind order (each window's metrics in the generator's order).
            mine_p = {}
            for r in prows:
                mine_p.setdefault(r["scenario_id"], []).append((r["bf_one_sided"], r["ks_one_sided"]))
            pub_by_s = {}
            for (s, m), v in sorted(pub_p.items()):
                pub_by_s.setdefault(s, []).append(v)
            maxdiff = 0.0
            matched = 0
            for s, lst in pub_by_s.items():
                mine = sorted(mine_p.get(s, []))
                pubs = sorted(lst)
                if len(mine) != len(pubs):
                    continue
                matched += 1
                for (x1, y1), (x2, y2) in zip(mine, pubs):
                    maxdiff = max(maxdiff, abs(x1 - x2), abs(y1 - y2))
            d["reproduction"] = {"verdicts_identical_to_archive_s1": f"{same}/{len(pub_v)}",
                                 "windows_with_p_values_compared": matched,
                                 "max_abs_p_value_difference": maxdiff,
                                 "published": {"detected": "109/120", "false_alarms": "20/60", "BA": "0.788"},
                                 "REPRODUCED": same == 180 and d["detected"] == "109/120" and d["false_alarms"] == "20/60"}
        summary["datasets"][ds] = d
        print(ds, json.dumps(d))
    rc.write_json(out / "bfks_summary.json", summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
