#!/usr/bin/env python3
"""Inventory of the runs (step 1 of the analysis).

Reads the runs folder (STATUS.json, markers/<id>.DONE, records/<id>/*.csv, ledgers/) and writes, per invocation and
per configuration (model x representation x rubric x dataset): judgements expected, successful, failed calls under
the protocol's failed-call rule, outages and attempts. One record per judgement key is the successful call; rows of
later attempts that repeat a key are cache hits and must carry the same verdict and score (checked). Hosted token
usage per model and invocation is summed from the tokens_in and tokens_out columns of the records.

Only runs/records is read for counts; the per-call payload records of the judge service in data/ai-logs
(which include four stub-model records written by the judge-service mock test) are not read.

Run (from the repository root, analysis interpreter with -I):
  python -I -B analysis/r1/inventory.py --runs <runs> --out <new-dir> [--manifest <runs-manifest.tsv>]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))

import r1_common as rc  # noqa: E402

TOKENS_RX = re.compile(r"request \((\d+) tokens\) exceeds the available context size \((\d+) tokens\)")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", required=True, help="the runs folder")
    ap.add_argument("--out", required=True, help="new output folder (files are never overwritten)")
    ap.add_argument("--manifest", help="runs manifest (path, bytes, sha256) written at the end of the runs")
    a = ap.parse_args(argv)
    runs = Path(a.runs).resolve()
    base = runs.parent
    out = Path(a.out)
    protocol = rc.load_protocol()
    inv_tab = rc.invocation_table(protocol)
    status = json.loads((runs / "STATUS.json").read_text(encoding="utf-8"))
    st_inv = {i["id"]: i for i in status["invocations"]}
    recs = rc.Records(runs / "records", protocol, base)
    checks = {}

    # ---- manifest: the records must not have changed since the end of the runs
    if a.manifest:
        man = {}
        for ln in Path(a.manifest).read_text(encoding="utf-8").splitlines()[1:]:
            p, b, h = ln.split("\t")
            man[p] = (int(b), h)
        rec_entries = {p: v for p, v in man.items() if p.startswith("runs/records/")}
        same = 0
        diff = []
        for p, (b, h) in sorted(rec_entries.items()):
            f = base / p
            if f.exists() and f.stat().st_size == b and rc.sha256_file(f) == h:
                same += 1
            else:
                diff.append(p)
        present = {rc.rel_to(f, base, "") for _, f in recs.files}
        extra = sorted(present - set(rec_entries))
        checks["records_unchanged_against_manifest"] = {"manifest_record_files": len(rec_entries), "identical": same,
                                                        "different_or_missing": diff, "files_not_in_manifest": extra}

    # ---- per invocation
    inv_rows = []
    for iid, it in inv_tab.items():
        rows = [r for r in recs.rows if r["_inv"] == iid]
        ai_rows = [r for r in rows if r["judge"] == "ai"]
        keys_ok = {(r["dataset_id"], r["scenario_id"], r["model"], r["representation"], r["_rubric"])
                   for r in ai_rows if not r["error_kind"].strip()}
        keys_err = {(r["dataset_id"], r["scenario_id"], r["model"], r["representation"], r["_rubric"])
                    for r in ai_rows if r["error_kind"].strip()}
        failed = keys_err - keys_ok
        marker = runs / "markers" / f"{iid}.DONE"
        mk = json.loads(marker.read_text(encoding="utf-8")) if marker.exists() else {}
        s = st_inv.get(iid, {})
        ledger = runs / "ledgers" / f"{iid}.classified.jsonl"
        classified = [json.loads(x) for x in ledger.read_text(encoding="utf-8").splitlines() if x.strip()] if ledger.exists() else []
        n_attempt_files = len([1 for i2, _ in recs.files if i2 == iid])
        inv_rows.append({
            "invocation": iid, "run": it["run"], "stream": it["stream"], "model": it["model"],
            "representations": ",".join(it["representations"]), "rubric": it["rubric"], "dataset": it["dataset"],
            "windows": it["windows"], "judgements_expected": it["judgements_expected"],
            "judgements_successful": len(keys_ok), "failed_calls": len(failed),
            "ai_rows": len(ai_rows), "ai_error_rows": sum(1 for r in ai_rows if r["error_kind"].strip()),
            "statistical_rows": sum(1 for r in rows if r["judge"] == "statistical"),
            "statistical_error_rows": sum(1 for r in rows if r["judge"] == "statistical" and r["error_kind"].strip()),
            "hybrid_rows": sum(1 for r in rows if r["judge"] == "hybrid"),
            "record_files": n_attempt_files, "attempts_status": s.get("attempts", ""),
            "status": s.get("status", ""), "start": s.get("start", ""), "end": s.get("end", ""),
            "outages": len(s.get("outages") or []), "marker_DONE": "yes" if marker.exists() else "no",
            "marker_written": mk.get("judgements_written", ""), "marker_failed": mk.get("judgements_failed", ""),
            "status_written": s.get("judgements_written", ""), "status_failed": s.get("judgements_failed", ""),
            "ledger_failed_call_entries": sum(1 for c in classified if c.get("class") == "failed_call"),
            "ledger_outage_entries": sum(1 for c in classified if c.get("class") != "failed_call"),
            "consistent_with_status_and_marker": "yes" if (len(keys_ok) == s.get("judgements_written") == mk.get("judgements_written")
                                                          and len(failed) == s.get("judgements_failed") == mk.get("judgements_failed")
                                                          and len(keys_ok) + len(failed) == it["judgements_expected"]) else "NO",
        })
    rc.write_csv(out / "inventory_invocations.csv", inv_rows)

    # ---- per configuration (model x representation x rubric x dataset)
    conf = defaultdict(lambda: {"expected": 0, "ok": 0, "failed": 0, "failed_ids": [], "invocations": set()})
    for iid, it in inv_tab.items():
        for rep in it["representations"]:
            k = (it["dataset"], it["model"], rep, it["rubric"])
            conf[k]["expected"] += it["windows"]
            conf[k]["invocations"].add(iid)
    for k, r in recs.ai_ok.items():
        conf[(k[0], k[2], k[3], k[4])]["ok"] += 1
    for k in recs.ai_failed:
        c = conf[(k[0], k[2], k[3], k[4])]
        c["failed"] += 1
        c["failed_ids"].append(k[1])
    conf_rows = []
    for (ds, m, rep, rub), c in sorted(conf.items(), key=lambda kv: (kv[0][0], rc.MODELS.index(kv[0][1]), kv[0][2], kv[0][3])):
        conf_rows.append({"dataset": ds, "model": m, "representation": rep, "rubric": rub,
                          "invocations": ",".join(sorted(c["invocations"])), "judgements_expected": c["expected"],
                          "scored": c["ok"], "failed_calls": c["failed"],
                          "failed_windows": ";".join(sorted(c["failed_ids"]))})
    rc.write_csv(out / "inventory_configurations.csv", conf_rows)

    # ---- failed calls, with the request size from the error text (no other part of the text is copied)
    fc_rows = []
    for k, rows in sorted(recs.ai_failed.items()):
        r = rows[0]
        mt = TOKENS_RX.search(r["error"])
        fc_rows.append({"invocation": r["_inv"], "dataset": k[0], "scenario_id": k[1], "family": r["family"],
                        "model": k[2], "representation": k[3], "rubric": k[4], "error_kind": r["error_kind"],
                        "request_tokens": mt.group(1) if mt else "", "context_tokens": mt.group(2) if mt else "",
                        "error_rows_for_key": len(rows), "record_file": r["_file"]})
    rc.write_csv(out / "failed_calls.csv", fc_rows)

    # ---- cache-hit duplicates
    dup_ai = sum(1 for rows in recs.ai.values() if len([r for r in rows if not r["error_kind"].strip()]) > 1)
    dup_hyb = sum(1 for rows in recs.hyb.values() if len([r for r in rows if not r["error_kind"].strip()]) > 1)
    checks["cache_hits"] = {"ai_keys_with_more_than_one_successful_row": dup_ai,
                            "hybrid_keys_with_more_than_one_successful_row": dup_hyb,
                            "differences_in_verdict_or_score": recs.cache_differences}

    # ---- hosted token usage (sums over the scored ai rows of the records)
    tok = defaultdict(lambda: [0, 0, 0, 0])
    for r in recs.rows:
        if r["judge"] != "ai" or r["model"] not in rc.HOSTED:
            continue
        for key in ((r["_run"], r["model"]), ("ALL", r["model"]), ("ALL", "ALL")):
            t = tok[key]
            t[0] += 1
            if r["tokens_in"].strip():
                t[1] += 1
                t[2] += int(float(r["tokens_in"]))
                t[3] += int(float(r["tokens_out"]))
    tok_rows = [{"run": k[0], "model": k[1], "ai_rows": v[0], "rows_with_tokens": v[1], "tokens_in": v[2],
                 "tokens_out": v[3]} for k, v in sorted(tok.items(), key=lambda kv: (kv[0][0] == "ALL", kv[0]))]
    rc.write_csv(out / "token_usage_hosted.csv", tok_rows)

    # ---- totals
    tot = Counter()
    for r in inv_rows:
        tot[(r["stream"], "expected")] += r["judgements_expected"]
        tot[(r["stream"], "successful")] += r["judgements_successful"]
        tot[(r["stream"], "failed")] += r["failed_calls"]
        tot[(r["stream"], "outages")] += r["outages"]
    by_run = defaultdict(Counter)
    for r in inv_rows:
        by_run[r["run"]]["expected"] += r["judgements_expected"]
        by_run[r["run"]]["successful"] += r["judgements_successful"]
        by_run[r["run"]]["failed"] += r["failed_calls"]
        by_run[r["run"]]["outages"] += r["outages"]
        by_run[r["run"]]["invocations"] += 1
        by_run[r["run"]]["start"] = min(by_run[r["run"]].get("start") or "9", r["start"]) if r["start"] else by_run[r["run"]].get("start")
        by_run[r["run"]]["end"] = max(by_run[r["run"]].get("end") or "", r["end"])
    summary = {
        "overall_status": status["overall"]["status"], "driver_started": status["overall"].get("driver_started"),
        "finished": status["overall"].get("finished"),
        "invocations": len(inv_rows), "invocations_DONE": sum(1 for r in inv_rows if r["status"] == "DONE"),
        "markers_DONE": sum(1 for r in inv_rows if r["marker_DONE"] == "yes"),
        "attempts_max": max(int(r["attempts_status"] or 0) for r in inv_rows),
        "record_files": len(recs.files),
        "hosted": {"expected": tot[("hosted", "expected")], "successful": tot[("hosted", "successful")],
                   "failed": tot[("hosted", "failed")], "outages": tot[("hosted", "outages")]},
        "local": {"expected": tot[("local", "expected")], "successful": tot[("local", "successful")],
                  "failed": tot[("local", "failed")], "outages": tot[("local", "outages")]},
        "by_run": {k: dict(v) for k, v in sorted(by_run.items())},
        "all_invocations_consistent_with_status_and_marker": all(r["consistent_with_status_and_marker"] == "yes" for r in inv_rows),
        "rows": {"ai": sum(r["ai_rows"] for r in inv_rows), "statistical": sum(r["statistical_rows"] for r in inv_rows),
                 "hybrid": sum(r["hybrid_rows"] for r in inv_rows)},
        "failed_call_request_tokens": {"min": min(int(r["request_tokens"]) for r in fc_rows) if fc_rows else None,
                                       "max": max(int(r["request_tokens"]) for r in fc_rows) if fc_rows else None,
                                       "context": sorted({r["context_tokens"] for r in fc_rows})},
        "checks": checks,
        "inputs": {"status": "runs/STATUS.json", "markers": "runs/markers/", "records": "runs/records/",
                   "ledgers": "runs/ledgers/", "protocol": rc.PROTOCOL_REL},
    }
    rc.write_json(out / "inventory_summary.json", summary)
    print(json.dumps({k: summary[k] for k in ("overall_status", "invocations", "hosted", "local", "rows",
                                             "all_invocations_consistent_with_status_and_marker")}, indent=1))
    print(json.dumps(checks, indent=1)[:2000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
