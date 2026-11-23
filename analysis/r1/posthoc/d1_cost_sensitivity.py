#!/usr/bin/env python3
"""D1: cost-sensitive reading of the published confusion counts (Reviewer 2, comment 1).

Expected loss per canary, as defined in the caption of Figure S17:

    L(r, p) = r * p * FNR + (1 - p) * FPR

with r the cost of a missed regression relative to a blocked healthy release, p
the regression prevalence, and FNR and FPR the miss rate and false-positive rate
measured on the 180-window benchmark and held fixed. No model is called.

What the script does, in order:
  1. rebuilds the confusion counts of the seven configurations from per-window
     verdicts of the data deposit and Archive S1;
  2. reproduces the published figures it starts from: the expected-loss
     definition (at benchmark prevalence 2/3 it must equal (r*FN + FP)/180, the
     form of results/agg/cost_ratio_comparison_set.md), the two published
     crossovers 9/64 and 5/6 of the comparison set, and the regime boundaries of
     Figure S17 at its eight prevalences (FACTS_B.md, FB1012-FB1019);
  3. finds the loss-minimising configuration on the grid r in {0.1, 0.2, 0.5,
     1, 2, 5, 10, 20, 50, 100} and p in {2/3, 0.2, 0.05, 0.01}, with exact
     crossovers per prevalence, once with all 60 benign windows and once with
     healed_transient excluded from them (FPR over the 40 NC/NE windows).

All arithmetic is exact (fractions); ties are reported, never broken silently.

Run (from the repository root):
  python -I -B analysis/r1/posthoc/d1_cost_sensitivity.py <data-archive> <archive-s1-dir> <new-output-dir>
"""

from __future__ import annotations

import sys
from fractions import Fraction
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))

import posthoc_common as pc  # noqa: E402

SCRIPT = "analysis/r1/posthoc/d1_cost_sensitivity.py"
R_GRID = ["0.1", "0.2", "0.5", "1", "2", "5", "10", "20", "50", "100"]
P_GRID = [("2/3", Fraction(2, 3)), ("0.2", Fraction(1, 5)), ("0.05", Fraction(1, 20)), ("0.01", Fraction(1, 100))]
D1_SET = ["statistical", "ensemble:tuned", "rule:guard", "bfks", "ai:raw:claude-opus-4-8-vlm",
          "ai:summary:phi4-llm", "hybrid:gated:phi4-llm"]
SHORT = {"statistical": "statistical judge", "ensemble:tuned": "tuned ensemble", "rule:guard": "guard rule",
         "bfks": "BF+KS", "ai:raw:claude-opus-4-8-vlm": "Claude Opus 4.8 (raw)",
         "ai:summary:phi4-llm": "Phi-4 (summary)", "hybrid:gated:phi4-llm": "hybrid as scored, Phi-4",
         "hybrid:guarded:mistral-nemo-llm": "guarded hybrid, Mistral-Nemo",
         "hybrid:guarded:phi4-llm": "guarded hybrid, Phi-4"}
# Published expectations the script must reproduce.
PUBLISHED_COUNTS = {  # TP, FP, TN, FN (Table 8; Table S38; cost_ratio_comparison_set.csv; FACTS_B FB1006-FB1011)
    "statistical": (38, 0, 60, 82), "ensemble:tuned": (102, 9, 51, 18), "rule:guard": (114, 12, 48, 6),
    "bfks": (109, 20, 40, 11), "ai:raw:claude-opus-4-8-vlm": (120, 24, 36, 0),
    "ai:summary:phi4-llm": (117, 27, 33, 3), "hybrid:gated:phi4-llm": (95, 20, 40, 25),
    "hybrid:guarded:mistral-nemo-llm": (93, 2, 58, 27), "hybrid:guarded:phi4-llm": (89, 0, 60, 31)}
PUBLISHED_COMPARISON_SET = "statistical for r in [0, 9/64); ensemble for r in [9/64, 5/6); frontier for r in [5/6, inf)"
PUBLISHED_FIG_S17 = {  # FACTS_B.md FB1012-FB1019: prevalence -> boundaries (statistical|ensemble, ensemble|guard, guard|frontier)
    Fraction(1, 100): ("891/32", "99/2", "396"), Fraction(1, 50): ("441/32", "49/2", "196"),
    Fraction(1, 20): ("171/32", "19/2", "76"), Fraction(1, 10): ("81/32", "9/2", "36"),
    Fraction(1, 5): ("9/8", "2", "16"), Fraction(1, 3): ("9/16", "1", "8"),
    Fraction(1, 2): ("9/32", "1/2", "4"), Fraction(2, 3): ("9/64", "1/4", "2")}


def rates(cfg, cid, labels, benign_keep):
    v = cfg[cid].verdicts
    lab = {s: t for s, t in labels.items() if s in v and (t == "FAIL" or benign_keep(s))}
    tp, fp, tn, fn = pc.confusion(v, lab)
    return {"TP": tp, "FP": fp, "TN": tn, "FN": fn, "FNR": Fraction(fn, tp + fn), "FPR": Fraction(fp, fp + tn)}


def loss(r: Fraction, p: Fraction, q) -> Fraction:
    return r * p * q["FNR"] + (1 - p) * q["FPR"]


def envelope_at(env_fn, members, q, p):
    """Lower envelope over r >= 0 of L/p = r*FNR + ((1-p)/p)*FPR, via the archived area-B envelope()."""
    lines = [(m, q[m]["FNR"], (1 - p) / p * q[m]["FPR"]) for m in members]
    # merge identical lines so that a tie is reported as one segment naming all members
    merged = {}
    for nm, a, b in lines:
        merged.setdefault((a, b), []).append(nm)
    lines = [(" = ".join(v), k[0], k[1]) for k, v in sorted(merged.items())]
    return env_fn(lines)


def env_text(env, names=None):
    names = names or {}
    out = []
    for i, (r, nm) in enumerate(env):
        hi = env[i + 1][0] if i + 1 < len(env) else None
        label = " = ".join(names.get(x, x) for x in nm.split(" = "))
        out.append(f"{label} for r in [{pc.frac_str(r)}, {pc.frac_str(hi) if hi is not None else 'inf'})")
    return "; ".join(out)


def main() -> int:
    if len(sys.argv) != 4:
        print(__doc__)
        return 2
    arch, s1, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    out.mkdir(parents=True, exist_ok=True)  # existing files are never overwritten (write_* refuse)
    F = pc.Facts(SCRIPT)
    series = pc.load_series(arch)
    labels = {s: d["label"] for s, d in series.items()}
    fam = {s: d["family"] for s, d in series.items()}
    cfg = pc.load_configs(arch, s1, series)
    extra = pc.add_reimplemented(cfg, arch, series)
    B = pc.area_b(arch)
    envelope = B["envelope"]

    everyone = lambda s: True  # noqa: E731
    no_ht = lambda s: fam[s] != "healed_transient"  # noqa: E731
    members = D1_SET + ["hybrid:guarded:mistral-nemo-llm", "hybrid:guarded:phi4-llm"]
    q_all = {m: rates(cfg, m, labels, everyone) for m in members}
    q_noht = {m: rates(cfg, m, labels, no_ht) for m in members}

    # ---- 1. confusion counts and their reproduction
    conf_rows, repro = [], []
    for m in members:
        a, b = q_all[m], q_noht[m]
        conf_rows.append({"config": m, "label": cfg[m].label, "source": cfg[m].source, "filter": cfg[m].filt,
                          "TP": a["TP"], "FP": a["FP"], "TN": a["TN"], "FN": a["FN"],
                          "FNR": pc.frac_str(a["FNR"]), "FPR": pc.frac_str(a["FPR"]),
                          "FP_without_HT": b["FP"], "TN_without_HT": b["TN"], "FPR_without_HT": pc.frac_str(b["FPR"]),
                          "in_D1_set": "yes" if m in D1_SET else "no (Table S38 only)"})
        got = (a["TP"], a["FP"], a["TN"], a["FN"])
        repro.append({"check": f"confusion counts TP/FP/TN/FN, {SHORT[m]}",
                      "published": "/".join(map(str, PUBLISHED_COUNTS[m])), "reproduced": "/".join(map(str, got)),
                      "match": "yes" if got == PUBLISHED_COUNTS[m] else "NO"})
        F.add(f"d1.conf.{m}", f"TP {a['TP']}, FP {a['FP']}, TN {a['TN']}, FN {a['FN']}; without healed_transient "
              f"FP {b['FP']} of {b['FP'] + b['TN']}",
              f"confusion counts on the 180-window benchmark, {SHORT[m]} (FAIL positive), and its false alarms on the "
              f"40 NC/NE windows", cfg[m].source, cfg[m].filt)
    repro.append({"check": "re-implemented statistical judge at allowedIncrease 1.05 agrees with the recorded verdicts",
                  "published": "180/180 (Section 4.1)", "reproduced": f"{extra['reimpl_agree_1.05']}/180",
                  "match": "yes" if extra["reimpl_agree_1.05"] == 180 else "NO"})

    # ---- 2. the expected-loss definition: at p = 2/3, r*p*FNR + (1-p)*FPR == (r*FN + FP)/180
    p_b = Fraction(2, 3)
    ident = all(loss(Fraction(r), p_b, q_all[m]) == (Fraction(r) * q_all[m]["FN"] + q_all[m]["FP"]) / 180
                for m in members for r in R_GRID)
    repro.append({"check": "Figure S17 caption loss r*p*FNR + (1-p)*FPR at p = 2/3 equals (r*FN + FP)/180 of "
                           "cost_ratio_comparison_set.md, every configuration and every r of the grid",
                  "published": "identity", "reproduced": "holds" if ident else "fails", "match": "yes" if ident else "NO"})
    F.add("d1.repro.loss_identity", "holds" if ident else "fails",
          "the expected loss of the Figure S17 caption, r*p*FNR + (1-p)*FPR, at benchmark prevalence p = 2/3 equals "
          "(r*FN + FP)/180, the count form of data-archive/results/agg/cost_ratio_comparison_set.md, for all nine "
          "configurations and all ten r of the grid", "data-archive/results/agg/cost_ratio_comparison_set.md",
          "exact rational arithmetic")

    # ---- 3. the published crossovers 9/64 and 5/6
    comp = {"statistical": "statistical", "ensemble:tuned": "ensemble", "ai:raw:claude-opus-4-8-vlm": "frontier",
            "hybrid:gated:phi4-llm": "hybrid"}
    env_c = envelope_at(envelope, list(comp), q_all, p_b)
    txt_c = env_text(env_c, comp)
    repro.append({"check": "loss-minimising regimes of the four-member comparison set at p = 2/3 "
                           "(data-archive/results/agg/cost_ratio_comparison_set.csv; FACTS_B FB490)",
                  "published": PUBLISHED_COMPARISON_SET, "reproduced": txt_c,
                  "match": "yes" if txt_c == PUBLISHED_COMPARISON_SET else "NO"})
    F.add("d1.repro.crossovers_comparison_set", txt_c,
          "loss-minimising configuration by cost ratio r at benchmark prevalence 2/3 over the published comparison set "
          "(statistical judge, tuned ensemble, Claude Opus 4.8 raw, hybrid as scored with Phi-4); the published "
          "crossovers are 9/64 and 5/6", "data-archive/results/agg/cost_ratio_comparison_set.csv",
          "lower envelope of r*FNR + ((1-p)/p)*FPR with the archived area-B envelope()")
    s38_unguarded = ["statistical", "ai:summary:phi4-llm", "ai:raw:claude-opus-4-8-vlm", "hybrid:gated:phi4-llm",
                     "ensemble:tuned"]
    env_s38 = envelope_at(envelope, s38_unguarded, q_all, p_b)
    txt_s38 = env_text(env_s38, SHORT)
    ok_s38 = [pc.frac_str(r) for r, _ in env_s38] == ["0", "9/64", "5/6"]
    repro.append({"check": "the same at p = 2/3 over the five configurations of Table S38 without the guarded hybrids",
                  "published": "crossovers 9/64 and 5/6", "reproduced": txt_s38, "match": "yes" if ok_s38 else "NO"})
    F.add("d1.repro.crossovers_s38_unguarded", txt_s38,
          "loss-minimising configuration by r at p = 2/3 over the five unguarded configurations of Table S38 "
          "(adds Phi-4 on summaries, which Claude Opus 4.8 raw dominates)", "posthoc/d1_confusion.csv",
          "lower envelope, exact")
    s38_all = s38_unguarded + ["hybrid:guarded:mistral-nemo-llm", "hybrid:guarded:phi4-llm"]
    env_s38a = envelope_at(envelope, s38_all, q_all, p_b)
    txt_s38a = env_text(env_s38a, SHORT)
    F.add("d1.s38_all_seven", txt_s38a,
          "loss-minimising configuration by r at p = 2/3 over all seven rows of Table S38, including the two guarded "
          "hybrids (designed on this benchmark; not activated by the deployed service)", "posthoc/d1_confusion.csv",
          "lower envelope, exact")
    fig6 = ["statistical", "ensemble:tuned", "rule:guard", "ai:raw:claude-opus-4-8-vlm", "ai:summary:phi4-llm",
            "hybrid:gated:phi4-llm"]
    fig_ok = 0
    for p, exp in PUBLISHED_FIG_S17.items():
        env = envelope_at(envelope, fig6, q_all, p)
        got = tuple(pc.frac_str(r) for r, _ in env[1:])
        names = [nm for _, nm in env]
        good = got == exp and names == ["statistical", "ensemble:tuned", "rule:guard", "ai:raw:claude-opus-4-8-vlm"]
        fig_ok += int(good)
        repro.append({"check": f"Figure S17 regime boundaries at prevalence {pc.frac_str(p)} (FACTS_B FB1012-FB1019)",
                      "published": ", ".join(exp), "reproduced": ", ".join(got), "match": "yes" if good else "NO"})
    F.add("d1.repro.figS17", f"{fig_ok}/8",
          "Figure S17 regime boundaries (statistical judge | tuned ensemble | guard rule | Claude Opus 4.8 raw) reproduced "
          "exactly at the eight published prevalences 1%, 2%, 5%, 10%, 20%, 33.3%, 50% and 66.7%",
          "data-archive/analysis/area-B-representation-cost/FACTS_B.md", "lower envelope over the six configurations of Figure S17")

    # ---- 4. the grid, full benign set and healed_transient excluded
    grid_rows, cross_rows = [], []
    for setname, q in (("all 60 benign windows", q_all), ("healed_transient excluded (40 benign windows)", q_noht)):
        tag = "all" if q is q_all else "noHT"
        for plabel, p in P_GRID:
            env = envelope_at(envelope, D1_SET, q, p)
            for i, (r0, nm) in enumerate(env):
                r1 = env[i + 1][0] if i + 1 < len(env) else None
                cross_rows.append({"benign_set": setname, "prevalence": plabel, "segment": i + 1,
                                   "loss_minimising": " = ".join(SHORT[x] for x in nm.split(" = ")),
                                   "r_from": pc.frac_str(r0), "r_from_decimal": f"{float(r0):.4g}",
                                   "r_to": pc.frac_str(r1) if r1 is not None else "inf",
                                   "r_to_decimal": f"{float(r1):.4g}" if r1 is not None else "inf"})
            F.add(f"d1.env.{tag}.p{plabel}", env_text(env, SHORT),
                  f"loss-minimising configuration by cost ratio r at prevalence {plabel}, {setname}, over the seven "
                  f"configurations of D1 (exact crossovers)", "posthoc/d1_confusion.csv",
                  "lower envelope of r*FNR + ((1-p)/p)*FPR, per-class rates held fixed")
            for r in R_GRID:
                rr = Fraction(r)
                ls = {m: loss(rr, p, q[m]) for m in D1_SET}
                best = min(ls.values())
                winners = [m for m in D1_SET if ls[m] == best]
                row = {"benign_set": setname, "prevalence": plabel, "r": r,
                       "loss_minimising": " = ".join(SHORT[m] for m in winners), "tie": "yes" if len(winners) > 1 else "no",
                       "min_loss": f"{float(best):.6f}"}
                for m in D1_SET:
                    row[f"loss[{SHORT[m]}]"] = f"{float(ls[m]):.6f}"
                grid_rows.append(row)
                F.add(f"d1.grid.{tag}.p{plabel}.r{r}", " = ".join(SHORT[m] for m in winners),
                      f"loss-minimising configuration at r = {r}, prevalence {plabel}, {setname}"
                      + (" (exact tie)" if len(winners) > 1 else ""), "posthoc/d1_grid.csv",
                      "argmin over the seven D1 configurations of r*p*FNR + (1-p)*FPR, exact")

    # ---- 5. statements that do not depend on r or p: dominance
    def dominated(q, m):
        return [o for o in D1_SET if o != m and q[o]["FNR"] <= q[m]["FNR"] and q[o]["FPR"] <= q[m]["FPR"]
                and (q[o]["FNR"] < q[m]["FNR"] or q[o]["FPR"] < q[m]["FPR"])]
    dom_rows = []
    for setname, q, tag in (("all 60 benign windows", q_all, "all"), ("healed_transient excluded", q_noht, "noHT")):
        for m in D1_SET:
            d = dominated(q, m)
            dom_rows.append({"benign_set": setname, "config": SHORT[m],
                             "dominated_by": "; ".join(SHORT[o] for o in d) or "none",
                             "can_minimise_loss": "no (dominated)" if d else "yes, for some r > 0 at every p"
                             if any(m in nm.split(" = ") for _, nm in envelope_at(envelope, D1_SET, q, p_b)) else
                             "no (never on the lower envelope)"})
        never = [SHORT[m] for m in D1_SET if not any(m in nm.split(" = ") for _, nm in envelope_at(envelope, D1_SET, q, p_b))]
        F.add(f"d1.never.{tag}", "; ".join(never),
              f"configurations of the D1 set that minimise expected loss at no cost ratio and no prevalence, {setname} "
              f"(the envelope order does not depend on p because every boundary scales by 2(1-p)/p)",
              "posthoc/d1_confusion.csv", "lower envelope; dominance on (FNR, FPR)")
    scale_ok = True
    for q in (q_all, q_noht):
        e0 = envelope_at(envelope, D1_SET, q, p_b)
        for _, p in P_GRID:
            e = envelope_at(envelope, D1_SET, q, p)
            k = 2 * (1 - p) / p
            scale_ok &= [nm for _, nm in e] == [nm for _, nm in e0] and all(r == r0 * k for (r, _), (r0, _) in zip(e, e0))
    F.add("d1.scaling", "holds" if scale_ok else "fails",
          "at every prevalence of the grid and for both benign sets, the loss-minimising configurations occur in the same "
          "order as at p = 2/3 and every crossover equals the p = 2/3 crossover times 2(1-p)/p",
          "posthoc/d1_crossovers.csv", "exact comparison of envelopes")

    pc.write_csv(out / "d1_confusion.csv", conf_rows)
    pc.write_csv(out / "d1_grid.csv", grid_rows)
    pc.write_csv(out / "d1_crossovers.csv", cross_rows)
    pc.write_csv(out / "d1_dominance.csv", dom_rows)
    pc.write_csv(out / "d1_reproduction.csv", repro)
    pc.write_json(out / "d1_facts.json", F.items)
    pc.write_json(out / "d1_run.json", {"script": SCRIPT, "code_sha256": pc.code_hashes(arch),
                                        "inputs_sha256": {f"data-archive/{pc.SERIES_REL}": pc.sha256_file(arch / pc.SERIES_REL),
                                                          f"data-archive/{pc.CORRECTED_REL}": pc.sha256_file(arch / pc.CORRECTED_REL),
                                                          f"data-archive/{pc.ENSEMBLE_REL}": pc.sha256_file(arch / pc.ENSEMBLE_REL),
                                                          f"archive-s1/{pc.BFKS_FILE}": pc.sha256_file(s1 / pc.BFKS_FILE)},
                                        "python": sys.version.split()[0]})
    bad = [r for r in repro if r["match"] != "yes"]
    print(f"D1: {len(repro) - len(bad)}/{len(repro)} reproduction checks passed")
    for r in bad:
        print("  MISMATCH:", r["check"], "| published:", r["published"], "| reproduced:", r["reproduced"])
    return 0 if not bad else 1


if __name__ == "__main__":
    raise SystemExit(main())
