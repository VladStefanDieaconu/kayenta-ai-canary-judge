#!/usr/bin/env python3
"""Metric naming and canary-config text for the scenario datasets. No stack, no model.

  1. The published datasets keep their gid-suffixed metric names and seeds, so every
     archived result stays keyed to the same scenario.
  2. A dataset registered without `metric_suffix` gets opaque tokens: distinct,
     reproducible from the token seed, not sequential in family order, and
     refused without a token seed.
  3. The per-scenario canary config names neither the family nor the label unless
     legacy_description=True.

Usage:
  python tools/test_dataset_naming.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import eval_dataset as ed  # noqa: E402


def main() -> int:
    failures = []

    def check(name, cond):
        print(f"  {name:66s} {'ok' if cond else 'FAIL'}")
        if not cond:
            failures.append(name)

    scs = ed.build_scenarios(20260621, 20, "original-180")
    check("original-180: 180 scenarios", len(scs) == 180)
    check("original-180: first metric keeps _g0000", scs[0].metrics[0].name == "http_request_duration_seconds_g0000")
    check("original-180: every name ends in its gid",
          all(m.name.endswith(f"_g{s.gid:04d}") for s in scs for m in s.metrics))
    check("original-180: no token", all(s.token == "" for s in scs))
    gen = ed.build_scenarios(20260621, 20, "generalisation-60")
    check("generalisation-60: first metric keeps _g1000", gen[0].metrics[0].name.endswith("_g1000"))

    cfg = ed.build_canary_config(scs[0], {})
    check("config description names neither family nor truth",
          "family=" not in cfg["description"] and "truth=" not in cfg["description"]
          and scs[0].family not in cfg["description"] and scs[0].family not in cfg["name"])
    legacy = ed.build_canary_config(scs[0], {}, legacy_description=True)
    check("legacy_description reproduces the v1.0.0 text",
          legacy["description"] == f"eval dataset scenario {scs[0].id} (family={scs[0].family}, truth={scs[0].truth})"
          and legacy["name"] == f"eval-{scs[0].id}")
    check("config differs only in name and description",
          {k: v for k, v in cfg.items() if k not in ("name", "description")}
          == {k: v for k, v in legacy.items() if k not in ("name", "description")})

    ed.DATASETS["_naming-test"] = {"spec": ed.FAMILY_SPEC, "families": ed.FAMILIES, "gid_base": 9000}
    try:
        try:
            ed.build_scenarios(20260621, 5, "_naming-test")
            check("opaque dataset refuses to build without a token seed", False)
        except ValueError:
            check("opaque dataset refuses to build without a token seed", True)
        a = ed.build_scenarios(20260621, 5, "_naming-test", token_seed=123)
        b = ed.build_scenarios(20260621, 5, "_naming-test", token_seed=123)
        c = ed.build_scenarios(20260621, 5, "_naming-test", token_seed=124)
        toks = [s.token for s in a]
        check("opaque: tokens distinct", len(set(toks)) == len(toks))
        check("opaque: reproducible from the token seed", toks == [s.token for s in b])
        check("opaque: a different token seed gives different tokens", toks != [s.token for s in c])
        check("opaque: not sorted in family order", toks != sorted(toks))
        check("opaque: names carry the token, not the gid",
              all(m.name.endswith("_" + s.token) and "_g9" not in m.name for s in a for m in s.metrics))
        check("opaque: seeds unchanged by naming",
              [s.seed for s in a] == [(20260621 * 1_000_003 + g * 97 + i) & 0x7FFFFFFF
                                      for g, i in ((9000 + k, k % 5) for k in range(45))])
        check("opaque: manifest carries the token", ed.scenario_to_manifest(a[0]).get("token") == a[0].token)
    finally:
        del ed.DATASETS["_naming-test"]

    if failures:
        print(f"FAILED: {failures}")
        return 1
    print("dataset naming checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
