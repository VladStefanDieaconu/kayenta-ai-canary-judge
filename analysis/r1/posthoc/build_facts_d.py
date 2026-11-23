#!/usr/bin/env python3
"""Merge the fact files of D1, D2 and D3 into one list of facts with ids D-001 onwards.

Reads d1_facts.json, d2_facts.json and d3_facts.json from the output directory and
writes facts-D.json (a list of {id, key, value, description, source_file,
filter_or_computation, script}) and FACTS-D.md (the same as a Markdown table).
Ids follow the order D1, D2, D3 and, within each, the order in which the script
recorded its facts. Existing files are never overwritten.

Run: python -I -B build_facts_d.py <output-dir>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    out = Path(sys.argv[1])
    items = []
    for name in ("d1_facts.json", "d2_facts.json", "d3_facts.json"):
        items += json.loads((out / name).read_text(encoding="utf-8"))
    facts_all = []
    for i, f in enumerate(items, 1):
        facts_all.append({"id": f"D-{i:03d}", **f})
    jf, mf = out / "facts-D.json", out / "FACTS-D.md"
    for f in (jf, mf):
        if f.exists():
            raise SystemExit(f"refusing to overwrite {f.name}")
    jf.write_text(json.dumps(facts_all, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    def cell(x: str) -> str:
        return str(x).replace("|", "\\|").replace("\n", " ")

    lines = ["# Facts of the post-hoc analyses (D1 to D3, first revision)", "",
             "Every number below was computed by the script named in its row from the file named in its row. "
             "Paths are relative to the root of Archive S2, to the published data deposit (data-archive/...), "
             "to Archive S1 (archive-s1/...) or to the repository root.", "",
             "| id | value | description | source file | filter or computation | script |",
             "|---|---|---|---|---|---|"]
    for f in facts_all:
        lines.append("| " + " | ".join(cell(f[k]) for k in ("id", "value", "description", "source_file",
                                                            "filter_or_computation", "script")) + " |")
    mf.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {jf.name} and {mf.name}: {len(facts_all)} facts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
