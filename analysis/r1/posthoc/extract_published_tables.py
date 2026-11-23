#!/usr/bin/env python3
"""Read the published tables the post-hoc analyses start from, out of the submitted .docx files.

Writes one JSON file with the cell text of Tables 1, 6 and 8 of the manuscript and
of Tables S13 and S38 of the Supplementary Materials, and the published sentences
of Sections 4.2, 4.6 and 5.1 that D2 evaluates, with the SHA-256 of both inputs.
The documents are opened read only with a Word-document reader; nothing is saved to them.

Run:
  python -I -B extract_published_tables.py <manuscript.docx> <supplementary.docx> <new-output.json>
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True

from docx import Document  # noqa: E402
from docx.table import Table  # noqa: E402
from docx.text.paragraph import Paragraph  # noqa: E402


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def blocks(path: Path):
    doc = Document(str(path))
    out = []
    for child in doc.element.body.iterchildren():
        tag = child.tag.split("}")[1]
        if tag == "p":
            out.append(("p", Paragraph(child, doc).text))
        elif tag == "tbl":
            t = Table(child, doc)
            out.append(("t", [[c.text.replace("\n", " / ") for c in row.cells] for row in t.rows]))
    return out


def table_after(bl, caption_start: str):
    """The first table after the paragraph that starts with `caption_start`."""
    for i, (kind, val) in enumerate(bl):
        if kind == "p" and val.startswith(caption_start):
            for kind2, val2 in bl[i + 1:]:
                if kind2 == "t":
                    return {"caption": val, "rows": val2}
    raise SystemExit(f"caption not found: {caption_start!r}")


def section_paragraphs(bl, heading: str, next_heading: str):
    out, on = [], False
    for kind, val in bl:
        if kind != "p":
            continue
        if val.strip() == heading:
            on = True
            continue
        if on and val.strip() == next_heading:
            break
        if on and val.strip():
            out.append(val)
    if not out:
        raise SystemExit(f"section not found: {heading!r}")
    return out


def main() -> int:
    if len(sys.argv) != 4:
        print(__doc__)
        return 2
    ms, sm, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out.name}")
    bm, bs = blocks(ms), blocks(sm)
    res = {
        "inputs": {"manuscript": {"file": ms.name, "sha256": sha256(ms)},
                   "supplementary": {"file": sm.name, "sha256": sha256(sm)}},
        "tables": {
            "Table 1": table_after(bm, "Table 1. "),
            "Table 6": table_after(bm, "Table 6. "),
            "Table 8": table_after(bm, "Table 8. "),
            "Table S13": table_after(bs, "Table S13. "),
            "Table S38": table_after(bs, "Table S38. "),
        },
        "sections": {
            "4.2": section_paragraphs(bm, "4.2. Detection and False Alarms of the Model Judges", "4.3. What the Detection Is Attributable To"),
            "4.6": section_paragraphs(bm, "4.6. Model-Free Comparators", "5. Discussion"),
            "5.1": section_paragraphs(bm, "5.1. Can Model Judges Enhance Automated Canary Analysis?",
                                      "5.2. Confounds and What the Benchmark Can Support"),
        },
    }
    out.write_text(json.dumps(res, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {out.name}: {len(res['tables'])} tables, sections " + ", ".join(
        f"{k} ({len(v)} paragraphs)" for k, v in res["sections"].items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
