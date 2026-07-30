#!/usr/bin/env python3
"""Render a figure inside judge-service, which is where matplotlib lives.

The host interpreter has no matplotlib and the package index here is
authenticated, so every figure in this repository is drawn inside the
judge-service container -- the same arrangement run_experiment.render_figures()
has always used. This module is the shared plumbing for it.

The split is deliberate and worth keeping even where matplotlib is available on
the host: the *selection* runs host-side through results_schema.load(), and only
a finished, already-aggregated spec crosses into the container. A generator
therefore never parses a raw results file, and a plotting script never decides
what a number means -- it receives values and draws them.

Figures are written to the ai-logs volume (the one directory the container and
host share) and copied out, then measured, because Word extents have to be set
from real pixel dimensions rather than from figsize x dpi arithmetic.
"""

from __future__ import annotations

import json
import shutil
import struct
import subprocess
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parent.parent
CONTAINER_OUT = "/app/data/ai-logs/_exp_figs"
HOST_OUT = REPO_ROOT / "data" / "ai-logs" / "_exp_figs"
FIG_DIR = REPO_ROOT / "results" / "figures"

_PREAMBLE = """
import json, os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
OUT = "{out}"
os.makedirs(OUT, exist_ok=True)
spec = json.loads(sys.stdin.readline())

def save(fig, name, dpi=200):
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("WROTE " + name)
"""


def png_size(path: Path) -> Dict[str, Any]:
    """(width, height) in pixels and the stored dpi, read from the PNG itself."""
    data = path.read_bytes()
    width = height = None
    dpi = None
    pos = 8
    while pos < len(data) - 8:
        length = struct.unpack(">I", data[pos:pos + 4])[0]
        ctype = data[pos + 4:pos + 8]
        body = data[pos + 8:pos + 8 + length]
        if ctype == b"IHDR":
            width, height = struct.unpack(">II", body[:8])
        elif ctype == b"pHYs" and len(body) >= 9:
            ppux, _, unit = struct.unpack(">IIB", body[:9])
            if unit == 1:  # pixels per metre
                dpi = round(ppux * 0.0254)
        pos += 12 + length
        if ctype == b"IEND":
            break
    return {"file": path.name, "width_px": width, "height_px": height, "dpi": dpi,
            "bytes": path.stat().st_size}


def render(plot_code: str, spec: Dict[str, Any], timeout: int = 300) -> List[Dict[str, Any]]:
    """Run `plot_code` in the container with `spec` available, return figure info.

    The script receives its spec on the first line of stdin and calls save(fig,
    name) for each figure. Names are returned with measured dimensions.
    """
    script = _PREAMBLE.format(out=CONTAINER_OUT) + "\n" + plot_code
    payload = json.dumps(spec, separators=(",", ":")) + "\n" + script
    proc = subprocess.run(
        ["docker", "compose", "exec", "-T", "judge-service", "python", "-c",
         "import sys; code=sys.stdin.read().split('\\n', 1); "
         "sys.stdin = __import__('io').StringIO(code[0] + '\\n'); exec(code[1])"],
        input=payload, capture_output=True, text=True, cwd=str(REPO_ROOT), timeout=timeout,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"figure render failed:\n{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}")

    written = [line.split(" ", 1)[1] for line in proc.stdout.splitlines() if line.startswith("WROTE ")]
    if not written:
        raise RuntimeError(f"figure script wrote nothing:\n{proc.stdout[-2000:]}")

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    info = []
    for name in written:
        src = HOST_OUT / name
        dst = FIG_DIR / name
        shutil.copy(src, dst)
        info.append(png_size(dst))
    return info


def report(info: List[Dict[str, Any]]) -> str:
    lines = ["| figure | pixels | dpi | size |", "|---|---|---|---|"]
    for i in info:
        lines.append(f"| `{i['file']}` | {i['width_px']} x {i['height_px']} | {i['dpi']} | "
                     f"{i['bytes']/1024:.0f} kB |")
    return "\n".join(lines)
