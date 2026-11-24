"""Render matplotlib drawing code inside the running judge-service container, where matplotlib lives.

The arrangement is that of tools/container_plot.py (the repository's figure plumbing): the spec is prepared on the
host and only a finished spec crosses into the container, through `docker compose exec -T judge-service python -c`.
Two things differ, both deliberate:

* only the running service of the Compose project in the environment is used (COMPOSE_PROJECT_NAME and
  COMPOSE_FILE, as for any docker compose command run from the repository root). container_plot.render() falls back
  to `docker run` of an image when the exec fails; here a failed exec stops with an error, so no new container is
  ever created and no other image is used. Before the exec the target container's name is checked against
  --container;
* the PNG comes back on standard output (base64), so nothing is written into the data/ai-logs volume that the
  service shares with the host.

PNG size and dpi are measured with tools/container_plot.png_size.
"""

from __future__ import annotations

import base64
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "tools"))

import container_plot  # noqa: E402  (png_size only)

RUNNER = ("import sys, io; code = sys.stdin.read().split('\\n', 1); "
          "sys.stdin = io.StringIO(code[0] + '\\n'); exec(code[1])")

PREAMBLE = """
import base64, io, json, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
spec = json.loads(sys.stdin.readline())

def emit(fig, name, dpi):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, facecolor="white")
    plt.close(fig)
    print("PNG " + name + " " + base64.b64encode(buf.getvalue()).decode("ascii"))
"""


def target_container() -> str:
    """Name of the container `docker compose exec judge-service` would reach (inspected with a format that
    prints the name only)."""
    for k in ("COMPOSE_PROJECT_NAME", "COMPOSE_FILE"):
        if not os.environ.get(k):
            raise SystemExit(f"{k} is not set: run with the Compose project of the stack in the environment")
    ids = subprocess.run(["docker", "compose", "ps", "-q", "judge-service"], cwd=str(REPO), capture_output=True,
                         text=True, timeout=60)
    cid = ids.stdout.strip().splitlines()
    if ids.returncode != 0 or len(cid) != 1:
        raise SystemExit(f"judge-service is not running in this Compose project: {ids.stderr[-500:]}")
    name = subprocess.run(["docker", "inspect", "--format", "{{.Name}}", cid[0]], capture_output=True, text=True,
                          timeout=60)
    return name.stdout.strip().lstrip("/")


def render(plot_code: str, spec: Dict[str, Any], out_dir: Path, expect_container: str, timeout: int = 600):
    """Run plot_code (which calls emit(fig, name, dpi)) in the service; write the PNGs to out_dir (never over an
    existing file) and return their measured sizes."""
    name = target_container()
    if name != expect_container:
        raise SystemExit(f"judge-service resolves to container {name!r}, expected {expect_container!r}; stopping")
    payload = json.dumps(spec, separators=(",", ":")) + "\n" + PREAMBLE + "\n" + plot_code
    proc = subprocess.run(["docker", "compose", "exec", "-T", "judge-service", "python", "-c", RUNNER],
                          input=payload, capture_output=True, text=True, cwd=str(REPO), timeout=timeout)
    if proc.returncode != 0:
        raise SystemExit(f"render failed in {name}:\n{proc.stderr[-3000:]}")
    out_dir.mkdir(parents=True, exist_ok=True)
    info = []
    for line in proc.stdout.splitlines():
        if not line.startswith("PNG "):
            continue
        _, fname, data = line.split(" ", 2)
        dst = out_dir / fname
        if dst.exists():
            raise SystemExit(f"refusing to overwrite {dst}")
        dst.write_bytes(base64.b64decode(data))
        info.append({**container_plot.png_size(dst), "container": name})
    if not info:
        raise SystemExit(f"the plot code emitted no figure:\n{proc.stdout[-2000:]}")
    return info
