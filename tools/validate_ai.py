#!/usr/bin/env python3
"""Per-model AI judge sweep.

Iterates each configured local model alias (from judge-service/models.yaml),
drives the judge-service directly with the saved sample request for a
representative mode (text -> summary, vision -> plot), and asserts a valid
CanaryJudgeResult comes back. Prints a table: alias | modality | mode | verdict |
score | latency | status.

Only models actually present locally are validated. An alias whose Ollama tag has
not been pulled is marked PENDING and skipped (not failed), so this passes with
just the one LLM and one VLM downloaded. A model that returns an invalid verdict
(the judge maps it to an "Error" classification) is a FAIL row, not a crash.

Exit 0 if every present model passed (pending/skipped don't count). Non-zero if
any present model failed, or if no models were present to test.

No external deps beyond `requests`; the YAML/registry files are parsed with
simple line scanning so this runs in the host venv (which has no PyYAML).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

REPO_ROOT = Path(__file__).resolve().parent.parent
REGISTRY = REPO_ROOT / "judge-service" / "models.yaml"
LITELLM_CONFIG = REPO_ROOT / "litellm" / "config.yaml"
SAMPLE = REPO_ROOT / "judge-service" / "sample_request.json"

JUDGE_URL = "http://localhost:5001/judge"
OLLAMA_URL = "http://localhost:11434"

MODE_FOR_MODALITY = {"text": "summary", "vision": "plot"}


def active_aliases() -> List[Tuple[str, str]]:
    """(alias, modality) for each uncommented entry in models.yaml."""
    out: List[Tuple[str, str]] = []
    if not REGISTRY.exists():
        return out
    for line in REGISTRY.read_text().splitlines():
        if line.lstrip().startswith("#"):
            continue
        m = re.match(r"\s+([A-Za-z0-9_-]+):\s*\{\s*modality:\s*(text|vision)", line)
        if m:
            out.append((m.group(1), m.group(2)))
    return out


def alias_to_ollama_tag() -> Dict[str, str]:
    """Map alias -> ollama tag from litellm/config.yaml (uncommented entries)."""
    mapping: Dict[str, str] = {}
    if not LITELLM_CONFIG.exists():
        return mapping
    current: Optional[str] = None
    for line in LITELLM_CONFIG.read_text().splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        m = re.search(r"model_name:\s*([A-Za-z0-9_-]+)", line)
        if m:
            current = m.group(1)
        t = re.search(r"model:\s*ollama(?:_chat)?/(\S+)", line)
        if t and current:
            mapping[current] = t.group(1)
            current = None
    return mapping


def ollama_present_tags() -> List[str]:
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=10)
        r.raise_for_status()
        return [m.get("name", "") for m in r.json().get("models", [])]
    except requests.RequestException:
        return []


def build_request(mode: str, model: str) -> Dict[str, Any]:
    """A RemoteJudgeRequest using the saved sample's pairs + chosen mode/model."""
    sample = json.loads(SAMPLE.read_text())
    canary_config = sample.get("canaryConfig") or {}
    canary_config.setdefault("judge", {})
    canary_config["judge"]["name"] = "RemoteJudge-v1.0"
    canary_config["judge"]["judgeConfigurations"] = {"mode": mode, "model": model}
    return {
        "canaryConfig": canary_config,
        "scoreThresholds": sample.get("scoreThresholds") or {"pass": 75, "marginal": 50},
        "metricSetPairList": sample.get("metricSetPairList") or [],
    }


def run_one(alias: str, modality: str, timeout: int = 300) -> Dict[str, Any]:
    mode = MODE_FOR_MODALITY.get(modality, "summary")
    body = build_request(mode, alias)
    t0 = time.time()
    try:
        r = requests.post(JUDGE_URL, json=body, timeout=timeout)
        latency = time.time() - t0
        r.raise_for_status()
        result = r.json()
    except requests.RequestException as e:
        return {"alias": alias, "modality": modality, "mode": mode, "status": "FAIL",
                "verdict": "-", "score": None, "latency": time.time() - t0, "note": str(e)[:80]}

    score = (result.get("score") or {}).get("score")
    verdict = (result.get("score") or {}).get("classification", "-")
    classes = [r.get("classification") for r in result.get("results", [])]
    is_error = any(c == "Error" for c in classes)
    valid = (
        isinstance(score, (int, float))
        and bool(result.get("results"))
        and verdict in ("Pass", "Marginal", "Fail")
        and not is_error
    )
    return {
        "alias": alias, "modality": modality, "mode": mode,
        "status": "ok" if valid else "FAIL",
        "verdict": verdict, "score": score, "latency": latency,
        "note": "" if valid else ("model returned Error/invalid verdict" if is_error else "invalid result"),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Per-model AI judge validation sweep")
    ap.add_argument("--timeout", type=int, default=300)
    args = ap.parse_args()

    if not SAMPLE.exists():
        print(f"[validate-ai] missing {SAMPLE}; run an AI analysis once to capture it.", file=sys.stderr)
        return 2

    aliases = active_aliases()
    tag_map = alias_to_ollama_tag()
    present = set(ollama_present_tags())

    print("=" * 78)
    print("AI judge per-model sweep (only models present locally are tested)")
    print("=" * 78)

    rows: List[Dict[str, Any]] = []
    for alias, modality in aliases:
        tag = tag_map.get(alias)
        if tag is None:
            # not an Ollama-backed alias (e.g. a cloud reader alias), out of local scope
            rows.append({"alias": alias, "modality": modality, "mode": "-", "status": "PENDING",
                         "verdict": "-", "score": None, "latency": None, "note": "non-local alias"})
            continue
        if tag not in present:
            rows.append({"alias": alias, "modality": modality, "mode": MODE_FOR_MODALITY.get(modality, "summary"),
                         "status": "PENDING", "verdict": "-", "score": None, "latency": None,
                         "note": f"not pulled ({tag})"})
            continue
        print(f"[validate-ai] testing {alias} ({modality}) via Ollama tag {tag} ...")
        rows.append(run_one(alias, modality, timeout=args.timeout))

    # table
    print("\n{:<16} {:<8} {:<8} {:<9} {:>6} {:>9}  {}".format(
        "ALIAS", "MODALITY", "MODE", "VERDICT", "SCORE", "LATENCY", "STATUS"))
    print("-" * 78)
    for r in rows:
        lat = f"{r['latency']:.1f}s" if r["latency"] is not None else "-"
        sc = f"{r['score']:.0f}" if isinstance(r["score"], (int, float)) else "-"
        note = f"  ({r['note']})" if r.get("note") else ""
        print("{:<16} {:<8} {:<8} {:<9} {:>6} {:>9}  {}{}".format(
            r["alias"], r["modality"], r["mode"], r["verdict"], sc, lat, r["status"], note))

    tested = [r for r in rows if r["status"] in ("ok", "FAIL")]
    failed = [r for r in tested if r["status"] == "FAIL"]
    pending = [r for r in rows if r["status"] == "PENDING"]
    print("-" * 78)
    print(f"tested={len(tested)} ok={len(tested)-len(failed)} failed={len(failed)} pending(skipped)={len(pending)}")

    if not tested:
        print("\n[validate-ai] no local models present to test (pull at least one). FAIL.", file=sys.stderr)
        return 1
    if failed:
        print(f"\n[validate-ai] {len(failed)} model(s) failed.", file=sys.stderr)
        return 1
    print("\n[validate-ai] OK: all present models returned a valid verdict.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
