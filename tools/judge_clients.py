#!/usr/bin/env python3
"""Host-side judge clients used by the experiment runner and the dataset probe.

Two ways to judge a metricSetPairList without a full standalone analysis, so the
real NetflixACAJudge and the AI judge score the same in-memory pairs:

  run_default_judge(pairs, config)  -> the real NetflixACAJudge verdict, via
      Kayenta's documented /metricSetPairList + /canaryConfig + /judges/judge
      callback (the same mechanism the in-service hybrid uses, just from the
      host). No metric re-fetch; it judges the stored pairs in-process.

  judge_ai(pairs, mode, model)      -> the configurable AI judge, by POSTing a
      RemoteJudgeRequest straight to judge-service /judge (like validate_ai.py).

Both return a normalised dict: {verdict (PASS/FAIL), classification (Pass/
Marginal/Fail), score, per_metric: [{name, classification, reason}], rationale,
judge_name, model, ok, error}.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import requests

KAYENTA_URL = "http://localhost:8090"
JUDGE_URL = "http://localhost:5001/judge"
HTTP_TIMEOUT = 60


def _classify(score: float, pass_t: float, marginal_t: float) -> str:
    if score >= pass_t:
        return "Pass"
    if score >= marginal_t:
        return "Marginal"
    return "Fail"


def _normalise(result: Dict[str, Any], pass_t: float, marginal_t: float,
               latency: float, model: str = "") -> Dict[str, Any]:
    score_obj = result.get("score") or {}
    score = float(score_obj.get("score", 0.0))
    classification = str(score_obj.get("classification") or _classify(score, pass_t, marginal_t))
    per_metric = []
    for r in result.get("results") or []:
        per_metric.append({
            "name": str(r.get("name", "?")),
            "classification": str(r.get("classification", "?")),
            "reason": str(r.get("classificationReason", "") or ""),
        })
        if not model:
            model = str((r.get("resultMetadata") or {}).get("model", "") or "")
    return {
        "verdict": "PASS" if score >= pass_t else "FAIL",
        "classification": classification,
        "score": score,
        "per_metric": per_metric,
        "rationale": str(score_obj.get("classificationReason", "") or ""),
        "judge_name": str(result.get("judgeName", "") or ""),
        "model": model,
        "latency": latency,
        "ok": True,
        "error": "",
    }


def _err(msg: str, latency: float = 0.0) -> Dict[str, Any]:
    return {"verdict": "FAIL", "classification": "Fail", "score": 0.0, "per_metric": [],
            "rationale": "", "judge_name": "", "model": "", "latency": latency,
            "ok": False, "error": msg[:300]}


# Real NetflixACAJudge via the Kayenta callback (from the host).
def run_default_judge(pairs: List[Dict[str, Any]], config: Dict[str, Any],
                      pass_t: float = 75.0, marginal_t: float = 50.0,
                      base_url: str = KAYENTA_URL) -> Dict[str, Any]:
    base = base_url.rstrip("/")
    t0 = time.time()
    mspl_id: Optional[str] = None
    cfg_id: Optional[str] = None
    try:
        r = requests.post(f"{base}/metricSetPairList", json=pairs, timeout=HTTP_TIMEOUT)
        if r.status_code >= 400:
            return _err(f"store metricSetPairList {r.status_code}: {r.text[:200]}", time.time() - t0)
        mspl_id = r.json()["metricSetPairListId"]

        cfg = dict(config)
        cfg.pop("id", None)
        cfg["name"] = f"eval-default-{mspl_id}"
        cfg.setdefault("judge", {})
        cfg["judge"] = {"name": "NetflixACAJudge-v1.0", "judgeConfigurations": {}}
        r = requests.post(f"{base}/canaryConfig", json=cfg, timeout=HTTP_TIMEOUT)
        if r.status_code >= 400:
            return _err(f"store canaryConfig {r.status_code}: {r.text[:200]}", time.time() - t0)
        cfg_id = r.json()["canaryConfigId"]

        r = requests.post(f"{base}/judges/judge", params={
            "canaryConfigId": cfg_id, "metricSetPairListId": mspl_id,
            "passThreshold": pass_t, "marginalThreshold": marginal_t,
        }, timeout=HTTP_TIMEOUT)
        if r.status_code >= 400:
            return _err(f"/judges/judge {r.status_code}: {r.text[:200]}", time.time() - t0)
        return _normalise(r.json(), pass_t, marginal_t, time.time() - t0)
    except requests.RequestException as e:
        return _err(str(e), time.time() - t0)
    finally:
        for path, oid in (("metricSetPairList", mspl_id), ("canaryConfig", cfg_id)):
            if oid:
                try:
                    requests.delete(f"{base}/{path}/{oid}", timeout=15)
                except requests.RequestException:
                    pass


# Configurable AI judge via judge-service /judge.
def judge_ai(pairs: List[Dict[str, Any]], config: Dict[str, Any], mode: str, model: str,
             pass_t: float = 75.0, marginal_t: float = 50.0,
             judge_url: str = JUDGE_URL, timeout: int = 300) -> Dict[str, Any]:
    cfg = dict(config)
    cfg["judge"] = {"name": "RemoteJudge-v1.0", "judgeConfigurations": {"mode": mode, "model": model}}
    body = {
        "canaryConfig": cfg,
        "scoreThresholds": {"pass": pass_t, "marginal": marginal_t},
        "metricSetPairList": pairs,
    }
    t0 = time.time()
    try:
        r = requests.post(judge_url, json=body, timeout=timeout)
        latency = time.time() - t0
        if r.status_code >= 400:
            return _err(f"/judge {r.status_code}: {r.text[:200]}", latency)
        return _normalise(r.json(), pass_t, marginal_t, latency, model=model)
    except requests.RequestException as e:
        return _err(str(e), time.time() - t0)
