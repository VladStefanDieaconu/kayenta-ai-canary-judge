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

    # judgeMetadata is how the AI path signals that no judgement was made. A 200
    # carrying a structurally valid result is not evidence that a model answered:
    # an empty completion also returns 200, with score 0.0 and classification
    # Fail, which reads exactly like a verdict of FAIL. Absent metadata (the
    # statistical and dummy paths, and any archived response) means ok.
    meta = result.get("judgeMetadata") or {}
    ok = bool(meta.get("ok", True))
    error = str(meta.get("error", "") or "")
    usage = meta.get("usage") or {}

    return {
        "verdict": "PASS" if score >= pass_t else "FAIL",
        "classification": classification,
        "score": score,
        "per_metric": per_metric,
        "rationale": str(score_obj.get("classificationReason", "") or ""),
        "judge_name": str(result.get("judgeName", "") or ""),
        "model": model,
        "latency": latency,
        "ok": ok,
        "error": error[:300],
        "error_kind": str(meta.get("error_kind", "") or ""),
        "prompt_id": str(meta.get("prompt_id", "") or ""),
        "prompt_hash": str(meta.get("prompt_hash", "") or ""),
        "finish_reason": str(meta.get("finish_reason") or ""),
        "tokens_in": usage.get("prompt_tokens"),
        "tokens_out": usage.get("completion_tokens"),
    }


def _err(msg: str, latency: float = 0.0, error_kind: str = "transport") -> Dict[str, Any]:
    return {"verdict": "FAIL", "classification": "Fail", "score": 0.0, "per_metric": [],
            "rationale": "", "judge_name": "", "model": "", "latency": latency,
            "ok": False, "error": msg[:300], "error_kind": error_kind,
            "prompt_id": "", "prompt_hash": "", "finish_reason": "",
            "tokens_in": None, "tokens_out": None}


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
             judge_url: str = JUDGE_URL, timeout: int = 300,
             prompt_id: str = "") -> Dict[str, Any]:
    cfg = dict(config)
    judge_cfg: Dict[str, Any] = {"mode": mode, "model": model}
    # Omitted rather than blank: an empty prompt_id in the config would have to be
    # distinguished from an absent one by the service, and the service's default
    # is the frozen rubric, which is what an unspecified caller wants.
    if prompt_id:
        judge_cfg["prompt_id"] = prompt_id
    cfg["judge"] = {"name": "RemoteJudge-v1.0", "judgeConfigurations": judge_cfg}
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
