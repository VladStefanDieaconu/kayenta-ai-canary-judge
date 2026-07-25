"""Kayenta callback client used by the hybrid mode of the remote judge.

The hybrid judge doesn't reimplement the statistical judge. Instead it asks
Kayenta to run the genuine NetflixACAJudge on the same already-fetched metric
set pairs, via three documented Kayenta REST calls:

  1. POST /metricSetPairList            (body: the List<MetricSetPair> we received)
       -> { "metricSetPairListId": "<id>" }
  2. POST /canaryConfig                  (body: the incoming config, but with
       judge.name = "NetflixACAJudge-v1.0")            -> { "canaryConfigId": "<id>" }
  3. POST /judges/judge?canaryConfigId&metricSetPairListId&passThreshold&marginalThreshold
       -> CanaryJudgeResult              (the real statistical verdict)

Then the temp pair-list and config are deleted (best-effort). No metric re-fetch
happens; /judges/judge judges the stored pairs in-process. The callback names
NetflixACAJudge (an in-process judge), so it never recurses back into this
remote service.

The Kayenta base URL is taken from KAYENTA_BASE_URL (so anyone can point this
service at their own Kayenta). Account names are left unset, so Kayenta uses its
default object store.
"""

from __future__ import annotations

import copy
import logging
import os
from typing import Any, Dict, List, Optional

import requests

log = logging.getLogger("judge-service.kayenta")

KAYENTA_BASE_URL = os.environ.get("KAYENTA_BASE_URL", "http://kayenta:8090")
DEFAULT_JUDGE_NAME = "NetflixACAJudge-v1.0"
HTTP_TIMEOUT = int(os.environ.get("KAYENTA_CALLBACK_TIMEOUT", "60"))


class KayentaCallbackError(RuntimeError):
    pass


def _base() -> str:
    return KAYENTA_BASE_URL.rstrip("/")


def store_metric_set_pair_list(metric_set_pair_list: List[Dict[str, Any]]) -> str:
    r = requests.post(f"{_base()}/metricSetPairList", json=metric_set_pair_list, timeout=HTTP_TIMEOUT)
    if r.status_code >= 400:
        raise KayentaCallbackError(f"store metricSetPairList failed {r.status_code}: {r.text[:300]}")
    return r.json()["metricSetPairListId"]


def store_default_judge_config(canary_config: Dict[str, Any], unique_suffix: str) -> str:
    """Clone the incoming config, force judge.name to the default judge, store it.

    The cloned config keeps the same metrics (direction/groups) and classifier
    (groupWeights/scoreThresholds); only the judge changes. Setting the judge to
    NetflixACAJudge is what prevents recursion back into this remote service.
    """
    cfg = copy.deepcopy(canary_config) if canary_config else {}
    cfg.pop("id", None)  # let Kayenta generate a fresh id
    cfg["name"] = f"hybrid-default-{unique_suffix}"
    cfg.setdefault("judge", {})
    cfg["judge"]["name"] = DEFAULT_JUDGE_NAME
    cfg["judge"]["judgeConfigurations"] = {}
    r = requests.post(f"{_base()}/canaryConfig", json=cfg, timeout=HTTP_TIMEOUT)
    if r.status_code >= 400:
        raise KayentaCallbackError(f"store canaryConfig failed {r.status_code}: {r.text[:300]}")
    return r.json()["canaryConfigId"]


def run_default_judge(
    metric_set_pair_list: List[Dict[str, Any]],
    canary_config: Dict[str, Any],
    pass_threshold: float,
    marginal_threshold: float,
) -> Dict[str, Any]:
    """Run NetflixACAJudge on the given pairs via Kayenta; return its CanaryJudgeResult."""
    mspl_id = store_metric_set_pair_list(metric_set_pair_list)
    config_id: Optional[str] = None
    try:
        config_id = store_default_judge_config(canary_config, mspl_id)
        params = {
            "canaryConfigId": config_id,
            "metricSetPairListId": mspl_id,
            "passThreshold": pass_threshold,
            "marginalThreshold": marginal_threshold,
        }
        log.info("hybrid: calling /judges/judge (NetflixACAJudge) cfg=%s pairs=%s", config_id, mspl_id)
        r = requests.post(f"{_base()}/judges/judge", params=params, timeout=HTTP_TIMEOUT)
        if r.status_code >= 400:
            raise KayentaCallbackError(f"/judges/judge failed {r.status_code}: {r.text[:300]}")
        return r.json()
    finally:
        _cleanup(mspl_id, config_id)


def _cleanup(mspl_id: Optional[str], config_id: Optional[str]) -> None:
    """Best-effort removal of the temp objects we created for this judgment."""
    if mspl_id:
        try:
            requests.delete(f"{_base()}/metricSetPairList/{mspl_id}", timeout=15)
        except requests.RequestException as e:
            log.warning("cleanup metricSetPairList %s failed: %s", mspl_id, e)
    if config_id:
        try:
            requests.delete(f"{_base()}/canaryConfig/{config_id}", timeout=15)
        except requests.RequestException as e:
            log.warning("cleanup canaryConfig %s failed: %s", config_id, e)
