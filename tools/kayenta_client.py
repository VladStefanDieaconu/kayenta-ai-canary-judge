#!/usr/bin/env python3
"""Thin Kayenta REST client for the standalone canary analysis (SCAPE) API.

Endpoints (verified against kayenta master, see README "Source-verified API"):
  POST /standalone_canary_analysis/        body: CanaryAnalysisAdhocExecutionRequest
        query: metricsAccountName, storageAccountName, application, user
        -> { "canaryAnalysisExecutionId": "<id>" }
  GET  /standalone_canary_analysis/{id}     query: storageAccountName
        -> CanaryAnalysisExecutionStatusResponse { complete, executionStatus,
             canaryAnalysisExecutionResult { didPassThresholds, canaryScores,
             canaryScoreMessage, canaryExecutionResults[...] }, exception, ... }

The ad-hoc form embeds the canary config in the body, so nothing has to be
pre-stored in the object store:
  { "canaryConfig": {...}, "executionRequest": {...CanaryAnalysisExecutionRequest...} }

CanaryAnalysisExecutionRequest fields (verified):
  scopes: [ CanaryAnalysisExecutionRequestScope ]   # flat, see build_scope()
  thresholds: { pass, marginal }
  lifetimeDurationMins, analysisIntervalMins, beginAfterMins, lookbackMins, siteLocal
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import requests


@dataclass
class JudgeOutcome:
    """Normalized view of one analysis, for printing + hybrid merge."""

    judge_label: str
    execution_id: str
    completed: bool
    passed: Optional[bool]            # didPassThresholds
    score: Optional[float]            # final/last canary score
    score_message: str
    per_metric: List[Dict[str, str]] = field(default_factory=list)  # name/classification
    raw_status: Dict[str, Any] = field(default_factory=dict)
    # The judge's own reported name. For AI modes this encodes the model, e.g.
    # "RemoteJudge-v1.0 (ai:summary:qwen-llm)"; "" if not found.
    judge_name: str = ""
    model: str = ""                   # resolved model alias, from result metadata

    @property
    def verdict(self) -> str:
        if not self.completed:
            return "INCOMPLETE"
        if self.passed is True:
            return "PASS"
        if self.passed is False:
            return "FAIL"
        return "UNKNOWN"

    @property
    def has_result(self) -> bool:
        return self.completed and self.score is not None


def iso_utc(dt: datetime) -> str:
    """ISO-8601 with trailing Z (Kayenta parses Instant.parse(...))."""
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def build_scope(
    start: datetime,
    end: datetime,
    step: int = 60,
    control_scope: str = "Control",
    experiment_scope: str = "Experiment",
    location: str = "vm",
    scope_name: str = "default",
) -> Dict[str, Any]:
    """A single CanaryAnalysisExecutionRequestScope (flat structure).

    Note: the real shape is flat (controlScope/experimentScope are strings with
    startTimeIso/endTimeIso siblings), not the nested controlScope:{scope,...}
    shape you might expect from the upstream docs. See README's
    source-verified config notes.
    """
    return {
        "scopeName": scope_name,
        "controlScope": control_scope,
        "controlLocation": location,
        "experimentScope": experiment_scope,
        "experimentLocation": location,
        "startTimeIso": iso_utc(start),
        "endTimeIso": iso_utc(end),
        "step": step,
        "extendedScopeParams": {},
    }


class KayentaClient:
    def __init__(self, base_url: str = "http://localhost:8090", timeout: int = 30):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    # health
    def health(self) -> Dict[str, Any]:
        r = requests.get(f"{self.base_url}/health", timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def wait_for_health(self, timeout_s: int = 240, interval_s: int = 5) -> bool:
        deadline = time.time() + timeout_s
        last = ""
        while time.time() < deadline:
            try:
                h = self.health()
                status = h.get("status")
                if status == "UP":
                    return True
                last = str(status)
            except requests.RequestException as e:
                last = str(e)
            time.sleep(interval_s)
        print(f"[kayenta] health never reached UP (last={last})")
        return False

    # submit / poll
    def submit_analysis(
        self,
        canary_config: Dict[str, Any],
        scopes: List[Dict[str, Any]],
        thresholds: Dict[str, float],
        metrics_account: str = "vm",
        storage_account: str = "minio-store",
        lifetime_mins: int = 25,
        analysis_interval_mins: int = 25,
        application: str = "ad-hoc",
        user: str = "canaryllm",
    ) -> str:
        execution_request = {
            "scopes": scopes,
            "thresholds": thresholds,
            "lifetimeDurationMins": lifetime_mins,
            "analysisIntervalMins": analysis_interval_mins,
            "beginAfterMins": 0,
            "lookbackMins": 0,
        }
        body = {"canaryConfig": canary_config, "executionRequest": execution_request}
        params = {
            "metricsAccountName": metrics_account,
            "storageAccountName": storage_account,
            "application": application,
            "user": user,
        }
        r = requests.post(
            f"{self.base_url}/standalone_canary_analysis/",
            params=params,
            json=body,
            timeout=self.timeout,
        )
        if r.status_code >= 400:
            raise RuntimeError(
                f"submit failed {r.status_code}: {r.text[:1000]}"
            )
        data = r.json()
        exec_id = data.get("canaryAnalysisExecutionId")
        if not exec_id:
            raise RuntimeError(f"no canaryAnalysisExecutionId in response: {data}")
        return exec_id

    def get_status(self, execution_id: str, storage_account: str = "minio-store") -> Dict[str, Any]:
        r = requests.get(
            f"{self.base_url}/standalone_canary_analysis/{execution_id}",
            params={"storageAccountName": storage_account},
            timeout=self.timeout,
        )
        r.raise_for_status()
        return r.json()

    def poll(
        self,
        execution_id: str,
        storage_account: str = "minio-store",
        timeout_s: int = 180,
        interval_s: int = 3,
    ) -> Dict[str, Any]:
        deadline = time.time() + timeout_s
        status: Dict[str, Any] = {}
        while time.time() < deadline:
            status = self.get_status(execution_id, storage_account)
            if status.get("complete"):
                return status
            time.sleep(interval_s)
        return status  # may be incomplete; caller decides

    # run + normalize
    def run_analysis(
        self,
        judge_label: str,
        canary_config: Dict[str, Any],
        scopes: List[Dict[str, Any]],
        thresholds: Dict[str, float],
        **kwargs,
    ) -> JudgeOutcome:
        exec_id = self.submit_analysis(canary_config, scopes, thresholds, **kwargs)
        status = self.poll(exec_id)
        return self._normalize(judge_label, exec_id, status)

    @staticmethod
    def _find_first(obj: Any, key: str) -> Optional[Any]:
        """Depth-first search for the first value under `key` anywhere in a nested
        dict/list. Used to dig out judgeResult.results across version-specific
        result nesting without hardcoding the full path."""
        if isinstance(obj, dict):
            if key in obj:
                return obj[key]
            for v in obj.values():
                found = KayentaClient._find_first(v, key)
                if found is not None:
                    return found
        elif isinstance(obj, list):
            for item in obj:
                found = KayentaClient._find_first(item, key)
                if found is not None:
                    return found
        return None

    def _normalize(self, judge_label: str, exec_id: str, status: Dict[str, Any]) -> JudgeOutcome:
        completed = bool(status.get("complete"))
        result = status.get("canaryAnalysisExecutionResult") or {}
        passed = result.get("didPassThresholds")
        scores = result.get("canaryScores") or []
        score = float(scores[-1]) if scores else None
        message = result.get("canaryScoreMessage") or ""

        # Surface any exception to make wiring failures obvious.
        exc = status.get("exception")
        if exc and not message:
            message = f"exception: {str(exc)[:400]}"

        per_metric: List[Dict[str, str]] = []
        judge_name = ""
        model = ""
        judge_result = self._find_first(status, "judgeResult")
        if isinstance(judge_result, dict):
            judge_name = str(judge_result.get("judgeName", "") or "")
            for r in judge_result.get("results", []) or []:
                if isinstance(r, dict):
                    per_metric.append(
                        {
                            "name": str(r.get("name", "?")),
                            "classification": str(r.get("classification", "?")),
                        }
                    )
                    # AI modes stamp the model into each result's resultMetadata.
                    if not model:
                        model = str((r.get("resultMetadata") or {}).get("model", "") or "")

        return JudgeOutcome(
            judge_label=judge_label,
            execution_id=exec_id,
            completed=completed,
            passed=passed if isinstance(passed, bool) else None,
            score=score,
            score_message=message,
            per_metric=per_metric,
            raw_status=status,
            judge_name=judge_name,
            model=model,
        )


def default_window(window_mins: int = 25) -> tuple[datetime, datetime]:
    """Analysis window: strictly inside the seeded [now-30m, now] range.

    end = now (minus a small guard), start = end - window_mins.
    """
    now = datetime.now(timezone.utc) - timedelta(seconds=30)
    start = now - timedelta(minutes=window_mins)
    return start, now
