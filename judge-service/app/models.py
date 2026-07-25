"""Pydantic models mirroring Kayenta's Remote Judge request/response contract.

Verified against kayenta source (master):
  - RemoteJudgeService.java   -> POST /judge, body RemoteJudgeRequest, returns CanaryJudgeResult
  - RemoteJudgeRequest.java   -> { canaryConfig, scoreThresholds, metricSetPairList }
  - CanaryJudgeResult.java     -> { judgeName, results[], groupScores[], score }
  - CanaryAnalysisResult.java  -> { name, tags, id, classification, classificationReason,
                                    groups, experimentMetadata, controlMetadata,
                                    resultMetadata, critical, muted }
  - CanaryJudgeScore.java      -> { score, classification, classificationReason }
  - CanaryJudgeGroupScore.java -> { name, score, classification, classificationReason }

The request models are permissive (extra fields allowed, almost everything
Optional) because Kayenta sends a large, version-specific payload and we only need
a few fields. The response models are strict enough to be valid for Kayenta's
Jackson deserializer, which ignores unknown properties.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


# Request side (what Kayenta POSTs to /judge). Permissive by design.
class _Lenient(BaseModel):
    model_config = ConfigDict(extra="allow")


class MetricSetPair(_Lenient):
    """A single paired control/experiment series. Only the fields we read are
    typed; everything else is preserved via extra='allow'."""

    name: Optional[str] = None
    id: Optional[str] = None
    tags: Dict[str, str] = Field(default_factory=dict)
    # values: { "control": [...], "experiment": [...] }
    values: Dict[str, List[Optional[float]]] = Field(default_factory=dict)


class RemoteJudgeRequest(_Lenient):
    canaryConfig: Optional[Dict[str, Any]] = None
    scoreThresholds: Optional[Dict[str, Any]] = None
    metricSetPairList: List[MetricSetPair] = Field(default_factory=list)


# Response side (what we return; must be a valid CanaryJudgeResult).
class CanaryJudgeScore(BaseModel):
    score: float
    classification: str
    classificationReason: Optional[str] = None


class CanaryJudgeGroupScore(BaseModel):
    name: str
    score: float
    classification: str
    classificationReason: Optional[str] = None


class CanaryAnalysisResult(BaseModel):
    name: str
    id: str
    tags: Dict[str, str] = Field(default_factory=dict)
    classification: str
    classificationReason: Optional[str] = ""
    groups: List[str] = Field(default_factory=list)
    experimentMetadata: Dict[str, Any] = Field(default_factory=dict)
    controlMetadata: Dict[str, Any] = Field(default_factory=dict)
    resultMetadata: Dict[str, Any] = Field(default_factory=dict)
    critical: bool = False
    muted: bool = False


class CanaryJudgeResult(BaseModel):
    judgeName: str
    results: List[CanaryAnalysisResult] = Field(default_factory=list)
    groupScores: List[CanaryJudgeGroupScore] = Field(default_factory=list)
    score: CanaryJudgeScore
