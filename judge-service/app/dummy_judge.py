"""The dummy (no-op) judge: the model-free default and regression fallback.

It does trivial work, a per-metric mean comparison between control and
experiment, just so Kayenta receives a well-formed CanaryJudgeResult and the
analysis completes with no model running. There's no real statistical test and
no model inference.

It stays the default `mode=dummy` so the model-free regression gate
(`make validate`) needs no model. The real judges live elsewhere: the AI judge
in `llm_judge.py` (`mode=summary/raw/plot`) and the hybrid in `hybrid.py`
(`mode=hybrid`). All three return a valid CanaryJudgeResult.
"""

from __future__ import annotations

from statistics import mean, pstdev
from typing import Dict, List, Optional

from .models import (
    CanaryAnalysisResult,
    CanaryJudgeGroupScore,
    CanaryJudgeResult,
    CanaryJudgeScore,
    MetricSetPair,
    RemoteJudgeRequest,
)

JUDGE_NAME = "RemoteJudge-v1.0"  # must match Kayenta's RemoteJudge.JUDGE_NAME
STUB_NOTE = "DUMMY remote judge (no-op default): trivial mean comparison, not a real AI."


def _clean(values: Optional[List[Optional[float]]]) -> List[float]:
    """Drop NaN/None placeholders Kayenta uses for missing samples."""
    if not values:
        return []
    out: List[float] = []
    for v in values:
        if v is None:
            continue
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        if f != f:  # NaN
            continue
        out.append(f)
    return out


def _stats(values: List[float]) -> Dict[str, float]:
    """Mirror the metadata shape Kayenta's standalone aggregation expects:
    {"stats": {count, mean, min, max, std}}. The aggregation reads
    metadata["stats"]["count"] (Stats.getCount), and a missing 'stats' key NPEs,
    so always emit a stats block (count=0 when empty)."""
    if not values:
        return {"stats": {"count": 0, "mean": 0.0, "min": 0.0, "max": 0.0, "std": 0.0}}
    return {
        "stats": {
            "count": len(values),
            "mean": round(mean(values), 6),
            "min": round(min(values), 6),
            "max": round(max(values), 6),
            "std": round(pstdev(values), 6) if len(values) > 1 else 0.0,
        }
    }


def _classify_pair(pair: MetricSetPair) -> CanaryAnalysisResult:
    """Trivial rule: if experiment mean is more than 25% above control mean,
    call it High (regression); if more than 25% below, Low; else Pass."""
    control = _clean(pair.values.get("control"))
    experiment = _clean(pair.values.get("experiment"))

    classification = "Pass"
    reason = STUB_NOTE
    ratio = None
    if control and experiment:
        c_mean = mean(control)
        e_mean = mean(experiment)
        ratio = (e_mean / c_mean) if c_mean else None
        if ratio is not None:
            if ratio > 1.25:
                classification = "High"
                reason = f"{STUB_NOTE} experiment mean {e_mean:.3f} >> control mean {c_mean:.3f}."
            elif ratio < 0.75:
                classification = "Low"
                reason = f"{STUB_NOTE} experiment mean {e_mean:.3f} << control mean {c_mean:.3f}."
            else:
                reason = f"{STUB_NOTE} experiment mean {e_mean:.3f} ~ control mean {c_mean:.3f}."
    elif not control and not experiment:
        classification = "Nodata"
        reason = f"{STUB_NOTE} no data for either control or experiment."

    return CanaryAnalysisResult(
        name=pair.name or pair.id or "unknown",
        id=pair.id or pair.name or "unknown",
        tags=pair.tags or {},
        classification=classification,
        classificationReason=reason,
        groups=["dummy-group"],
        experimentMetadata=_stats(experiment),
        controlMetadata=_stats(control),
        resultMetadata={"ratio": ratio, "stub": True},
    )


def judge(request: RemoteJudgeRequest) -> CanaryJudgeResult:
    """Produce a valid (but no-op) CanaryJudgeResult for the given request."""
    results = [_classify_pair(p) for p in request.metricSetPairList]

    # Overall score: start at 100, subtract 30 per non-Pass metric (floored at 0).
    # Deterministic and obviously a placeholder.
    non_pass = sum(1 for r in results if r.classification not in ("Pass", "Nodata"))
    score_value = float(max(0, 100 - 30 * non_pass))
    overall_class = "Pass" if score_value >= 75 else ("Marginal" if score_value >= 50 else "Fail")

    group_score = CanaryJudgeGroupScore(
        name="dummy-group",
        score=score_value,
        classification=overall_class,
        classificationReason=STUB_NOTE,
    )
    overall = CanaryJudgeScore(
        score=score_value,
        classification=overall_class,
        classificationReason=f"{STUB_NOTE} {non_pass} metric(s) flagged of {len(results)}.",
    )

    return CanaryJudgeResult(
        judgeName=JUDGE_NAME,
        results=results,
        groupScores=[group_score],
        score=overall,
    )
