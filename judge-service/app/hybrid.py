"""Hybrid mode for the remote judge.

Combines two real verdicts on the same already-fetched data and returns one
CanaryJudgeResult, so the hybrid is itself a normal Kayenta analysis with its own
execution id, viewable and comparable in Referee:

  - result A: the genuine NetflixACAJudge, obtained by calling back into Kayenta
    (kayenta_callback.run_default_judge). Never reimplemented here.
  - result B: this service's AI verdict. Defaults to the no-op dummy so the hybrid
    works with no model and `make validate` stays model-free, but if the canary
    config sets judgeConfigurations.aiMode = summary|raw|plot, B is the real
    configurable AI judge (LLM/VLM via LiteLLM) for that representation and model.

So the hybrid is "real statistical + (dummy or real AI)", chosen by config:
  judgeConfigurations: { "mode": "hybrid", "aiMode": "summary", "model": "qwen-llm" }
  judgeConfigurations: { "mode": "hybrid" }                       # dummy AI half (default)

`decide_policy()` is the merge itself, selectable by
`judgeConfigurations.hybridPolicy = gated | or | and` (default `gated`). It's a
pure function of the two judges' (verdict, score, per-metric), so the experiment
runner (tools/run_experiment.py) reuses the same code to score all three policies
offline. The policy lives in hybrid_policy.py.

Which AI verdict the policy receives (v1.0.1). The offline harness that scored
every published hybrid number passes `classify(AI score)` and never reads the
model's own `overallVerdict` label. v1.0.0 of this service passed the label, so
the served hybrid was a different decision function from the scored one. The
default is now the score, as in the harness; `judgeConfigurations.hybridAiVerdict
= "label"` restores the v1.0.0 behaviour.

A failed AI half is not a verdict (v1.0.1). v1.0.0 passed the placeholder of a
failed call (Fail at score 0) into the policy, where it overrides a clean
statistical PASS. By default the hybrid now returns the statistical verdict alone,
with `judgeMetadata.ok = false` and `error_kind = "ai_half_failed"`;
`judgeConfigurations.onAiError = "fail"` returns an explicit Fail at score 0
instead (fail closed). The failure is named in the reason either way.

`prompt_id` and `seed` in judgeConfigurations are forwarded to the AI half
(v1.0.1); v1.0.0 dropped them, so a hybrid ran under the default rubric and seed.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from . import kayenta_callback
from .dummy_judge import judge as dummy_ai_judge
from .hybrid_policy import (
    DEFAULT_POLICY,
    PASS_OK,
    PolicyDecision,
    classify as _classify,
    decide_policy,
    text_flags_blindspot,
)
from .judge_config import AI_MODES, resolve
from .llm_judge import _group_scores, _groups_from_config, judge_ai
from .models import (
    CanaryAnalysisResult,
    CanaryJudgeResult,
    CanaryJudgeScore,
    RemoteJudgeRequest,
)

log = logging.getLogger("judge-service.hybrid")

# Where the policy's AI verdict comes from: "score" is classify(AI score), what the
# offline harness scored; "label" is the model's own overallVerdict (v1.0.0).
AI_VERDICT_SOURCES = ("score", "label")
DEFAULT_AI_VERDICT_SOURCE = "score"
# What a failed AI half does: "statistical" returns the statistical verdict alone,
# "fail" returns an explicit Fail at score 0. Both mark judgeMetadata.ok false.
ON_AI_ERROR = ("statistical", "fail")
DEFAULT_ON_AI_ERROR = "statistical"


# Service entry point (called by main.py for mode=hybrid). The policy itself
# lives in hybrid_policy.decide_policy (shared with the experiment runner).
def judge_hybrid(
    parsed: RemoteJudgeRequest,
    raw_metric_set_pair_list: List[Dict[str, Any]],
    raw_canary_config: Dict[str, Any],
    score_thresholds: Dict[str, Any],
) -> CanaryJudgeResult:
    pass_t = float(score_thresholds.get("pass", 75))
    marginal_t = float(score_thresholds.get("marginal", 50))

    jc = ((raw_canary_config or {}).get("judge") or {}).get("judgeConfigurations") or {}
    ai_mode = str(jc.get("aiMode", "dummy")).lower()
    policy = str(jc.get("hybridPolicy", DEFAULT_POLICY)).lower()
    ai_verdict_source = str(jc.get("hybridAiVerdict", DEFAULT_AI_VERDICT_SOURCE)).lower()
    on_ai_error = str(jc.get("onAiError", DEFAULT_ON_AI_ERROR)).lower()

    # B: the AI half, real configurable judge if aiMode is summary/raw/plot, else dummy.
    if ai_mode in AI_MODES:
        ai_jc: Dict[str, Any] = {"mode": ai_mode, "model": jc.get("model")}
        for key in ("prompt_id", "seed"):
            if jc.get(key) not in (None, ""):
                ai_jc[key] = jc[key]
        ai_settings = resolve({"judge": {"judgeConfigurations": ai_jc}})
        ai_result = judge_ai(ai_settings, raw_metric_set_pair_list, raw_canary_config)
        ai_desc = f"ai:{ai_settings.mode}:{ai_settings.model}"
        log.info("hybrid AI half: REAL judge (%s, prompt=%s)", ai_desc, ai_settings.prompt_id)
    else:
        ai_result = dummy_ai_judge(parsed)
        ai_desc = "dummy AI"
        log.info("hybrid AI half: dummy (no model)")

    # A: the genuine NetflixACAJudge, computed by Kayenta on the same pairs.
    default_result = kayenta_callback.run_default_judge(
        raw_metric_set_pair_list, raw_canary_config, pass_t, marginal_t
    )

    judge_name = f"RemoteJudge-v1.0 (hybrid:{policy}: NetflixACAJudge + {ai_desc})"
    return merge(default_result, ai_result, pass_t, marginal_t, raw_canary_config, judge_name, policy,
                 ai_verdict_source=ai_verdict_source, on_ai_error=on_ai_error)


def _ai_blindspot(ai_result: CanaryJudgeResult) -> bool:
    parts = [ai_result.score.classificationReason or ""]
    parts += [r.classificationReason or "" for r in ai_result.results]
    return text_flags_blindspot(" ".join(parts))


def merge(
    default_result: Dict[str, Any],
    ai_result: CanaryJudgeResult,
    pass_t: float,
    marginal_t: float,
    canary_config: Dict[str, Any],
    judge_name: str,
    policy: str = DEFAULT_POLICY,
    ai_verdict_source: str = DEFAULT_AI_VERDICT_SOURCE,
    on_ai_error: str = DEFAULT_ON_AI_ERROR,
) -> CanaryJudgeResult:
    if ai_verdict_source not in AI_VERDICT_SOURCES:
        raise ValueError(f"hybridAiVerdict must be one of {AI_VERDICT_SOURCES}, got {ai_verdict_source!r}")
    if on_ai_error not in ON_AI_ERROR:
        raise ValueError(f"onAiError must be one of {ON_AI_ERROR}, got {on_ai_error!r}")
    a_score = float((default_result.get("score") or {}).get("score", 0.0))   # statistical
    b_score = float(ai_result.score.score)                                   # AI

    a_by_name: Dict[str, Dict[str, Any]] = {
        str(r.get("name")): r for r in (default_result.get("results") or [])
    }
    d_verdict = _classify(a_score, pass_t, marginal_t)
    ai_meta: Dict[str, Any] = dict(ai_result.judgeMetadata or {})
    ai_ok = bool(ai_meta.get("ok", True))
    if ai_verdict_source == "score":
        # As scored offline (tools/run_experiment.py): the threshold class of the
        # AI score. The model's own label is not read.
        a_verdict = _classify(b_score, pass_t, marginal_t)
    else:
        a_verdict = ai_result.score.classification
    d_has_nonpass = any(str(r.get("classification")) not in PASS_OK for r in (default_result.get("results") or []))

    if not ai_ok:
        kind = str(ai_meta.get("error_kind") or "unspecified")
        err = str(ai_meta.get("error") or "AI judge returned no usable response")[:300]
        if on_ai_error == "fail":
            decision = PolicyDecision("Fail", 0.0,
                                      f"HYBRID: AI half failed ({kind}): {err}; onAiError=fail -> explicit FAIL",
                                      "error")
        else:
            decision = PolicyDecision(d_verdict, a_score,
                                      f"HYBRID: AI half failed ({kind}): {err}; statistical verdict only",
                                      "statistical")
    else:
        decision = decide_policy(
            policy, a_score, d_verdict, d_has_nonpass,
            b_score, a_verdict, _ai_blindspot(ai_result),
            pass_t, marginal_t,
        )

    metric_groups, all_groups = _groups_from_config(canary_config)
    default_group = all_groups[0] if all_groups else "dummy-group"

    merged_results: List[CanaryAnalysisResult] = []
    for b_r in ai_result.results:
        a_r = a_by_name.get(b_r.name, {})
        a_cls = str(a_r.get("classification", "?"))
        b_cls = b_r.classification
        # Surface any non-Pass classification (conservative for the per-metric view).
        hybrid_cls = "Pass" if (a_cls in PASS_OK and b_cls in PASS_OK) else (
            a_cls if a_cls not in PASS_OK and a_cls != "?" else b_cls
        )
        merged_results.append(
            CanaryAnalysisResult(
                name=b_r.name,
                id=b_r.id,
                tags=b_r.tags,
                classification=hybrid_cls,
                classificationReason=f"hybrid[{policy}]: statistical={a_cls}, ai={b_cls}",
                groups=metric_groups.get(b_r.name) or [default_group],
                controlMetadata=a_r.get("controlMetadata") or b_r.controlMetadata,
                experimentMetadata=a_r.get("experimentMetadata") or b_r.experimentMetadata,
                resultMetadata={"statisticalScore": a_score, "aiScore": b_score, "policy": policy,
                                "followed": decision.followed},
            )
        )

    log.info("hybrid[%s] -> %s (%.2f) [statistical=%.1f(%s) ai=%.1f(%s via %s, ok=%s) followed=%s]",
             policy, decision.verdict, decision.score, a_score, d_verdict, b_score, a_verdict,
             ai_verdict_source, ai_ok, decision.followed)
    judge_metadata: Dict[str, Any] = {
        "ok": ai_ok,
        "policy": policy,
        "hybridAiVerdict": ai_verdict_source,
        "aiVerdictUsed": a_verdict if ai_ok else "",
        "aiLabel": ai_result.score.classification,
        "followed": decision.followed,
        "ai": ai_meta,
    }
    if not ai_ok:
        judge_metadata.update({"error_kind": "ai_half_failed", "error": decision.reason,
                               "onAiError": on_ai_error})
    return CanaryJudgeResult(
        judgeName=judge_name,
        results=merged_results,
        groupScores=_group_scores(merged_results, all_groups, decision.score, decision.verdict, decision.reason),
        score=CanaryJudgeScore(score=decision.score, classification=decision.verdict,
                               classificationReason=decision.reason),
        judgeMetadata=judge_metadata,
    )
