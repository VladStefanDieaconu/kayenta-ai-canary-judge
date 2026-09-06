"""The configurable AI judge (LLM + VLM), provider-agnostic.

A single code path for every model and provider. It:
  1. builds the representation text (summary | raw) and, for `plot`, a chart image;
  2. calls one OpenAI-compatible endpoint (the LiteLLM gateway) with the chosen
     model alias, branching only on modality (text vs vision), not on provider;
  3. enforces a strict JSON verdict (with a tolerant parser and one retry);
  4. maps the verdict onto the same valid CanaryJudgeResult the dummy returns,
     attaching computed stats metadata (Kayenta's aggregation needs stats.count);
  5. logs the prompt, image (saved and hashed), raw response, parsed verdict, and
     the model alias.

Never crashes the analysis: on any model or parse failure it returns a valid
CanaryJudgeResult with an "Error" classification, the same way the dummy always
returns something valid.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI

from . import prompt_registry, verdict_parser
from .dummy_judge import _stats  # NPE-safe {"stats": {count, ...}} metadata
from .judge_config import JudgeSettings
from .models import (
    CanaryAnalysisResult,
    CanaryJudgeGroupScore,
    CanaryJudgeResult,
    CanaryJudgeScore,
)
from .representations import (
    clean,
    render_plot_png,
    representation_text,
)

log = logging.getLogger("judge-service.llm")

LOG_DIR = Path(os.environ.get("AI_LOG_DIR", "/app/data/ai-logs"))

# The rubric now lives in prompts/, one file per variant, selected by id (see
# prompt_registry). For the scored experiment it is frozen at v1: don't tune it
# against scenario outcomes, that would overfit to the baseline's known
# weaknesses. Comparing variants is a configuration change, not a code change,
# and the resolved id and text hash are stamped into every row this module
# produces so no result is ever ambiguous about which rubric judged it.

_CLASS_MAP = {"pass": "Pass", "high": "High", "low": "Low", "nodata": "Nodata"}
_VERDICT_MAP = {"pass": "Pass", "marginal": "Marginal", "fail": "Fail"}


def _client(settings: JudgeSettings) -> OpenAI:
    base = settings.litellm_base_url.rstrip("/")
    if not base.endswith("/v1"):
        base = base + "/v1"
    # LiteLLM accepts any key unless a master key is configured.
    return OpenAI(base_url=base, api_key=os.environ.get("LITELLM_API_KEY", "sk-noop"))


def _build_messages(
    settings: JudgeSettings,
    metric_set_pair_list: List[Dict[str, Any]],
    prompt: prompt_registry.Prompt,
) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """Return (messages, image_b64 or None)."""
    image_b64: Optional[str] = None
    if settings.mode == "plot" and settings.modality == "vision":
        image_b64 = render_plot_png(metric_set_pair_list)

    # For plot we still include the summary text as a companion (legible context),
    # plus the image. For summary/raw we send text only.
    text_mode = "summary" if settings.mode == "plot" else settings.mode
    rep_text = representation_text(text_mode, metric_set_pair_list)
    user_text = f"{rep_text}\n\n{prompt.rubric}"

    if image_b64:
        user_content: Any = [
            {"type": "text", "text": user_text},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
        ]
    else:
        user_content = user_text

    messages = [
        {"role": "system", "content": prompt.system},
        {"role": "user", "content": user_content},
    ]
    return messages, image_b64


_UNSUPPORTED_SAMPLING_RE = re.compile(r"`?(temperature|top_p)`?\s+is\s+deprecated", re.IGNORECASE)


def _usage_dict(resp: Any) -> Dict[str, Any]:
    """The gateway's usage block, flattened to the three counts that matter.

    Captured because without it the paper's cost figure is an estimate. It also
    makes a truncation diagnosis immediate: completion_tokens equal to the cap
    alongside finish_reason='length' is the gpt-oss failure exactly.
    """
    usage = getattr(resp, "usage", None)
    if usage is None:
        return {}
    return {
        "prompt_tokens": getattr(usage, "prompt_tokens", None),
        "completion_tokens": getattr(usage, "completion_tokens", None),
        "total_tokens": getattr(usage, "total_tokens", None),
    }


def _call_model(
    settings: JudgeSettings, messages: List[Dict[str, Any]]
) -> Tuple[str, Optional[str], Dict[str, Any]]:
    """Return (content, finish_reason, usage). Content may legitimately be ''."""
    client = _client(settings)
    kwargs: Dict[str, Any] = {
        "model": settings.model,
        "messages": messages,
        "temperature": settings.temperature,
        "top_p": settings.top_p,
        "max_tokens": settings.max_tokens,
    }
    if settings.json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    if settings.seed is not None:
        kwargs["seed"] = settings.seed  # dropped by LiteLLM for backends that lack it
    try:
        resp = client.chat.completions.create(**kwargs)
    except Exception as e:  # noqa: BLE001
        # Some newer hosted models (e.g. Bedrock's latest Claude releases) reject an
        # explicit temperature/top_p outright rather than letting LiteLLM's drop_params
        # strip it silently. drop_params only knows about params it has a static mapping
        # for, and brand-new models aren't in it yet. Retry once without the sampling
        # knobs. This branch doesn't fire for a model that accepts them, so it leaves
        # the existing local models unchanged.
        if not _UNSUPPORTED_SAMPLING_RE.search(str(e)):
            raise
        log.warning("model rejected temperature/top_p (%s); retrying without them", settings.model)
        fallback_kwargs = {k: v for k, v in kwargs.items() if k not in ("temperature", "top_p")}
        resp = client.chat.completions.create(**fallback_kwargs)
    choice = resp.choices[0] if resp.choices else None
    content = (getattr(getattr(choice, "message", None), "content", None) or "") if choice else ""
    finish_reason = getattr(choice, "finish_reason", None) if choice else None
    return content, finish_reason, _usage_dict(resp)


# The tolerant parser lives in verdict_parser so it can be replayed against the
# archive without the OpenAI SDK or a gateway. Re-exported under their previous
# private names because callers (and the replay tool) already reference them.
_parse_json = verdict_parser.parse_json
_iter_balanced_objects = verdict_parser.iter_balanced_objects
_looks_like_verdict = verdict_parser.looks_like_verdict


def _save_log(settings: JudgeSettings, messages, image_b64, raw, parsed,
              prompt: prompt_registry.Prompt, outcome: Dict[str, Any]) -> Optional[str]:
    """Persist prompt/response/verdict (+ image) for reproducibility."""
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%dT%H%M%S")
        stamp = f"{ts}-{settings.mode}-{settings.model}"
        image_hash = None
        if image_b64:
            import base64 as _b64

            img_bytes = _b64.b64decode(image_b64)
            image_hash = hashlib.sha256(img_bytes).hexdigest()[:16]
            (LOG_DIR / f"{stamp}-{image_hash}.png").write_bytes(img_bytes)
        # Strip the inline image from the saved messages (keep a pointer).
        safe_msgs = json.loads(json.dumps(messages))
        for msg in safe_msgs:
            if isinstance(msg.get("content"), list):
                for part in msg["content"]:
                    if part.get("type") == "image_url":
                        part["image_url"] = {"url": f"<png sha256:{image_hash}>"}
        record = {
            # prompt_version is retained under its original name so the 20k
            # archived payloads stay readable by the same tooling; prompt_id and
            # prompt_hash are the fields to key on from here on.
            "prompt_version": prompt.id,
            "prompt_id": prompt.id,
            "prompt_hash": prompt.hash,
            "mode": settings.mode,
            "model": settings.model,
            "modality": settings.modality,
            "temperature": settings.temperature,
            "top_p": settings.top_p,
            "max_tokens": settings.max_tokens,
            "seed": settings.seed,
            "image_sha256": image_hash,
            "messages": safe_msgs,
            "raw_response": raw,
            "parsed_verdict": parsed,
            "finish_reason": outcome.get("finish_reason"),
            "usage": outcome.get("usage") or {},
            "error_kind": outcome.get("error_kind", ""),
            "error": outcome.get("error", ""),
        }
        path = LOG_DIR / f"{stamp}.json"
        path.write_text(json.dumps(record, indent=2))
        return str(path)
    except Exception as e:  # noqa: BLE001 - logging must never break judging
        log.warning("could not write AI log: %s", e)
        return None


def _metadata_for(pair: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    values = pair.get("values") or {}
    return _stats(clean(values.get("control"))), _stats(clean(values.get("experiment")))


def _groups_from_config(canary_config: Dict[str, Any]) -> Tuple[Dict[str, List[str]], List[str]]:
    """Map metricName -> its configured groups, and the ordered list of all groups.

    Referee renders group scores against the config's classifier.groupWeights, so
    the result's per-metric `groups` and the `groupScores` have to use the config's
    real group names (not a hardcoded 'dummy-group') or the report won't render.
    """
    metric_groups: Dict[str, List[str]] = {}
    all_groups: List[str] = []
    for m in (canary_config or {}).get("metrics", []) or []:
        groups = [str(g) for g in (m.get("groups") or [])]
        metric_groups[str(m.get("name"))] = groups
        for g in groups:
            if g not in all_groups:
                all_groups.append(g)
    if not all_groups:
        weights = ((canary_config or {}).get("classifier") or {}).get("groupWeights") or {}
        all_groups = [str(g) for g in weights.keys()] or ["dummy-group"]
    return metric_groups, all_groups


def _judge_metadata(
    settings: JudgeSettings, prompt: prompt_registry.Prompt, outcome: Dict[str, Any], ok: bool
) -> Dict[str, Any]:
    """Provenance and outcome, carried out of band from the Kayenta contract."""
    return {
        "ok": ok,
        "error_kind": outcome.get("error_kind", ""),
        "error": outcome.get("error", ""),
        "prompt_id": prompt.id,
        "prompt_hash": prompt.hash,
        "mode": settings.mode,
        "model": settings.model,
        "modality": settings.modality,
        "max_tokens": settings.max_tokens,
        "finish_reason": outcome.get("finish_reason"),
        "usage": outcome.get("usage") or {},
        "attempts": outcome.get("attempts", 0),
    }


def _map_to_result(
    settings: JudgeSettings,
    metric_set_pair_list: List[Dict[str, Any]],
    parsed: Optional[Dict[str, Any]],
    canary_config: Dict[str, Any],
) -> CanaryJudgeResult:
    judge_name = f"RemoteJudge-v1.0 (ai:{settings.mode}:{settings.model})"
    metric_groups, all_groups = _groups_from_config(canary_config)
    default_group = all_groups[0] if all_groups else "dummy-group"

    by_name: Dict[str, Dict[str, Any]] = {}
    for m in (parsed or {}).get("metrics", []) or []:
        if isinstance(m, dict) and m.get("name"):
            by_name[str(m["name"])] = m

    results: List[CanaryAnalysisResult] = []
    for pair in metric_set_pair_list:
        name = str(pair.get("name") or pair.get("id") or "metric")
        control_md, experiment_md = _metadata_for(pair)
        m = by_name.get(name, {})
        raw_cls = str(m.get("classification", "")).lower()
        classification = _CLASS_MAP.get(raw_cls)
        if classification is None:
            # no data present? then Nodata; else Error (model said nothing usable)
            classification = "Nodata" if control_md["stats"]["count"] == 0 else "Error"
        reason = (str(m.get("reason", "")) or f"{settings.mode} judge ({settings.model})")[:300]
        groups = metric_groups.get(name) or [default_group]
        results.append(
            CanaryAnalysisResult(
                name=name,
                id=str(pair.get("id") or name),
                tags=pair.get("tags") or {},
                classification=classification,
                classificationReason=reason,
                groups=groups,
                controlMetadata=control_md,
                experimentMetadata=experiment_md,
                resultMetadata={"mode": settings.mode, "model": settings.model},
            )
        )

    # Overall score/verdict from the model (fallback derived from per-metric).
    score_val: float
    verdict: str
    if parsed and isinstance(parsed.get("overallScore"), (int, float)):
        score_val = float(max(0.0, min(100.0, float(parsed["overallScore"]))))
        verdict = _VERDICT_MAP.get(str(parsed.get("overallVerdict", "")).lower(), "")
        if not verdict:
            verdict = "Pass" if score_val >= 75 else ("Marginal" if score_val >= 50 else "Fail")
    else:
        non_pass = sum(1 for r in results if r.classification not in ("Pass", "Nodata"))
        score_val = float(max(0, 100 - 30 * non_pass))
        verdict = "Pass" if score_val >= 75 else ("Marginal" if score_val >= 50 else "Fail")

    rationale = (str((parsed or {}).get("rationale", "")) or f"AI judge {settings.mode}/{settings.model}")[:500]
    return CanaryJudgeResult(
        judgeName=judge_name,
        results=results,
        groupScores=_group_scores(results, all_groups, score_val, verdict, rationale),
        score=CanaryJudgeScore(score=score_val, classification=verdict, classificationReason=rationale),
    )


def _group_scores(
    results: List[CanaryAnalysisResult], all_groups: List[str], overall: float, verdict: str, reason: str
) -> List[CanaryJudgeGroupScore]:
    """One CanaryJudgeGroupScore per configured group (names have to match the
    config's groupWeights so Referee can render). Per-group score: 100 if every
    metric in that group passed, else the overall score."""
    out: List[CanaryJudgeGroupScore] = []
    for g in all_groups:
        members = [r for r in results if g in (r.groups or [])]
        bad = [r for r in members if r.classification not in ("Pass", "Nodata")]
        if members and not bad:
            gscore, gclass = 100.0, "Pass"
        else:
            gscore, gclass = overall, verdict
        out.append(CanaryJudgeGroupScore(name=g, score=gscore, classification=gclass, classificationReason=reason))
    return out


def _error_result(
    settings: JudgeSettings, metric_set_pair_list: List[Dict[str, Any]], err: str,
    canary_config: Dict[str, Any], judge_metadata: Optional[Dict[str, Any]] = None,
) -> CanaryJudgeResult:
    """Valid CanaryJudgeResult signalling the AI path failed (never crash)."""
    metric_groups, all_groups = _groups_from_config(canary_config)
    default_group = all_groups[0] if all_groups else "dummy-group"
    results: List[CanaryAnalysisResult] = []
    for pair in metric_set_pair_list:
        name = str(pair.get("name") or pair.get("id") or "metric")
        control_md, experiment_md = _metadata_for(pair)
        results.append(
            CanaryAnalysisResult(
                name=name,
                id=str(pair.get("id") or name),
                tags=pair.get("tags") or {},
                classification="Error",
                classificationReason=f"AI judge error ({settings.mode}/{settings.model}): {err}"[:300],
                groups=metric_groups.get(name) or [default_group],
                controlMetadata=control_md,
                experimentMetadata=experiment_md,
                resultMetadata={"mode": settings.mode, "model": settings.model, "error": True},
            )
        )
    reason = f"AI judge error ({settings.mode}/{settings.model}): {err}"[:500]
    return CanaryJudgeResult(
        judgeName=f"RemoteJudge-v1.0 (ai:{settings.mode}:{settings.model})",
        results=results,
        groupScores=[CanaryJudgeGroupScore(name=g, score=0.0, classification="Fail", classificationReason=reason)
                     for g in all_groups],
        score=CanaryJudgeScore(score=0.0, classification="Fail", classificationReason=reason),
        judgeMetadata=judge_metadata or {"ok": False, "error": err, "error_kind": "unspecified"},
    )


def judge_ai(
    settings: JudgeSettings,
    metric_set_pair_list: List[Dict[str, Any]],
    canary_config: Optional[Dict[str, Any]] = None,
) -> CanaryJudgeResult:
    """Run the configurable AI judge. Returns a valid CanaryJudgeResult always."""
    canary_config = canary_config or {}

    # Resolve the rubric first. An unknown id is a configuration error and must
    # stop the call: silently judging under some other prompt is precisely the
    # class of fault that produced a run nobody could reconstruct afterwards.
    try:
        prompt = prompt_registry.load(settings.prompt_id)
    except (prompt_registry.UnknownPromptError, ValueError, OSError) as e:
        log.error("prompt selection failed: %s", e)
        return _error_result(
            settings, metric_set_pair_list, f"prompt selection failed: {e}", canary_config,
            {"ok": False, "error_kind": "unknown_prompt", "error": str(e),
             "prompt_id": settings.prompt_id, "prompt_hash": "",
             "mode": settings.mode, "model": settings.model},
        )

    messages, image_b64 = _build_messages(settings, metric_set_pair_list, prompt)
    log.info(
        "AI judge: mode=%s model=%s modality=%s image=%s prompt=%s(%s)",
        settings.mode, settings.model, settings.modality, bool(image_b64),
        prompt.id, prompt.hash,
    )

    raw = ""
    parsed: Optional[Dict[str, Any]] = None
    outcome: Dict[str, Any] = {"error_kind": "", "error": "", "finish_reason": None, "usage": {}}
    attempts = 0
    for attempt in (1, 2):  # one tolerant retry
        attempts = attempt
        try:
            raw, finish_reason, usage = _call_model(settings, messages)
            outcome["finish_reason"] = finish_reason
            outcome["usage"] = usage
            if not (raw or "").strip():
                # An empty completion is not a verdict of any kind. It is the
                # gpt-oss failure: the model spent its whole budget on reasoning
                # tokens and returned no content, with finish_reason='length'.
                # Kept distinct from a parse failure because the two have
                # different causes and different fixes -- one is a token budget,
                # the other is a model that will not follow the schema.
                outcome["error_kind"] = "empty_completion"
                outcome["error"] = (
                    f"model returned an empty completion (finish_reason={finish_reason!r}, "
                    f"completion_tokens={usage.get('completion_tokens')}, max_tokens={settings.max_tokens})"
                )
                parsed = None
            else:
                parsed = _parse_json(raw)
                if parsed is not None:
                    outcome["error_kind"] = ""
                    outcome["error"] = ""
                    break
                outcome["error_kind"] = "parse_failure"
                outcome["error"] = "model returned a non-empty completion that is not valid JSON"
            # tighten instruction for the retry
            messages = messages + [
                {"role": "user", "content": "Your previous reply was not valid JSON. Reply with STRICT JSON ONLY."}
            ]
        except Exception as e:  # noqa: BLE001
            outcome["error_kind"] = "api_error"
            outcome["error"] = str(e)
            log.warning("AI call attempt %d failed: %s", attempt, outcome["error"])
    outcome["attempts"] = attempts

    log_path = _save_log(settings, messages, image_b64, raw, parsed, prompt, outcome)
    log.info("AI judge raw response (logged at %s): %s", log_path, (raw or "")[:500])

    if parsed is None:
        # Error, never a verdict. The returned object stays a structurally valid
        # CanaryJudgeResult so a live analysis does not crash, but judgeMetadata
        # marks it not-ok so the experiment runner records an error row and its
        # retry and abort logic can see it.
        log.error(
            "AI judge FAILED (%s) mode=%s model=%s prompt=%s: %s",
            outcome["error_kind"], settings.mode, settings.model, prompt.id, outcome["error"],
        )
        return _error_result(
            settings, metric_set_pair_list, outcome["error"] or "no response", canary_config,
            _judge_metadata(settings, prompt, outcome, ok=False),
        )

    result = _map_to_result(settings, metric_set_pair_list, parsed, canary_config)
    result.judgeMetadata = _judge_metadata(settings, prompt, outcome, ok=True)
    log.info(
        "AI verdict: score=%.1f classification=%s per-metric=%s",
        result.score.score, result.score.classification,
        [(r.name, r.classification) for r in result.results],
    )
    return result
