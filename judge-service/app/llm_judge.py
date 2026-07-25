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

# Frozen prompt
# Bump PROMPT_VERSION only when the prompt text below changes. For the scored
# experiment the prompt is frozen at v1: don't tune it against scenario outcomes,
# that would overfit to the baseline's known weaknesses. It's the same for every
# model and representation, never includes a ground-truth label, always presents
# control before experiment (position bias), and bounds the rationale (verbosity
# bias). Changing it means bumping PROMPT_VERSION and re-running the whole sweep.
PROMPT_VERSION = "v1-frozen-2026-06"

# Fixed verdict schema (one template, never includes any ground-truth label).
SYSTEM_PROMPT = (
    "You are an automated canary-release judge. You compare a baseline (control) "
    "against a new version (canary, the experiment) using the provided metrics "
    "and decide whether the canary is safe to promote. Be objective and concise."
)

VERDICT_INSTRUCTION = (
    "Decide a verdict for the canary. Reply with STRICT JSON ONLY, no prose, "
    "matching exactly this schema:\n"
    "{\n"
    '  "overallVerdict": "pass" | "marginal" | "fail",\n'
    '  "overallScore": <integer 0-100, higher = healthier canary>,\n'
    '  "metrics": [ { "name": "<metric name>", '
    '"classification": "pass" | "high" | "low" | "nodata", "reason": "<short>" } ],\n'
    '  "rationale": "<= 60 words"\n'
    "}\n"
    "Judge HOLISTICALLY — do not look at the mean/median alone. A canary is "
    "unhealthy ('high') if, versus control, it shows materially higher spread/"
    "variance (instability/flapping), a worse tail (p90/p95/p99/max), or an "
    "emerging upward trend over time (positive slope), EVEN IF the median is "
    "similar. Rules: clearly worse on an error/latency/resource metric is 'high'; "
    "clearly better/lower is 'low'; comparable and stable is 'pass'; missing data "
    "is 'nodata'. overallScore is 0-100 (higher = healthier). Rationale < 60 words."
)

_CLASS_MAP = {"pass": "Pass", "high": "High", "low": "Low", "nodata": "Nodata"}
_VERDICT_MAP = {"pass": "Pass", "marginal": "Marginal", "fail": "Fail"}


def _client(settings: JudgeSettings) -> OpenAI:
    base = settings.litellm_base_url.rstrip("/")
    if not base.endswith("/v1"):
        base = base + "/v1"
    # LiteLLM accepts any key unless a master key is configured.
    return OpenAI(base_url=base, api_key=os.environ.get("LITELLM_API_KEY", "sk-noop"))


def _build_messages(
    settings: JudgeSettings, metric_set_pair_list: List[Dict[str, Any]]
) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """Return (messages, image_b64 or None)."""
    image_b64: Optional[str] = None
    if settings.mode == "plot" and settings.modality == "vision":
        image_b64 = render_plot_png(metric_set_pair_list)

    # For plot we still include the summary text as a companion (legible context),
    # plus the image. For summary/raw we send text only.
    text_mode = "summary" if settings.mode == "plot" else settings.mode
    rep_text = representation_text(text_mode, metric_set_pair_list)
    user_text = f"{rep_text}\n\n{VERDICT_INSTRUCTION}"

    if image_b64:
        user_content: Any = [
            {"type": "text", "text": user_text},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
        ]
    else:
        user_content = user_text

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    return messages, image_b64


_UNSUPPORTED_SAMPLING_RE = re.compile(r"`?(temperature|top_p)`?\s+is\s+deprecated", re.IGNORECASE)


def _call_model(settings: JudgeSettings, messages: List[Dict[str, Any]]) -> str:
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
    return resp.choices[0].message.content or ""


_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def _iter_balanced_objects(text: str):
    """Yield every top-level {...} substring with balanced braces (string-aware).

    More robust than a greedy `\\{.*\\}` regex, which over-captures when the model
    emits prose or several objects (e.g. reasoning models that wrap JSON)."""
    depth = 0
    start = -1
    in_str = False
    esc = False
    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start >= 0:
                    yield text[start : i + 1]


def _looks_like_verdict(obj: Any) -> bool:
    return isinstance(obj, dict) and (
        "overallVerdict" in obj or "overallScore" in obj or "metrics" in obj
    )


def _parse_json(content: str) -> Optional[Dict[str, Any]]:
    """Tolerant JSON extraction that prefers a real verdict object.

    Handles clean JSON, markdown ```json fences, reasoning models that emit a
    <think>...</think> preamble (e.g. deepseek-r1), and stray prose around the
    object. Scans every balanced {...} block and keeps the first that looks like
    a verdict, falling back to the first valid object."""
    if not content:
        return None
    # 1) strip reasoning preamble and code fences.
    cleaned = _THINK_RE.sub(" ", content)
    fence = _FENCE_RE.search(cleaned)
    candidates: List[str] = []
    if fence:
        candidates.append(fence.group(1))
    candidates.append(cleaned)

    # 2) try a direct parse of each candidate first (cheapest).
    for cand in candidates:
        try:
            obj = json.loads(cand.strip())
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass

    # 3) scan balanced {...} blocks; prefer one that looks like a verdict.
    fallback: Optional[Dict[str, Any]] = None
    for block in _iter_balanced_objects(cleaned):
        try:
            obj = json.loads(block)
        except json.JSONDecodeError:
            continue
        if _looks_like_verdict(obj):
            return obj
        if fallback is None and isinstance(obj, dict):
            fallback = obj
    return fallback


def _save_log(settings: JudgeSettings, messages, image_b64, raw, parsed) -> Optional[str]:
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
            "prompt_version": PROMPT_VERSION,
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
    settings: JudgeSettings, metric_set_pair_list: List[Dict[str, Any]], err: str, canary_config: Dict[str, Any]
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
    )


def judge_ai(
    settings: JudgeSettings,
    metric_set_pair_list: List[Dict[str, Any]],
    canary_config: Optional[Dict[str, Any]] = None,
) -> CanaryJudgeResult:
    """Run the configurable AI judge. Returns a valid CanaryJudgeResult always."""
    canary_config = canary_config or {}
    messages, image_b64 = _build_messages(settings, metric_set_pair_list)
    log.info(
        "AI judge: mode=%s model=%s modality=%s image=%s",
        settings.mode, settings.model, settings.modality, bool(image_b64),
    )

    raw = ""
    parsed: Optional[Dict[str, Any]] = None
    err: Optional[str] = None
    for attempt in (1, 2):  # one tolerant retry
        try:
            raw = _call_model(settings, messages)
            parsed = _parse_json(raw)
            if parsed is not None:
                break
            err = "model did not return valid JSON"
            # tighten instruction for the retry
            messages = messages + [
                {"role": "user", "content": "Your previous reply was not valid JSON. Reply with STRICT JSON ONLY."}
            ]
        except Exception as e:  # noqa: BLE001
            err = str(e)
            log.warning("AI call attempt %d failed: %s", attempt, err)

    log_path = _save_log(settings, messages, image_b64, raw, parsed)
    log.info("AI judge raw response (logged at %s): %s", log_path, (raw or "")[:500])

    if parsed is None:
        log.warning("AI judge falling back to Error result: %s", err)
        return _error_result(settings, metric_set_pair_list, err or "no response", canary_config)

    result = _map_to_result(settings, metric_set_pair_list, parsed, canary_config)
    log.info(
        "AI verdict: score=%.1f classification=%s per-metric=%s",
        result.score.score, result.score.classification,
        [(r.name, r.classification) for r in result.results],
    )
    return result
