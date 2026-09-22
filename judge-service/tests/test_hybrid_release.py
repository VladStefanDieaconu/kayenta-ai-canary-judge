"""Checks for the v1.0.1 service changes, with every network call stubbed.

  1. The hybrid's AI verdict comes from the score by default, as the offline
     harness scored it; `hybridAiVerdict=label` restores the v1.0.0 behaviour.
  2. A failed AI half is not a verdict: statistical verdict alone by default,
     explicit Fail with `onAiError=fail`, judgeMetadata.ok false either way.
  3. prompt_id and seed reach the hybrid's AI half.
  4. score_check and truncated are recorded, and become errors only when the
     opt-in environment variables are set.

Run from judge-service/ with the service's dependencies installed:
    python -m tests.test_hybrid_release
or inside the container:
    docker compose exec -T judge-service python -m tests.test_hybrid_release
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
os.environ.setdefault("MODELS_REGISTRY", str(HERE.parent / "models.yaml"))
os.environ.setdefault("PROMPTS_DIR", str(HERE.parent.parent / "prompts"))
os.environ.setdefault("AI_LOG_DIR", str(Path(os.environ.get("TMPDIR", "/tmp")) / "judge-test-ai-logs"))
sys.path.insert(0, str(HERE.parent))

from app import hybrid, kayenta_callback, llm_judge  # noqa: E402
from app.judge_config import JudgeSettings  # noqa: E402

PAIRS = [{"name": "m1", "id": "m1", "tags": {},
          "values": {"control": [1.0, 1.1, 0.9], "experiment": [1.0, 1.2, 0.8]}}]
CONFIG = {"metrics": [{"name": "m1", "groups": ["g"]}],
          "classifier": {"groupWeights": {"g": 100}}}
CLEAN_STAT_PASS = {"score": {"score": 100.0},
                   "results": [{"name": "m1", "classification": "Pass"}]}


def _settings(max_tokens: int = 512) -> JudgeSettings:
    return JudgeSettings(mode="summary", model="stub-model", modality="text", temperature=0.0,
                         top_p=1.0, max_tokens=max_tokens, seed=42, litellm_base_url="http://stub")


def _ai(label: str, score: float, reason: str):
    parsed = {"overallVerdict": label, "overallScore": score, "rationale": reason,
              "metrics": [{"name": "m1", "classification": "high", "reason": reason}]}
    res = llm_judge._map_to_result(_settings(), PAIRS, parsed, CONFIG)
    res.judgeMetadata = {"ok": True}
    return res


def check(name: str, cond: bool, failures: list) -> None:
    print(f"  {name:62s} {'ok' if cond else 'FAIL'}")
    if not cond:
        failures.append(name)


def main() -> int:
    failures: list = []

    # 1. verdict source. Label 'fail', score 80, a blind-spot keyword cited: the
    # harness (classify(80) = Pass) keeps the clean statistical PASS; the label
    # path overrides it to FAIL.
    ai = _ai("fail", 80, "variance is materially higher")
    d = hybrid.merge(CLEAN_STAT_PASS, ai, 75, 50, CONFIG, "t", "gated")
    check("default hybridAiVerdict=score keeps clean PASS (label fail, 80)", d.score.classification == "Pass", failures)
    check("default records aiVerdictUsed=Pass", d.judgeMetadata.get("aiVerdictUsed") == "Pass", failures)
    d = hybrid.merge(CLEAN_STAT_PASS, ai, 75, 50, CONFIG, "t", "gated", ai_verdict_source="label")
    check("hybridAiVerdict=label reproduces v1.0.0 override to FAIL", d.score.classification == "Fail", failures)
    ai = _ai("pass", 20, "looks fine")
    d = hybrid.merge(CLEAN_STAT_PASS, ai, 75, 50, CONFIG, "t", "gated")
    check("default: label pass at score 20 overrides to FAIL (as harness)", d.score.classification == "Fail", failures)

    # 2. failed AI half
    err = llm_judge._error_result(_settings(), PAIRS, "stubbed outage", CONFIG,
                                  {"ok": False, "error_kind": "api_error", "error": "stubbed outage"})
    d = hybrid.merge(CLEAN_STAT_PASS, err, 75, 50, CONFIG, "t", "gated")
    check("failed AI half: statistical verdict only (Pass)", d.score.classification == "Pass", failures)
    check("failed AI half: judgeMetadata ok=false, error_kind", (d.judgeMetadata.get("ok") is False
          and d.judgeMetadata.get("error_kind") == "ai_half_failed"), failures)
    d = hybrid.merge(CLEAN_STAT_PASS, err, 75, 50, CONFIG, "t", "gated", on_ai_error="fail")
    check("failed AI half, onAiError=fail: explicit Fail at 0", (d.score.classification == "Fail"
          and d.score.score == 0.0 and d.judgeMetadata.get("ok") is False), failures)

    # 3. prompt_id and seed forwarded
    seen = {}
    orig_ai, orig_cb = hybrid.judge_ai, kayenta_callback.run_default_judge
    hybrid.judge_ai = lambda settings, pairs, cfg: seen.update(prompt=settings.prompt_id, seed=settings.seed) or _ai("pass", 90, "fine")
    kayenta_callback.run_default_judge = lambda *a, **k: CLEAN_STAT_PASS
    try:
        cfg = dict(CONFIG, judge={"judgeConfigurations": {"mode": "hybrid", "aiMode": "summary",
                                                          "model": "qwen-llm", "prompt_id": "v2-operational-2026-07",
                                                          "seed": 7}})
        hybrid.judge_hybrid(None, PAIRS, cfg, {"pass": 75, "marginal": 50})
    finally:
        hybrid.judge_ai, kayenta_callback.run_default_judge = orig_ai, orig_cb
    check("prompt_id forwarded to AI half", seen.get("prompt") == "v2-operational-2026-07", failures)
    check("seed forwarded to AI half", seen.get("seed") == 7, failures)

    # 4. score_check and truncation
    check("score_check out_of_range", llm_judge.score_check({"overallScore": 150}) == "out_of_range", failures)
    check("score_check boolean", llm_judge.score_check({"overallScore": True}) == "boolean", failures)
    check("score_check non_finite", llm_judge.score_check({"overallScore": float("nan")}) == "non_finite", failures)
    check("score_check ok", llm_judge.score_check({"overallScore": 42}) == "ok", failures)
    check("truncated at cap with finish_reason stop",
          llm_judge.is_truncated("stop", {"completion_tokens": 1024}, 1024), failures)
    check("not truncated below cap", not llm_judge.is_truncated("stop", {"completion_tokens": 10}, 1024), failures)

    body = ('{"overallVerdict": "pass", "overallScore": 150, "rationale": "fine", '
            '"metrics": [{"name": "m1", "classification": "pass", "reason": "ok"}]}')
    orig_call = llm_judge._call_model
    try:
        llm_judge._call_model = lambda s, m: (body, "stop", {"completion_tokens": 512})
        for var in ("JUDGE_STRICT_SCORE", "JUDGE_TRUNCATION_IS_ERROR"):
            os.environ.pop(var, None)
        r = llm_judge.judge_ai(_settings(max_tokens=512), PAIRS, CONFIG)
        check("default: out-of-range score recorded, still a verdict", (r.judgeMetadata.get("ok") is True
              and r.judgeMetadata.get("score_check") == "out_of_range" and r.score.score == 100.0), failures)
        check("default: truncation recorded, still a verdict", r.judgeMetadata.get("truncated") is True, failures)
        os.environ["JUDGE_STRICT_SCORE"] = "1"
        r = llm_judge.judge_ai(_settings(max_tokens=512), PAIRS, CONFIG)
        check("JUDGE_STRICT_SCORE=1: parse_failure error", (r.judgeMetadata.get("ok") is False
              and r.judgeMetadata.get("error_kind") == "parse_failure"), failures)
        os.environ.pop("JUDGE_STRICT_SCORE")
        os.environ["JUDGE_TRUNCATION_IS_ERROR"] = "1"
        r = llm_judge.judge_ai(_settings(max_tokens=512), PAIRS, CONFIG)
        check("JUDGE_TRUNCATION_IS_ERROR=1: truncated error", (r.judgeMetadata.get("ok") is False
              and r.judgeMetadata.get("error_kind") == "truncated"), failures)
    finally:
        os.environ.pop("JUDGE_TRUNCATION_IS_ERROR", None)
        llm_judge._call_model = orig_call

    if failures:
        print(f"FAILED: {failures}")
        return 1
    print("all v1.0.1 service checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
