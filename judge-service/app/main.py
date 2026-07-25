"""FastAPI app implementing Kayenta's Remote Judge contract.

Single judging route: POST /judge  (path fixed by RemoteJudgeService.java).
Also exposes GET /health for container/host health checks.

On every /judge call we log the full incoming JSON to stdout, a record of the
precise RemoteJudgeRequest / MetricSetPair shapes for this Kayenta version.
"""

from __future__ import annotations

import json
import logging
import sys

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from .dummy_judge import judge
from .hybrid import judge_hybrid
from .judge_config import AI_MODES, DUMMY_MODES, resolve
from .llm_judge import judge_ai
from .models import RemoteJudgeRequest

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger("judge-service")

app = FastAPI(
    title="canaryLLM remote judge",
    description="Configurable implementation of Kayenta's Remote Judge (RemoteJudge-v1.0): dummy | AI (summary/raw/plot) | hybrid.",
    version="1.0.0",
)


@app.get("/health")
def health():
    return {"status": "UP"}


@app.get("/")
def root():
    return {"service": "canaryLLM-judge-service", "judge": "RemoteJudge-v1.0", "stub": True}


@app.post("/judge")
async def judge_endpoint(request: Request):
    """Receive a RemoteJudgeRequest, log it verbatim, return a CanaryJudgeResult."""
    raw = await request.body()
    try:
        payload = json.loads(raw) if raw else {}
    except json.JSONDecodeError:
        log.exception("Failed to parse incoming /judge body as JSON")
        payload = {}

    # Full verbatim log: a record of the exact request schema Kayenta sends.
    log.info("==== INCOMING RemoteJudgeRequest ====")
    log.info(json.dumps(payload, indent=2, default=str))
    log.info("==== END RemoteJudgeRequest ====")

    parsed = RemoteJudgeRequest.model_validate(payload)
    pair_names = [p.name or p.id for p in parsed.metricSetPairList]

    # Mode and model are chosen per-analysis from the canary config's judge block:
    #   judge.judgeConfigurations.mode  = dummy | ai | hybrid | summary | raw | plot
    #   judge.judgeConfigurations.model = a registry alias (for the AI modes)
    # with env fallbacks (JUDGE_MODE defaults to 'dummy', JUDGE_MODEL). A single
    # published service covers every role and the config picks; dummy/ai stay on
    # the no-op path.
    canary_config = payload.get("canaryConfig") or {}
    settings = resolve(canary_config)
    mode = settings.mode
    log.info(
        "Parsed %d metric set pair(s): %s | mode=%s model=%s",
        len(parsed.metricSetPairList), pair_names, mode, settings.model,
    )

    raw_pairs = payload.get("metricSetPairList") or []
    if mode == "hybrid":
        # genuine NetflixACAJudge (via Kayenta) merged with the AI verdict.
        result = judge_hybrid(parsed, raw_pairs, canary_config, payload.get("scoreThresholds") or {})
    elif mode in AI_MODES:
        # configurable AI judge (summary | raw | plot); won't crash the analysis.
        # canary_config is passed so group names match the config (Referee needs this).
        result = judge_ai(settings, raw_pairs, canary_config)
    else:
        # dummy / ai (and any unknown mode) -> the no-op dummy judge (default).
        if mode not in DUMMY_MODES:
            log.warning("unknown mode '%s'; falling back to dummy judge", mode)
        result = judge(parsed)

    log.info(
        "Returning CanaryJudgeResult (mode=%s): score=%.1f classification=%s",
        mode,
        result.score.score,
        result.score.classification,
    )
    # by_alias not needed; field names already match Kayenta's JSON.
    return JSONResponse(content=result.model_dump())
