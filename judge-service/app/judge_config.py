"""Resolve judge configuration: representation (`mode`) and `model`.

Both are pure configuration, read from the canary config's
`judge.judgeConfigurations`, falling back to environment defaults. Nothing here
branches on provider; the only model attribute the judge cares about is its
`modality` (text vs vision), looked up in the model registry (models.yaml).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

log = logging.getLogger("judge-service.config")

# Modes that go through the AI path (everything else is dummy/hybrid, handled
# by the existing code).
AI_MODES = {"summary", "raw", "plot"}
DUMMY_MODES = {"dummy", "ai"}  # 'ai' kept as the existing no-op alias

_REGISTRY_PATH = Path(os.environ.get("MODELS_REGISTRY", "/app/models.yaml"))


@dataclass
class JudgeSettings:
    mode: str
    model: str
    modality: str          # "text" | "vision"
    temperature: float
    top_p: float
    max_tokens: int
    seed: Optional[int]
    litellm_base_url: str
    json_mode: bool = True  # send response_format={"type":"json_object"} (see json_mode_for)


def _load_registry() -> Dict[str, Dict[str, Any]]:
    try:
        with open(_REGISTRY_PATH) as f:
            data = yaml.safe_load(f) or {}
        return data.get("models", {}) or {}
    except FileNotFoundError:
        log.warning("model registry %s not found; defaulting all models to text", _REGISTRY_PATH)
        return {}


def modality_for(model: str) -> str:
    reg = _load_registry()
    entry = reg.get(model) or {}
    modality = str(entry.get("modality", "text")).lower()
    if modality not in ("text", "vision"):
        modality = "text"
    return modality


def json_mode_for(model: str) -> bool:
    """Whether to request response_format={"type":"json_object"}. Default True
    (unchanged behaviour for every existing model). Set `json_mode: false` in
    models.yaml for a model whose provider mistranslates OpenAI JSON-mode into a
    forced tool-call that drops the actual answer (observed with some very new
    Bedrock Claude releases via LiteLLM); the frozen prompt's own "STRICT JSON
    ONLY" instruction plus the tolerant parser handle it fine without JSON mode."""
    reg = _load_registry()
    entry = reg.get(model) or {}
    return bool(entry.get("json_mode", True))


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def resolve(canary_config: Dict[str, Any]) -> JudgeSettings:
    """Resolve effective settings from judgeConfigurations + env fallbacks."""
    judge_cfg = ((canary_config or {}).get("judge") or {}).get("judgeConfigurations") or {}

    mode = str(judge_cfg.get("mode") or os.environ.get("JUDGE_MODE", "dummy")).lower()
    model = str(judge_cfg.get("model") or os.environ.get("JUDGE_MODEL", "qwen-llm"))

    seed_raw = judge_cfg.get("seed", os.environ.get("JUDGE_SEED", "42"))
    try:
        seed: Optional[int] = int(seed_raw) if seed_raw not in (None, "") else None
    except (TypeError, ValueError):
        seed = None

    return JudgeSettings(
        mode=mode,
        model=model,
        modality=modality_for(model),
        json_mode=json_mode_for(model),
        temperature=_env_float("JUDGE_TEMPERATURE", 0.0),
        top_p=_env_float("JUDGE_TOP_P", 1.0),
        max_tokens=int(_env_float("JUDGE_MAX_TOKENS", 1024)),
        seed=seed,
        litellm_base_url=os.environ.get("LITELLM_BASE_URL", "http://litellm:4000"),
    )
