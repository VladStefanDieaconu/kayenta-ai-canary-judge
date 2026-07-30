"""Extract a verdict object from a model completion.

Split out of llm_judge because this is the one piece of the AI path that has to
be runnable without a network, a gateway or the OpenAI SDK: proving that a change
to the judge did not silently alter how an old response is read means replaying
thousands of archived completions through exactly this code (see
tools/replay_ai_logs.py). A parser that can only be exercised by making a call
cannot be checked against the archive at all.

Pure and dependency-free by design. Anything here that needs configuration,
credentials or a clock belongs in llm_judge instead.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Iterator, List, Optional

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def iter_balanced_objects(text: str) -> Iterator[str]:
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


def looks_like_verdict(obj: Any) -> bool:
    return isinstance(obj, dict) and (
        "overallVerdict" in obj or "overallScore" in obj or "metrics" in obj
    )


def parse_json(content: str) -> Optional[Dict[str, Any]]:
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
    for block in iter_balanced_objects(cleaned):
        try:
            obj = json.loads(block)
        except json.JSONDecodeError:
            continue
        if looks_like_verdict(obj):
            return obj
        if fallback is None and isinstance(obj, dict):
            fallback = obj
    return fallback


def classify_completion(content: str) -> str:
    """`empty_completion` | `parse_failure` | `parsed`.

    The three-way split the guard acts on. An empty completion and a completion
    that will not parse are both non-verdicts, but they have different causes --
    a token budget versus a model that will not follow the schema -- and
    collapsing them loses the only signal that distinguishes them.
    """
    if not (content or "").strip():
        return "empty_completion"
    return "parsed" if parse_json(content) is not None else "parse_failure"
