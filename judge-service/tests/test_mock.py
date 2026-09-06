"""Mock-first test for the configurable AI judge.

Drives the configurable AI judge with the model endpoint stubbed to return
canned JSON, then asserts a valid CanaryJudgeResult comes out. This exercises
the whole representation -> prompt -> parse -> map chain with no model required.

Run inside the judge-service container (it has the app deps):
    docker compose exec -T judge-service python - < judge-service/tests/test_mock.py
or via `make test-judge-mock`.
"""

import sys

from app import llm_judge
from app.judge_config import JudgeSettings

# Canned model output (what the gateway would return for a healthy canary judge).
CANNED = (
    '{"overallVerdict": "fail", "overallScore": 42, '
    '"metrics": [{"name": "dummy_latency", "classification": "high", "reason": "up"}, '
    '{"name": "dummy_error_rate", "classification": "high", "reason": "up"}], '
    '"rationale": "both metrics regressed"}'
)

PAIRS = [
    {"name": "dummy_latency", "id": "m1", "tags": {},
     "values": {"control": [100.0, 101.0, 99.0], "experiment": [130.0, 131.0, 129.0]}},
    {"name": "dummy_error_rate", "id": "m2", "tags": {},
     "values": {"control": [1.0, 1.1, 0.9], "experiment": [2.0, 2.1, 1.9]}},
]


def _settings(mode: str) -> JudgeSettings:
    return JudgeSettings(
        mode=mode, model="stub-model", modality="text",
        temperature=0.0, top_p=1.0, max_tokens=512, seed=42,
        litellm_base_url="http://stub",
    )


# _call_model returns (content, finish_reason, usage), so a stub has to return
# all three. Returning the content alone makes every call raise on unpacking, and
# the judge correctly converts that into an api_error result -- which is a
# well-formed object, so the test failed with a parse-looking message rather than
# the shape mismatch that caused it.
STUB_USAGE = {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}


def _stub(content: str, finish_reason: str = "stop"):
    return lambda settings, messages: (content, finish_reason, dict(STUB_USAGE))


def main() -> int:
    # Stub the only network call.
    llm_judge._call_model = _stub(CANNED)  # type: ignore

    failures = []
    for mode in ("summary", "raw"):
        result = llm_judge.judge_ai(_settings(mode), PAIRS)
        ok = (
            isinstance(result.score.score, (int, float))
            and abs(result.score.score - 42.0) < 1e-6
            and result.score.classification == "Fail"
            and len(result.results) == 2
            and all(r.classification == "High" for r in result.results)
            and all(r.controlMetadata.get("stats", {}).get("count") == 3 for r in result.results)
        )
        print(f"[mock] mode={mode}: score={result.score.score} verdict={result.score.classification} "
              f"metrics={[(r.name, r.classification) for r in result.results]} -> {'OK' if ok else 'FAIL'}")
        if not ok:
            failures.append(mode)

    # Also assert the tolerant path: garbage -> valid Error result (never crash).
    llm_judge._call_model = _stub("not json at all")  # type: ignore
    err = llm_judge.judge_ai(_settings("summary"), PAIRS)
    err_ok = err.score.classification == "Fail" and all(r.classification == "Error" for r in err.results)
    print(f"[mock] garbage-response -> {'OK (valid Error result)' if err_ok else 'FAIL'}")
    if not err_ok:
        failures.append("garbage")

    if failures:
        print(f"[mock] FAILURES: {failures}", file=sys.stderr)
        return 1
    print("[mock] OK: representation/parse/map verified with no model.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
