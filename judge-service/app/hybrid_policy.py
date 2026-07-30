"""The hybrid policy: a pure, dependency-free decision function.

Kept apart from hybrid.py (which pulls in FastAPI/pydantic/Kayenta) so that both
the in-service hybrid judge and the host-side experiment runner
(tools/run_experiment.py) import the same policy code, which guarantees the
scored hybrid matches what the service would return. Pure stdlib.

Three policies, selectable by
`judgeConfigurations.hybridPolicy = gated | or | and` (default gated):
  - gated: trust statistical FAILs (high precision); on a clean statistical PASS
    escalate to the AI only when it fails with high confidence or cites a blind-
    spot dimension (variance/tail/trend); in the statistical marginal/near-
    threshold regime defer to the AI. This is the recommended hybrid.
  - or:  fail if either judge fails (safety net, high recall, inherits AI FPs).
  - and: fail only if both fail (strict, high precision, low recall).
"""

from __future__ import annotations

from dataclasses import dataclass

PASS_OK = {"Pass", "Nodata"}

DEFAULT_POLICY = "gated"
# How far above the pass threshold the statistical score must sit (with no flagged
# metric) to count as a clear PASS the gated policy won't override.
MARGINAL_BAND = 10.0
# An AI FAIL is "high confidence" if its score is at or below this.
AI_FAIL_CONF_MAX = 50.0

# Blind-spot dimensions the rank test is structurally insensitive to. If the AI's
# per-metric reasoning/rationale cites one and the AI fails, the gated hybrid
# trusts the AI over a clean statistical PASS.
#
# 30 substrings (11 spread + 8 tail + 11 temporal), of which 29 are effective:
# "stddev" contains "std", so any text matching "stddev" already matched "std"
# and the entry can never fire on its own. It is the only such superstring.
# Both counts are recorded because the tuple is frozen at the published values
# and a reader comparing it against the paper needs to know which is which.
BLINDSPOT_KEYWORDS = (
    "varian", "spread", "std", "stddev", "deviation", "instab", "unstable",
    "flap", "noisy", "volatil", "erratic",
    "tail", "p90", "p95", "p99", "percentile", "spike", "outlier", "max ",
    "trend", "slope", "drift", "ramp", "increasing", "growing", "climb",
    "leak", "over time", "progressiv", "upward",
)


def classify(score: float, pass_t: float, marginal_t: float) -> str:
    if score >= pass_t:
        return "Pass"
    if score >= marginal_t:
        return "Marginal"
    return "Fail"


def text_flags_blindspot(text: str) -> bool:
    t = (text or "").lower()
    return any(k in t for k in BLINDSPOT_KEYWORDS)


@dataclass
class PolicyDecision:
    verdict: str          # "Pass" | "Marginal" | "Fail"
    score: float          # threshold-consistent overall score (0-100)
    reason: str
    followed: str         # "statistical" | "ai" | "both"


def decide_policy(
    policy: str,
    d_score: float,
    d_verdict: str,
    d_has_nonpass_metric: bool,
    a_score: float,
    a_verdict: str,
    a_blindspot: bool,
    pass_t: float,
    marginal_t: float,
    marginal_band: float = MARGINAL_BAND,
    ai_fail_conf_max: float = AI_FAIL_CONF_MAX,
    recovered_transient: bool = False,
    equal_variance_noise: bool = False,
) -> PolicyDecision:
    """Combine the statistical (d_*) and AI (a_*) verdicts under `policy`.

    Verdicts are "Pass"|"Marginal"|"Fail". The returned `score` is set so its
    threshold mapping agrees with the verdict, because when the hybrid runs
    through Kayenta the standalone aggregation re-derives promote/rollback from
    the score against the thresholds.

    `recovered_transient` and `equal_variance_noise` both default False, so an
    existing caller's behaviour is unchanged unless it opts in. They come from
    the raw control/experiment series via
    series_guards.detect_recovered_transient / detect_equal_variance_noise,
    the same "healed_transient"/"noise_equivalent" shape detectors the
    statistical-ensemble judge uses. When either is True, the gated policy's two
    AI-driven FAIL-escalation branches are suppressed: a benign
    transient-then-recovered or equal-variance-noise shape shouldn't be
    overridden into a FAIL just because the AI's rationale sounds confident or
    cites a blind-spot keyword. This only touches the `gated` policy's escalation
    paths (branches 2 and 3 below), not the `d_fail` branch (the statistical
    judge already passes both these families, so there's nothing to suppress),
    and `or`/`and` are left alone (out of scope for these guards).
    """
    policy = (policy or DEFAULT_POLICY).lower()
    d_fail = d_verdict == "Fail"
    a_fail = a_verdict == "Fail"
    fp_guard = recovered_transient or equal_variance_noise

    # OR: safety net, fail if either judge fails (high recall)
    if policy == "or":
        if d_fail or a_fail:
            score = min(d_score, a_score)
            return PolicyDecision("Fail", min(score, marginal_t - 0.01),
                                  f"OR: statistical={d_verdict}({d_score:.0f}), ai={a_verdict}({a_score:.0f}) -> FAIL",
                                  "statistical" if d_fail and not a_fail else ("ai" if a_fail and not d_fail else "both"))
        score = max(d_score, a_score)
        return PolicyDecision(classify(score, pass_t, marginal_t), score,
                              f"OR: neither failed (statistical={d_score:.0f}, ai={a_score:.0f})", "both")

    # AND: strict, fail only if both fail (high precision)
    if policy == "and":
        if d_fail and a_fail:
            score = max(d_score, a_score)
            return PolicyDecision("Fail", min(score, marginal_t - 0.01),
                                  f"AND: both failed (statistical={d_score:.0f}, ai={a_score:.0f}) -> FAIL", "both")
        score = max(d_score, a_score)
        return PolicyDecision(classify(score, pass_t, marginal_t), max(score, pass_t),
                              f"AND: not both failed (statistical={d_verdict}, ai={a_verdict})",
                              "statistical" if not d_fail else "ai")

    # GATED (default, the recommended hybrid)
    # 1) Statistical FAIL is high-precision: trust it.
    if d_fail:
        return PolicyDecision("Fail", d_score,
                              f"GATED: statistical FAIL ({d_score:.0f}) is high-precision; trust it", "statistical")

    # 2) Clear statistical PASS (well above threshold, no flagged metric):
    #    override to FAIL only if the AI fails with high confidence or cites a
    #    blind-spot dimension (variance/tail/trend).
    clear_pass = (d_score >= pass_t + marginal_band) and not d_has_nonpass_metric
    if clear_pass:
        if a_fail and (a_score <= ai_fail_conf_max or a_blindspot) and not fp_guard:
            why = "high-confidence" if a_score <= ai_fail_conf_max else "blind-spot"
            return PolicyDecision("Fail", min(d_score, a_score),
                                  f"GATED: clean statistical PASS but AI {why} FAIL -> override to FAIL", "ai")
        guard_note = " (FP guard suppressed an AI override)" if (a_fail and fp_guard) else ""
        return PolicyDecision("Pass", d_score,
                              f"GATED: clean statistical PASS ({d_score:.0f}); AI did not override{guard_note}",
                              "statistical")

    # 3) Statistical marginal / near-threshold: its least reliable regime, defer to AI.
    if (a_fail or a_verdict == "Marginal") and not fp_guard:
        return PolicyDecision("Fail", min(d_score, a_score),
                              f"GATED: statistical marginal ({d_score:.0f}); defer to AI {a_verdict} -> FAIL", "ai")
    guard_note = " (FP guard suppressed deferring to AI)" if (fp_guard and (a_fail or a_verdict == "Marginal")) else ""
    return PolicyDecision("Pass", max(d_score, a_score, pass_t),
                          f"GATED: statistical marginal ({d_score:.0f}); AI PASS breaks the tie{guard_note}", "ai")
