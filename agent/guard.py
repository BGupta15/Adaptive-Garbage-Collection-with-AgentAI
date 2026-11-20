ALLOWED_ACTIONS = {"collect", "evict_cache", "release_retained", "tune_threshold", "flag_cycle", "flag_leak", "noop"}
MUTATING = {"collect", "evict_cache", "release_retained", "tune_threshold"}  # change runtime/application state
HYPOTHESES = {"cycle_garbage", "unbounded_retention", "bursty", "healthy", "unknown"}

MAX_ACTIONS = 8          # mutating actions allowed per run
MIN_CONFIDENCE = 0.5
FUTILE_LIMIT = 2         # consecutive futile collects before we stop collecting


class DecisionError(ValueError):
    """Raised when raw model output does not match the schema."""


def validate(raw):
    """
    STRICT schema check on parsed-JSON model output. Returns a clean dict
    with only known keys (extra keys the model invents are dropped) or
    raises DecisionError. Strict on purpose: a malformed answer triggers a
    retry/fallback, which is safer than silently "fixing" it.
    """
    if not isinstance(raw, dict):
        raise DecisionError("response is not a JSON object")

    action = raw.get("action")
    if action not in ALLOWED_ACTIONS:
        raise DecisionError(f"unknown action: {action!r}")

    reason = raw.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        raise DecisionError("missing reason")

    # bool is a subclass of int in Python (True == 1), so it must be excluded
    # explicitly or {"confidence": true} would pass as 1.
    conf = raw.get("confidence")
    if isinstance(conf, bool) or not isinstance(conf, (int, float)) or not 0 <= conf <= 1:
        raise DecisionError("confidence must be a number in [0, 1]")

    hypothesis = raw.get("hypothesis", "unknown")
    if hypothesis not in HYPOTHESES:
        raise DecisionError(f"unknown hypothesis: {hypothesis!r}")

    clean = {"action": action, "reason": reason.strip()[:300],
             "hypothesis": hypothesis, "confidence": float(conf)}

    if action == "tune_threshold":
        t = raw.get("threshold")
        if isinstance(t, bool) or not isinstance(t, int):
            raise DecisionError("tune_threshold needs an integer 'threshold'")
        clean["threshold"] = t           # actuator clamps to its own bounds as well
    if action == "evict_cache":
        ratio = raw.get("evict_ratio", 0.25)
        if isinstance(ratio, bool) or not isinstance(ratio, (int, float)):
            raise DecisionError("evict_cache needs a numeric evict_ratio")
        if not 0.05 <= float(ratio) <= 0.50:
            raise DecisionError("evict_ratio must be between 0.05 and 0.50")
        clean["evict_ratio"] = float(ratio)
    if action == "release_retained":
        ratio = raw.get("release_ratio", 0.50)
        if isinstance(ratio, bool) or not isinstance(ratio, (int, float)):
            raise DecisionError("release_retained needs a numeric release_ratio")
        if not 0.10 <= float(ratio) <= 0.75:
            raise DecisionError("release_ratio must be between 0.10 and 0.75")
        clean["release_ratio"] = float(ratio)
    if isinstance(raw.get("recommendation"), str):
        clean["recommendation"] = raw["recommendation"].strip()[:300]
    return clean


def _downgrade(decision, new_action, why):
    out = dict(decision)
    out["proposed_action"] = decision["action"]
    out["action"] = new_action
    out["reason"] = f"[guard: {why}] {decision['reason']}"
    out["guard"] = why
    return out


def enforce_rails(decision, history, futile_streak, actions_taken):
    """
    A rail may only DOWNGRADE a decision (to noop or to a non-mutating
    flag), never upgrade it. Non-mutating actions always pass.
    `history` is accepted so rails can be extended without a signature change.
    """
    if decision["action"] not in MUTATING:
        return decision
    if actions_taken >= MAX_ACTIONS:
        return _downgrade(decision, "noop", "max actions per run reached")
    if decision["confidence"] < MIN_CONFIDENCE:
        return _downgrade(decision, "noop",
                          f"confidence {decision['confidence']:.2f} below {MIN_CONFIDENCE}")
    # Futility: repeated collects that freed ~nothing mean the memory is
    # referenced, not garbage. Stop collecting and report instead.
    if decision["action"] in {"collect", "release_retained"} and futile_streak >= FUTILE_LIMIT:
        return _downgrade(decision, "flag_leak",
                          f"last {futile_streak} reclaim actions recovered <5%; memory remains retained")
    return decision
