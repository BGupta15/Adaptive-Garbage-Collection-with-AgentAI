"""
Rule-based stand-in for the future LangGraph + LLM decision layer.

This exists so the full monitor -> analyze -> decide -> act -> evaluate
loop runs end-to-end today, without needing an API key yet. Swapping this
for a real LLM agent later means writing a new module with the SAME
decide(samples, analysis_fn) signature and same returned schema:

    {"action": "collect" | "tune_threshold" | "flag_cycle" | "noop",
     "reason": "<short string, from the LLM or from a rule>"}

Nothing else in the project needs to change to make that swap -- main.py
just imports a different decide function.
"""

import time

# Tune these against what you actually observe from your synthetic
# workloads -- these starting numbers are guesses, not measured values.
GROWTH_THRESHOLD_BYTES_PER_SEC = 50_000
MIN_SECONDS_BETWEEN_ACTIONS = 5

_state = {"last_action_time": 0.0}


def decide(samples, analysis_fn):
    analysis = analysis_fn(samples)
    now = time.time()

    # Cooldown safety rail: this is what stops the agent from thrashing
    # (triggering collect() every single sample if growth briefly spikes).
    if now - _state["last_action_time"] < MIN_SECONDS_BETWEEN_ACTIONS:
        return {"action": "noop", "reason": "cooldown active"}

    growth = analysis["growth_rate_bytes_per_sec"]

    if growth > GROWTH_THRESHOLD_BYTES_PER_SEC:
        _state["last_action_time"] = now
        return {
            "action": "collect",
            "reason": f"growth rate {growth:.0f} B/s exceeds threshold "
                      f"{GROWTH_THRESHOLD_BYTES_PER_SEC} B/s",
        }

    return {"action": "noop", "reason": "growth within normal range"}
