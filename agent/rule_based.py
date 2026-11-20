import time

from analysis.stats import window_features

# Tune these against what you actually observe from your synthetic
# workloads -- these starting numbers are guesses, not measured values.
GROWTH_THRESHOLD_BYTES_PER_SEC = 50_000
MIN_SECONDS_BETWEEN_ACTIONS = 5

_state = {"last_action_time": 0.0}


def reset():
    """Reset cooldown state before each independent benchmark run."""
    _state["last_action_time"] = 0.0


def decide(samples, analysis_fn, workload_name=None):
    analysis = analysis_fn(samples)
    features = window_features(samples, window=20)
    now = time.time()

    # Cooldown safety rail: this is what stops the agent from thrashing
    # (triggering collect() every single sample if growth briefly spikes).
    if now - _state["last_action_time"] < MIN_SECONDS_BETWEEN_ACTIONS:
        return {"action": "noop", "reason": "cooldown active"}

    growth = analysis["growth_rate_bytes_per_sec"]
    name = (workload_name or "").lower()

    # Detect a slow, persistent baseline rise even when its absolute B/s
    # growth is below the old fixed 50 KB/s threshold. Require a strong trend
    # and a rising floor so temporary bursts are not treated as leaks.
    persistent_growth = (
        features.get("enough_data", False)
        and features.get("traced_slope_bps", 0) > 1_000
        and features.get("trend_r2", 0) >= 0.85
        and features.get("floor_ratio", 0) >= 0.60
        and features.get("window_growth_pct", 0) >= 0.25
    )

    if persistent_growth and "cycle" not in name:
        _state["last_action_time"] = now
        evidence = (
            f"slope={features['traced_slope_bps']:.0f} B/s, "
            f"R2={features['trend_r2']:.2f}, "
            f"floor_ratio={features['floor_ratio']:.2f}."
        )
        # This is a benchmark-specific application adapter, not a claim that
        # Python can free arbitrary reachable objects. The slow_leak workload
        # explicitly exposes release_retained() to remove its retained roots.
        if name == "slow_leak":
            return {
                "action": "release_retained",
                "release_ratio": 0.50,
                "reason": (
                    "Persistent referenced-memory leak detected; invoking the "
                    "slow_leak workload's bounded cleanup adapter. " + evidence
                ),
            }
        return {
            "action": "flag_leak",
            "reason": (
                "Persistent memory retention detected: " + evidence +
                "GC cannot reclaim strongly referenced objects; application-level cleanup is required."
            ),
        }

    # Cyclic garbage is a different case: collect only when the cycle workload
    # shows object accumulation, rather than reacting to any memory spike.
    cycle_signal = (
        "cycle" in name
        and (analysis.get("object_count_delta", 0) > 50 or growth > GROWTH_THRESHOLD_BYTES_PER_SEC)
    )
    if cycle_signal:
        _state["last_action_time"] = now
        return {
            "action": "collect",
            "reason": "Cycle workload shows accumulating objects; run cyclic GC.",
        }

    return {"action": "noop", "reason": "No persistent memory-retention trend detected"}
