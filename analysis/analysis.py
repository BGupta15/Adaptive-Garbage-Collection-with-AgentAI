"""
Turns raw monitor samples into the signals the agent actually reasons
over: growth rate and GC frequency. Only "observation"-tagged samples are
used here -- "action_pre"/"action_post" samples are deliberately excluded
so an intervention doesn't get misread as organic workload behavior.
"""


def _observations(samples):
    return [s for s in samples if s["tag"] == "observation"]


def compute_growth_rate(samples, window=10):
    """
    Bytes/sec change in traced memory over the last `window` observation
    samples. Positive = growing, negative = shrinking, ~0 = stable.
    """
    obs = _observations(samples)
    if len(obs) < 2:
        return 0.0

    recent = obs[-window:]
    start_bytes = recent[0]["traced_current_bytes"]
    end_bytes = recent[-1]["traced_current_bytes"]
    elapsed = recent[-1]["timestamp"] - recent[0]["timestamp"]

    if elapsed <= 0:
        return 0.0

    return (end_bytes - start_bytes) / elapsed


def compute_gc_frequency(samples, window=10):
    """
    How many generation-0 collections happened across the last `window`
    observation samples. gen0 is the fastest-cycling generation, so a
    rising delta here is a reasonable proxy for allocation pressure.
    """
    obs = _observations(samples)
    if len(obs) < 2:
        return 0

    recent = obs[-window:]
    gen0_start = recent[0]["gc_counts"][0]
    gen0_end = recent[-1]["gc_counts"][0]
    return gen0_end - gen0_start


def analyze(samples, window=10):
    return {
        "growth_rate_bytes_per_sec": compute_growth_rate(samples, window),
        "gc_gen0_delta": compute_gc_frequency(samples, window),
    }
