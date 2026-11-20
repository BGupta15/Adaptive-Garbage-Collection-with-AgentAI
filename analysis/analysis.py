import statistics


def _observations(samples):
    return [s for s in samples if s["tag"] == "observation"]


def compute_growth_rate(samples, window=10):
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


def compute_rss_growth_rate(samples, window=10):
    obs = _observations(samples)
    if len(obs) < 2:
        return 0.0

    recent = obs[-window:]
    start_rss = recent[0]["rss_bytes"]
    end_rss = recent[-1]["rss_bytes"]
    elapsed = recent[-1]["timestamp"] - recent[0]["timestamp"]

    if elapsed <= 0:
        return 0.0

    return (end_rss - start_rss) / elapsed


def compute_gc_frequency(samples, window=10):
    obs = _observations(samples)
    if len(obs) < 2:
        return 0

    recent = obs[-window:]
    gen0_start = recent[0]["gc_stats"][0]["collections"]
    gen0_end = recent[-1]["gc_stats"][0]["collections"]
    return gen0_end - gen0_start


def compute_object_count_delta(samples, window=10):
    obs = _observations(samples)
    if len(obs) < 2:
        return 0

    recent = obs[-window:]
    return recent[-1]["live_object_count"] - recent[0]["live_object_count"]


def compute_uncollectable_delta(samples, window=10):
    obs = _observations(samples)
    if len(obs) < 2:
        return 0

    recent = obs[-window:]
    return recent[-1]["uncollectable_total"] - recent[0]["uncollectable_total"]


def compute_growth_volatility(samples, window=10):
    obs = _observations(samples)
    if len(obs) < 3:
        return 0.0

    recent = obs[-window:]
    deltas = [
        recent[i + 1]["traced_current_bytes"] - recent[i]["traced_current_bytes"]
        for i in range(len(recent) - 1)
    ]
    if len(deltas) < 2:
        return 0.0

    return statistics.pstdev(deltas)


def analyze(samples, window=10):
    return {
        "growth_rate_bytes_per_sec": compute_growth_rate(samples, window),
        "rss_growth_rate_bytes_per_sec": compute_rss_growth_rate(samples, window),
        "gc_gen0_delta": compute_gc_frequency(samples, window),
        "object_count_delta": compute_object_count_delta(samples, window),
        "uncollectable_delta": compute_uncollectable_delta(samples, window),
        "growth_volatility": compute_growth_volatility(samples, window),
    }
