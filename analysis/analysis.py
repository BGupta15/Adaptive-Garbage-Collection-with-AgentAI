"""
Turns raw monitor samples into the signals the agent reasons over. Only
"observation"-tagged samples are used here -- "action_pre"/"action_post"
samples are deliberately excluded so an intervention doesn't get misread
as organic workload behavior.
"""

import statistics


def _observations(samples):
    return [s for s in samples if s["tag"] == "observation"]


def compute_growth_rate(samples, window=10):
    """
    Bytes/sec change in tracemalloc-traced memory over the last `window`
    observation samples. Positive = growing, negative = shrinking, ~0 =
    stable. Answers: "is Python-level allocated memory increasing?"
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


def compute_rss_growth_rate(samples, window=10):
    """
    Bytes/sec change in OS-level RSS (resident set size) over the last
    `window` samples. Answers a DIFFERENT question than traced growth:
    "is the process's actual memory footprint growing, as the OS sees
    it?" tracemalloc only sees Python-level allocations; RSS also
    reflects things like memory fragmentation and C-extension
    allocations. A workload where traced growth is flat but RSS keeps
    climbing points at something tracemalloc can't see.
    """
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
    """
    How many generation-0 collections happened across the last `window`
    observation samples. Answers: "how much allocation pressure is there?"
    -- gen0 is the fastest-cycling generation, so a rising delta here is
    a reasonable proxy for allocation pressure, independent of whether
    that pressure is a problem or not.

    IMPORTANT: this uses gc_stats[0]["collections"] (from gc.get_stats()),
    NOT gc.get_count()[0]. get_count() is a live allocation counter that
    RESETS toward zero every time a collection fires -- subtracting it
    across a window can go negative if a collection happened mid-window
    (this was a real bug here: unbounded_cache produced a delta of -53
    before this fix). get_stats()[i]["collections"] is a cumulative,
    monotonically increasing counter and doesn't have that problem.
    """
    obs = _observations(samples)
    if len(obs) < 2:
        return 0

    recent = obs[-window:]
    gen0_start = recent[0]["gc_stats"][0]["collections"]
    gen0_end = recent[-1]["gc_stats"][0]["collections"]
    return gen0_end - gen0_start


def compute_object_count_delta(samples, window=10):
    """
    Change in total live object count over the last `window` samples.
    Answers: "is the number of distinct objects growing?" -- this is a
    signal independent of tracemalloc's byte accounting, so it can catch
    growth patterns (e.g. many small objects) that byte-based growth
    rate alone might understate.
    """
    obs = _observations(samples)
    if len(obs) < 2:
        return 0

    recent = obs[-window:]
    return recent[-1]["live_object_count"] - recent[0]["live_object_count"]


def compute_uncollectable_delta(samples, window=10):
    """
    Change in the count of "uncollectable" objects (objects the collector
    found in a cycle but could not free) over the last `window` samples.

    CAVEAT, confirmed by testing: in Python 3.4+ (PEP 442), objects with
    __del__ are generally collectable by the cyclic GC -- the old
    restriction that made __del__-cycles permanently uncollectable was
    removed. Running this against all three workloads here, this value
    stayed 0 the entire time for cycle_generator too, since its dict
    cycles have no __del__ or problematic finalizers. In practice this
    signal will rarely fire for ordinary Python objects; it mainly
    catches cycles involving C-extension types or specific weakref/
    finalizer edge cases. Kept here because it's a real, correct signal
    when it DOES fire (a true positive is a strong one) -- but don't
    rely on it as your primary cycle-detection signal, since it will
    likely read 0 for most demo workloads including this project's own.
    """
    obs = _observations(samples)
    if len(obs) < 2:
        return 0

    recent = obs[-window:]
    return recent[-1]["uncollectable_total"] - recent[0]["uncollectable_total"]


def compute_growth_volatility(samples, window=10):
    """
    Standard deviation of per-step traced-memory deltas over the last
    `window` samples. Answers: "is this a steady trend or noisy spikes?"
    -- a steady leak has low volatility relative to its growth rate; a
    bursty-but-healthy workload (e.g. a request handler with big
    temporary allocations) can show a similar average growth rate but
    much higher volatility. Useful for telling those two apart instead
    of treating every growth-rate spike the same way.
    """
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
