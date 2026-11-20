import statistics
import tracemalloc


def _obs(samples, window):
    return [s for s in samples if s["tag"] == "observation"][-window:]


def _ols(xs, ys):
    """
    Ordinary least squares fit of y on x -> (slope, r_squared).
    statistics.linear_regression (Python 3.10+).
    """
    if len(xs) < 3 or xs[-1] - xs[0] <= 0:
        return 0.0, 0.0
    try:
        slope = statistics.linear_regression(xs, ys).slope
        r = statistics.correlation(xs, ys)
    except statistics.StatisticsError:
        return 0.0, 0.0
    return slope, r * r


def _floor_slope(ts, ys, k=5):
    """
    Slope of the "min envelope": the minimum over each sliding sub-window
    of k points. Ignores the sawtooth peaks and tracks only the low points.
    """
    if len(ys) < k + 3:
        return 0.0
    mins = [min(ys[i:i + k]) for i in range(len(ys) - k + 1)]
    slope, _ = _ols(ts[k - 1:], mins)
    return slope


def window_features(samples, window=20):
    obs = _obs(samples, window)
    if len(obs) < 5:
        return {"n_samples": len(obs), "enough_data": False}

    t0 = obs[0]["timestamp"]
    ts = [s["timestamp"] - t0 for s in obs]
    traced = [s["traced_current_bytes"] for s in obs]
    rss = [s["rss_bytes"] for s in obs]
    objs = [s["live_object_count"] for s in obs]

    slope, r2 = _ols(ts, traced)
    floor = _floor_slope(ts, traced)
    rss_slope, _ = _ols(ts, rss)
    obj_slope, _ = _ols(ts, objs)

    # Objects the collector itself reclaimed during this window
    gc_collected = (sum(g["collected"] for g in obs[-1]["gc_stats"])
                    - sum(g["collected"] for g in obs[0]["gc_stats"]))

    deltas = [b - a for a, b in zip(traced, traced[1:])]
    mean_delta = statistics.fmean(deltas)
    vol = statistics.pstdev(deltas)
    mean_level = statistics.fmean(traced)

    return {
        "enough_data": True,
        "n_samples": len(obs),
        "window_seconds": round(ts[-1], 3),
        "traced_slope_bps": round(slope, 1),
        "window_growth_pct": round(100 * slope * ts[-1] / mean_level, 1) if mean_level else 0.0,
        "trend_r2": round(r2, 3),
        "floor_ratio": round(floor / slope, 3) if slope > 0 else 0.0,
        "rss_slope_bps": round(rss_slope, 1),
        "object_slope_per_s": round(obj_slope, 1),
        "volatility_cv": round(vol / (abs(mean_delta) + 1e-9), 2),
        "sawtooth_amplitude_pct": round(100 * (max(traced) - min(traced)) / mean_level, 1) if mean_level else 0.0,
        "gc_collected_in_window": gc_collected,
        "current_traced_bytes": traced[-1],
    }


def action_history(samples, last_n=5):
    """
    Pair each action_pre with its action_post and compute what the action
    actually achieved. This is the feedback signal.
    """
    history = []
    for a, b in zip(samples, samples[1:]):
        if a["tag"] == "action_pre" and b["tag"] == "action_post":
            freed = a["traced_current_bytes"] - b["traced_current_bytes"]
            history.append({
                "t_since_pre_s": round(b["timestamp"] - a["timestamp"], 4),
                "bytes_before": a["traced_current_bytes"],
                "bytes_after": b["traced_current_bytes"],
                "bytes_freed": freed,
                "reclaim_ratio": round(freed / a["traced_current_bytes"], 3)
                                 if a["traced_current_bytes"] else 0.0,
                "objects_freed": a["live_object_count"] - b["live_object_count"],
            })
    return history[-last_n:]


def futile_streak(history, min_ratio=0.05):
    """Number of most-recent consecutive actions that reclaimed < min_ratio."""
    streak = 0
    for h in reversed(history):
        if h["reclaim_ratio"] < min_ratio:
            streak += 1
        else:
            break
    return streak


def top_allocation_sites(n=3):
    """
    Where the live traced memory sits, by source line. Gives the LLM evidence
    about WHERE memory is held. Only call when about to consult the LLM.
    """
    if not tracemalloc.is_tracing():
        return []
    snap = tracemalloc.take_snapshot()
    snap = snap.filter_traces((
        tracemalloc.Filter(False, tracemalloc.__file__),
        tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
    ))
    out = []
    for stat in snap.statistics("lineno")[:n]:
        frame = stat.traceback[0]
        out.append({"where": f"{frame.filename.split('/')[-1]}:{frame.lineno}",
                    "kb": round(stat.size / 1024, 1), "blocks": stat.count})
    return out


def suspicion_score(f):
    """
    Cheap 0..1 gate: "is there anything here worth an LLM call?"
    Deliberately NOT a decision -- only decides whether to ask the LLM.
    """
    if not f.get("enough_data"):
        return 0.0

    trend = 0.0
    # Do not require 5% growth in a short rolling window: a real slow leak
    # can be highly linear and persistent while adding less than 5%.
    # R2 + rising floor distinguish this from short-lived burst spikes.
    if (f["traced_slope_bps"] > 1_000
            and f["trend_r2"] >= 0.85
            and f["floor_ratio"] >= 0.60
            and f["window_growth_pct"] >= 0.25):
        persistence = max(0.0, min(1.0, f["floor_ratio"]))
        trend = f["trend_r2"] * (0.5 + 0.5 * persistence)

    pressure = 0.6 * min(1.0, f["sawtooth_amplitude_pct"] / 100)
    if f["sawtooth_amplitude_pct"] < 20:
        pressure = 0.0

    return round(max(trend, pressure), 3)
