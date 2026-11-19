from analysis.stats import window_features, suspicion_score, action_history, futile_streak


def mk(traced, t0=0.0, dt=0.1, tag="observation", objs=None):
    return [{"tag": tag, "timestamp": t0 + i * dt, "traced_current_bytes": b, "rss_bytes": 1,
             "live_object_count": (objs[i] if objs else 0), "uncollectable_total": 0,
             "traced_peak_bytes": b, "gc_counts": (0, 0, 0),
             "gc_stats": [{"collected": 0, "collections": 0, "uncollectable": 0}] * 3}
            for i, b in enumerate(traced)]


def test_steady_leak_is_flagged():
    f = window_features(mk([100_000 + 20_000 * i for i in range(30)]))
    assert f["trend_r2"] > 0.99 and f["floor_ratio"] > 0.9
    assert suspicion_score(f) > 0.9


def test_flat_series_is_not_suspicious():
    assert suspicion_score(window_features(mk([500_000] * 30))) == 0.0


def test_sawtooth_without_trend_passes_gate_via_pressure():
    saw = [100_000 + (80_000 if i % 5 == 0 else 0) for i in range(30)]
    f = window_features(mk(saw))
    assert f["sawtooth_amplitude_pct"] > 20 and suspicion_score(f) >= 0.35


def test_too_few_samples():
    assert window_features(mk([1, 2, 3]))["enough_data"] is False


def test_action_history_and_futility():
    pre = {"tag": "action_pre", "timestamp": 1.0, "traced_current_bytes": 1000, "live_object_count": 50}
    post = {"tag": "action_post", "timestamp": 1.1, "traced_current_bytes": 990, "live_object_count": 50}
    h = action_history([pre, post, pre, post])
    assert len(h) == 2 and h[0]["reclaim_ratio"] == 0.01
    assert futile_streak(h) == 2
