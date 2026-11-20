import tracemalloc

from actuator.actuator import Actuator
from workloads import slow_leak


def test_slow_leak_cleanup_releases_retained_references():
    slow_leak.reset()
    gen = slow_leak.run(iterations=20, leak_bytes=4096)
    for _ in range(20):
        next(gen)
    assert slow_leak.retained_count() == 20
    result = slow_leak.release_retained(ratio=0.5)
    assert result == 10
    assert slow_leak.retained_count() == 10


def test_actuator_uses_only_explicit_cleanup_adapter(tmp_path):
    slow_leak.reset()
    gen = slow_leak.run(iterations=10, leak_bytes=4096)
    for _ in range(10):
        next(gen)
    actuator = Actuator(log_path=str(tmp_path / "actions.jsonl"), workload_module=slow_leak)
    result = actuator.apply({"action": "release_retained", "release_ratio": 0.5})
    assert result["released_items"] == 5
    assert slow_leak.retained_count() == 5
