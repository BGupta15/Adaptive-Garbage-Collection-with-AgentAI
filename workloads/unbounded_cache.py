"""
A dict that grows forever unless an eviction policy or agentic intervention
is triggered. Useful as a contrast case against cycle_generator.py:
here, objects are referenced and live, so standard cyclic GC cannot free them.
Only application-level eviction (authorized via Human-in-the-Loop) can reclaim memory.
"""

import random

_cache = {}


def evict(ratio=0.5):
    """
    Evicts the oldest `ratio` proportion of entries from the active cache.
    Returns the number of evicted items.
    """
    global _cache
    if not _cache:
        return 0
    to_remove = max(1, int(len(_cache) * ratio))
    keys = list(_cache.keys())[:to_remove]
    for k in keys:
        del _cache[k]
    return len(keys)


def get_cache_size():
    return len(_cache)


def run(iterations=2000, growth_per_step=100, value_size_range=(50, 200)):
    """
    iterations: how many sampling windows this workload runs for
    growth_per_step: how many new keys get added per window
    value_size_range: (min, max) length of the random string stored per key
    """
    global _cache
    _cache.clear()
    key_counter = 0

    for i in range(iterations):
        for _ in range(growth_per_step):
            key_counter += 1
            size = random.randint(*value_size_range)
            _cache[key_counter] = "x" * size

        yield {
            "step": i,
            "cache_entries": len(_cache),
        }


if __name__ == "__main__":
    for step_info in run(iterations=5, growth_per_step=20):
        print(step_info)
