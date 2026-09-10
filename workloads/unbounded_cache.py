"""
A dict that grows forever with no eviction policy -- the simplest kind of
leak. No cycles involved; this is pure unbounded retention, useful as a
contrast case against cycle_generator.py.
"""

import random


def run(iterations=2000, growth_per_step=100, value_size_range=(50, 200)):
    """
    iterations: how many sampling windows this workload runs for
    growth_per_step: how many new keys get added per window
    value_size_range: (min, max) length of the random string stored per key

    Everything stored here is intentionally kept alive for the whole run --
    that's the point. Nothing is ever deleted from `cache`.
    """
    cache = {}
    key_counter = 0

    for i in range(iterations):
        for _ in range(growth_per_step):
            key_counter += 1
            size = random.randint(*value_size_range)
            cache[key_counter] = "x" * size

        yield {
            "step": i,
            "cache_entries": len(cache),
        }


if __name__ == "__main__":
    for step_info in run(iterations=5, growth_per_step=20):
        print(step_info)
