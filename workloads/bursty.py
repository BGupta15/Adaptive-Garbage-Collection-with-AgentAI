"""
Healthy but spiky: every 10th step allocates a large temporary and frees it
at once (refcount, no GC involved). Big sawtooth, no trend, no persistent
floor. A good agent leaves it alone; a naive "memory spiked -> collect"
agent over-reacts.
"""


def run(iterations=2000, burst_every=10, burst_items=20_000):
    for i in range(iterations):
        if i % burst_every == 0:
            tmp = [bytes(40) for _ in range(burst_items)]
            yield {"step": i, "burst": True}
            del tmp
        else:
            yield {"step": i, "burst": False}
