"""
Continuously creates reference cycles (a <-> b, mutually referencing each
other) so they're unreachable by simple refcounting but ARE eventually
caught by Python's generational cycle collector.

Design note: this yields after every batch rather than running to
completion in one call, so the monitor gets a chance to sample mid-flight.
If this ran as one tight loop with no yield, the monitor would never see
anything until the workload was already finished.
"""

import gc


def run(iterations=2000, batch_size=200):
    """
    iterations: how many sampling windows this workload will run for
    batch_size: how many cycles are created per window

    Yields a small dict each step so callers/logs can see workload-side
    state (not just monitor-side state) if useful for debugging.
    """
    total_created = 0

    for i in range(iterations):
        batch = []
        for _ in range(batch_size):
            a = {}
            b = {}
            a["other"] = b
            b["other"] = a
            batch.append(a)
            total_created += 1

        # Deliberately drop all local references. 'a' and 'b' still
        # reference each other, so refcounting alone can't free them --
        # only the generational cycle collector can.
        del batch

        yield {
            "step": i,
            "cycles_created_total": total_created,
            # gc.garbage only fills up if collector was told to keep
            # unreachable objects with __del__; empty list here is normal.
            "gc_garbage_len": len(gc.garbage),
        }


if __name__ == "__main__":
    # Quick standalone sanity check -- run a few steps and print progress.
    for step_info in run(iterations=5, batch_size=50):
        print(step_info)
