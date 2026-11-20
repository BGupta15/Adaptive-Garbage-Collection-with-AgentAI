"""
Synthetic referenced-memory leak used to test detection and safe remediation.

The retained objects are intentionally reachable, so gc.collect() alone cannot
free them. The workload exposes release_retained() as an application-level
cleanup adapter; real applications must provide their own domain-specific hook.
"""

_retained = []


def reset():
    """Clear state before an independent benchmark run."""
    _retained.clear()


def release_retained(ratio=0.5):
    """Release a bounded fraction of retained references; return item count."""
    ratio = max(0.10, min(0.75, float(ratio)))
    count = int(len(_retained) * ratio)
    if count <= 0 and _retained:
        count = 1
    if count:
        del _retained[:count]
    return count


def retained_count():
    """Expose a read-only count for tests and benchmark diagnostics."""
    return len(_retained)


def run(iterations=2000, leak_bytes=1500):
    """Retain roughly ``leak_bytes`` per iteration until cleanup is requested."""
    reset()
    for i in range(iterations):
        _retained.append(bytearray(leak_bytes))
        yield {"step": i, "retained_items": len(_retained)}
