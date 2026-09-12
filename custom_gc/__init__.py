"""
Exposes a single shared Heap through module-level functions shaped like
the real `gc` module's API (get_count, get_stats, get_objects, collect,
set_threshold). This is what lets code written against the real `gc`
module (our Monitor, our Actuator) work against OUR collector with a
one-line import swap:

    import gc                  # before: real interpreter GC
    import custom_gc as gc     # after: our own mark-sweep collector

Everything downstream of that import -- analysis, agent, actuator,
dashboard -- keeps working unmodified, because they only ever call the
functions below, never touch Heap internals directly.
"""

from .heap import Heap

_heap = Heap()


def get_heap():
    """
    Escape hatch for workloads: they need real access to allocate()/
    add_reference()/add_root() to build the object graph. Monitor and
    Actuator never call this -- they only use the gc-shaped functions
    below.
    """
    return _heap


def get_count():
    return _heap.get_count()


def get_stats():
    return _heap.get_stats()


def get_objects():
    return _heap.get_objects()


def collect():
    return _heap.collect()


def set_threshold(n):
    # Our collector doesn't auto-trigger on a threshold (collection only
    # happens when collect() is explicitly called, e.g. by the actuator)
    # -- this exists purely so Actuator's tune_threshold action doesn't
    # crash when pointed at this module. Stored but currently unused.
    _heap.threshold = n


# Kept for interface parity with real gc.garbage. Our sweep just deletes
# unreachable objects outright rather than quarantining any of them, so
# this always stays empty -- there's no __del__-based uncollectable case
# in this simplified model.
garbage = []
