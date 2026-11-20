from .heap import Heap

_heap = Heap()


def get_heap():
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
