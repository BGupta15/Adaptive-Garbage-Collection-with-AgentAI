import gc as _real_gc

# Disable CPython's own cyclic collector so it can't silently clean up
# our simulated heap's cycles behind our back. This is what makes "our
# collector froze N objects" an honest, attributable number.
_real_gc.disable()


class GCObject:
    """A single object living in our simulated heap."""

    _next_id = 0

    def __init__(self, name=None):
        self.id = GCObject._next_id
        GCObject._next_id += 1
        self.name = name or f"obj{self.id}"
        self.references = []   # other GCObjects this one points to
        self.marked = False    # scratch flag used during mark phase


class Heap:
    """
    Our own heap: tracks every allocated object, a set of root object
    ids, and runs mark-sweep collection on demand.
    """

    def __init__(self):
        self.objects = {}          # id -> GCObject, every currently-live object
        self.roots = set()         # ids treated as reachable from "outside"
        self.alloc_count_since_collect = 0
        self.stats = {"collections": 0, "collected": 0, "uncollectable": 0}

    # --- building the object graph -------------------------------------

    def allocate(self, name=None):
        obj = GCObject(name)
        self.objects[obj.id] = obj
        self.alloc_count_since_collect += 1
        return obj

    def add_reference(self, from_obj, to_obj):
        from_obj.references.append(to_obj)

    def remove_reference(self, from_obj, to_obj):
        from_obj.references.remove(to_obj)

    def add_root(self, obj):
        self.roots.add(obj.id)

    def remove_root(self, obj):
        self.roots.discard(obj.id)

    # --- the actual algorithm --------------------------------------------

    def _mark(self):
        for obj in self.objects.values():
            obj.marked = False

        # Iterative DFS from roots (avoids Python recursion-depth limits
        # on deep or cyclic reference graphs).
        stack = [self.objects[rid] for rid in self.roots if rid in self.objects]
        while stack:
            obj = stack.pop()
            if obj.marked:
                continue
            obj.marked = True
            for ref in obj.references:
                if not ref.marked:
                    stack.append(ref)

    def _sweep(self):
        unmarked_ids = [oid for oid, obj in self.objects.items() if not obj.marked]
        for oid in unmarked_ids:
            del self.objects[oid]
        return len(unmarked_ids)

    def collect(self):
        """Run one full mark-sweep pass. Returns count of objects freed."""
        self._mark()
        freed = self._sweep()
        self.stats["collections"] += 1
        self.stats["collected"] += freed
        self.alloc_count_since_collect = 0
        return freed

    def reset(self):
        """
        Clears the heap entirely. Used between independent experiment
        runs (see main_custom_gc.py) so one workload's leftover objects
        don't confound the next workload's numbers -- without this, a
        fresh workload would start counting against a heap that already
        has unrelated objects sitting on it from the previous run.
        """
        self.objects.clear()
        self.roots.clear()
        self.alloc_count_since_collect = 0
        self.stats = {"collections": 0, "collected": 0, "uncollectable": 0}

    # --- stats surface, shaped to match the real `gc` module -------------
    # This shape-matching is what lets Monitor swap `gc` for this module
    # with a one-line change instead of rewriting monitor.py.

    def get_count(self):
        # Real gc.get_count() returns (gen0, gen1, gen2) allocation
        # counters. This is a single-generation collector, so gen1/gen2
        # are always 0 -- kept as a 3-tuple purely for interface parity.
        return (self.alloc_count_since_collect, 0, 0)

    def get_stats(self):
        # Real gc.get_stats() returns a list of 3 per-generation dicts
        # with "collections"/"collected"/"uncollectable" keys. Same
        # parity reasoning as get_count().
        empty_gen = {"collections": 0, "collected": 0, "uncollectable": 0}
        return [dict(self.stats), dict(empty_gen), dict(empty_gen)]

    def get_objects(self):
        return list(self.objects.values())
