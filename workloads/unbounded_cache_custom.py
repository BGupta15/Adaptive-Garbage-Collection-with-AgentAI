"""
A cache container that is a root (so it's always reachable), holding a
growing list of entries via references. No cycles here -- entries are
genuinely live, reachable, and referenced the whole time.

This is the contrast case against cycle_generator_custom.py:
growth without a cycle, so our collector legitimately CANNOT and SHOULD NOT
free these entries unless an eviction is authorized to drop their root reference.
"""

from custom_gc import get_heap

_cache_root = None


def evict(ratio=0.5):
    """
    Evicts the oldest `ratio` proportion of entries from the active custom heap cache root.
    Returns the number of evicted references.
    """
    global _cache_root
    if not _cache_root or not _cache_root.references:
        return 0
    to_remove_count = max(1, int(len(_cache_root.references) * ratio))
    to_remove = list(_cache_root.references[:to_remove_count])
    for ref in to_remove:
        _cache_root.references.remove(ref)
    return len(to_remove)


def run(iterations=2000, growth_per_step=50):
    global _cache_root
    heap = get_heap()
    _cache_root = heap.allocate("cache_root")
    heap.add_root(_cache_root)  # kept alive deliberately -- this is the "cache"

    for i in range(iterations):
        for _ in range(growth_per_step):
            entry = heap.allocate("cache_entry")
            heap.add_reference(_cache_root, entry)  # reachable via cache_root -> entry

        yield {
            "step": i,
            "cache_entries": len(_cache_root.references),
            "heap_live_objects": len(heap.objects),
        }


if __name__ == "__main__":
    for step_info in run(iterations=5, growth_per_step=20):
        print(step_info)
