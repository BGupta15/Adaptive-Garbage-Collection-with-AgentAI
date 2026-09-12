"""
A cache container that is a root (so it's always reachable), holding a
growing list of entries via references. No cycles here -- entries are
genuinely live, reachable, and referenced the whole time. This is the
contrast case against cycle_generator_custom.py: growth without a cycle,
so our collector legitimately CANNOT and SHOULD NOT free these entries.
"""

from custom_gc import get_heap


def run(iterations=2000, growth_per_step=50):
    heap = get_heap()
    cache_root = heap.allocate("cache_root")
    heap.add_root(cache_root)  # kept alive deliberately -- this is the "cache"

    for i in range(iterations):
        for _ in range(growth_per_step):
            entry = heap.allocate("cache_entry")
            heap.add_reference(cache_root, entry)  # reachable via cache_root -> entry

        yield {
            "step": i,
            "cache_entries": len(cache_root.references),
            "heap_live_objects": len(heap.objects),
        }


if __name__ == "__main__":
    for step_info in run(iterations=5, growth_per_step=20):
        print(step_info)
