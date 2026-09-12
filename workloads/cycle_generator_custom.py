"""
Continuously creates reference cycles ON OUR OWN HEAP (custom_gc), not
using real Python dicts. Two GCObjects reference each other and neither
is ever added as a root -- unreachable from outside, only reachable from
each other. Only OUR mark-sweep collect() can free these.
"""

from custom_gc import get_heap


def run(iterations=2000, batch_size=50):
    heap = get_heap()
    total_created = 0

    for i in range(iterations):
        for _ in range(batch_size):
            a = heap.allocate("cycle_a")
            b = heap.allocate("cycle_b")
            heap.add_reference(a, b)
            heap.add_reference(b, a)
            total_created += 1
            # Deliberately NOT calling heap.add_root() on either -- these
            # are only reachable from each other, i.e. a classic cycle.

        yield {
            "step": i,
            "cycles_created_total": total_created,
            "heap_live_objects": len(heap.objects),
        }


if __name__ == "__main__":
    for step_info in run(iterations=5, batch_size=20):
        print(step_info)
