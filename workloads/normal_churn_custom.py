"""
Baseline workload: each step allocates a batch of objects under a
temporary root, then removes that root -- making the whole batch
unreachable again. No cycles, no permanent retention. This is what
"healthy" behavior looks like on our own heap.
"""

from custom_gc import get_heap


def run(iterations=2000, batch_size=50):
    heap = get_heap()

    for i in range(iterations):
        batch_root = heap.allocate("batch_root")
        heap.add_root(batch_root)

        for _ in range(batch_size):
            entry = heap.allocate("churn_entry")
            heap.add_reference(batch_root, entry)

        # Drop reachability -- batch_root and everything under it becomes
        # garbage the moment the next collect() runs. No cycle involved.
        heap.remove_root(batch_root)

        yield {
            "step": i,
            "heap_live_objects": len(heap.objects),
        }


if __name__ == "__main__":
    for step_info in run(iterations=5, batch_size=20):
        print(step_info)
