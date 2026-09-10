"""
Control/baseline workload: objects are created and immediately dropped,
with no cycles and no unbounded retention. This is what "healthy" memory
behavior looks like -- without this, you can't prove the other two
workloads are actually anomalous rather than just noisy.
"""


def run(iterations=2000, batch_size=200):
    for i in range(iterations):
        # Plain objects, no cross-references -- refcounting alone frees
        # these the moment `batch` goes out of scope.
        batch = [object() for _ in range(batch_size)]
        del batch

        yield {
            "step": i,
            "objects_churned_total": (i + 1) * batch_size,
        }


if __name__ == "__main__":
    for step_info in run(iterations=5, batch_size=50):
        print(step_info)
