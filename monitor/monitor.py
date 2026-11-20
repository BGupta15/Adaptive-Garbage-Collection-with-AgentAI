import gc as _real_gc
import os
import time
import tracemalloc

import psutil


class Monitor:
    def __init__(self, sample_interval=0.3, gc_module=None):
        self.sample_interval = sample_interval
        self.action_in_progress = False
        self.samples = []
        self.process = psutil.Process(os.getpid())

        # Bytes of traced memory the monitor itself has retained by storing
        # samples. tracemalloc is process-wide, so without subtracting this,
        # EVERY workload -- even a healthy one -- shows steady fake "growth"
        # equal to the monitor's own bookkeeping (~16 KB/s at 100ms interval).
        self._self_overhead_bytes = 0

        self.gc_module = gc_module if gc_module is not None else _real_gc

        if not tracemalloc.is_tracing():
            tracemalloc.start()

    def sample(self, tag="observation"):
        """
        tag is one of: "observation", "action_pre", "action_post".
        """
        current, peak = tracemalloc.get_traced_memory()
        overhead_so_far = self._self_overhead_bytes
        gc_stats = self.gc_module.get_stats()

        uncollectable_total = sum(g.get("uncollectable", 0) for g in gc_stats)

        snapshot = {
            "timestamp": time.time(),
            "tag": tag,
            "gc_counts": self.gc_module.get_count(),
            "gc_stats": gc_stats,
            # Net of the monitor's own stored-sample overhead
            "traced_current_bytes": max(0, current - overhead_so_far),
            "traced_peak_bytes": max(0, peak - overhead_so_far),
            "rss_bytes": self.process.memory_info().rss,
            "live_object_count": len(self.gc_module.get_objects()),
            "uncollectable_total": uncollectable_total,
        }
        self.samples.append(snapshot)

        # Whatever tracemalloc grew by since the start of this call is
        # what storing this sample cost (transient temporaries are already
        # freed by now, so this is just the dict we appended).
        self._self_overhead_bytes += tracemalloc.get_traced_memory()[0] - current
        return snapshot

    def run(self, workload_gen, agent_decide_fn, actuator, max_steps=200,
            workload_module=None, workload_name=None):
        """
        workload_gen: a generator that yields periodically
        agent_decide_fn: callable(samples, **kwargs) -> decision dict
        actuator: an Actuator instance
        max_steps: safety cap so a run always terminates
        workload_module: passed through to actuator for evict actions
        workload_name: passed through to agent for context
        """
        for workload_step, _workload_step_info in enumerate(workload_gen, start=1):
            if self.action_in_progress:
                continue

            self.sample(tag="observation")
            # All agent implementations share this call contract. Avoid catching
            # TypeError here: it could hide a real bug raised inside an agent.
            decision = agent_decide_fn(self.samples, workload_name=workload_name)

            if decision.get("action") != "noop":
                self._apply_action(decision, actuator, workload_module=workload_module)

            time.sleep(self.sample_interval)

            # Count workload yields, not observation + action_pre + action_post
            # snapshots. Both agents now receive the same number of steps.
            if workload_step >= max_steps:
                break

    def _apply_action(self, decision, actuator, workload_module=None):
        """
        Isolation boundary: flag up, pre-snapshot, apply, post-snapshot, log, flag down.
        """
        self.action_in_progress = True
        try:
            pre = self.sample(tag="action_pre")
            result = actuator.apply(decision, workload_module=workload_module)
            post = self.sample(tag="action_post")
            actuator.log_event(decision, pre, post, result)

            # Feed outcome back to the agentic layer so it learns efficacy
            try:
                from agent.agentic_gc import get_agent
                agent = get_agent()
                agent.record_action_result(
                    action=decision.get("action"),
                    result=result,
                    pre_bytes=pre["traced_current_bytes"],
                    post_bytes=post["traced_current_bytes"],
                )
            except Exception:
                pass
        finally:
            self.action_in_progress = False
