"""
In-process monitor: samples gc/tracemalloc/psutil state while a workload
generator runs, interleaved on the same thread.

Actuator actions are isolated with an explicit `action_in_progress` flag
so pre/post intervention samples are distinctly tagged and do not contaminate
the regular observation window.
"""

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
        self.gc_module = gc_module if gc_module is not None else _real_gc

        if not tracemalloc.is_tracing():
            tracemalloc.start()

    def sample(self, tag="observation"):
        current, peak = tracemalloc.get_traced_memory()
        gc_stats = self.gc_module.get_stats()
        uncollectable_total = sum(g.get("uncollectable", 0) for g in gc_stats)

        snapshot = {
            "timestamp": time.time(),
            "tag": tag,
            "gc_counts": self.gc_module.get_count(),
            "gc_stats": gc_stats,
            "traced_current_bytes": current,
            "traced_peak_bytes": peak,
            "rss_bytes": self.process.memory_info().rss,
            "live_object_count": len(self.gc_module.get_objects()),
            "uncollectable_total": uncollectable_total,
        }
        self.samples.append(snapshot)
        return snapshot

    def run(self, workload_gen, agent_decide_fn, actuator, max_steps=200, workload_module=None, workload_name=None):
        """
        Runs the interleaved monitor-agent-actuator loop.
        """
        for _workload_step_info in workload_gen:
            if self.action_in_progress:
                continue

            self.sample(tag="observation")

            try:
                decision = agent_decide_fn(self.samples, workload_name=workload_name)
            except TypeError:
                decision = agent_decide_fn(self.samples)

            if decision.get("action") != "noop":
                self._apply_action(decision, actuator, workload_module=workload_module)

            time.sleep(self.sample_interval)

            if len(self.samples) >= max_steps:
                break

    def _apply_action(self, decision, actuator, workload_module=None):
        self.action_in_progress = True
        try:
            pre = self.sample(tag="action_pre")
            result = actuator.apply(decision, workload_module=workload_module)
            post = self.sample(tag="action_post")
            actuator.log_event(decision, pre, post, result)

            # Give feedback to the agent so it learns the real impact of its intervention
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
