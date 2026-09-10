"""
In-process monitor: samples gc/tracemalloc/psutil state while a workload
generator runs, interleaved on the same thread.

Key design decision (see our earlier discussion): actuator actions are
isolated with an explicit `action_in_progress` flag rather than mixed
synchronously into the regular sampling loop. This is NOT a threading lock
-- there's no concurrency here, it's a single-threaded state guard whose
only job is to let us tag samples as "observation" vs "action_pre" /
"action_post" so before/after comparisons aren't contaminated by
regular sampling noise.
"""

import gc
import os
import time
import tracemalloc

import psutil


class Monitor:
    def __init__(self, sample_interval=0.3):
        self.sample_interval = sample_interval
        self.action_in_progress = False
        self.samples = []
        self.process = psutil.Process(os.getpid())

        # tracemalloc must be started before any snapshot calls work.
        if not tracemalloc.is_tracing():
            tracemalloc.start()

    def sample(self, tag="observation"):
        """
        tag is one of: "observation", "action_pre", "action_post".
        This tag is what lets the Evaluator later separate "the workload
        doing its normal thing" from "the actuator just did something."
        """
        current, peak = tracemalloc.get_traced_memory()

        snapshot = {
            "timestamp": time.time(),
            "tag": tag,
            "gc_counts": gc.get_count(),        # (gen0, gen1, gen2) collection counters
            "gc_stats": gc.get_stats(),          # per-generation collection stats
            "traced_current_bytes": current,     # tracemalloc: current traced allocations
            "traced_peak_bytes": peak,           # tracemalloc: peak since start/reset
            "rss_bytes": self.process.memory_info().rss,  # OS-level resident memory
        }
        self.samples.append(snapshot)
        return snapshot

    def run(self, workload_gen, agent_decide_fn, actuator, max_steps=200):
        """
        workload_gen: a generator (e.g. cycle_generator.run()) that yields
                      periodically so we get a turn to sample
        agent_decide_fn: callable(samples: list) -> decision dict, e.g.
                      {"action": "collect"|"tune_threshold"|"flag_cycle"|"noop",
                       "reason": "..."}
        actuator: an Actuator instance (applies decisions + logs them)
        max_steps: safety cap so a run always terminates
        """
        for _workload_step_info in workload_gen:
            # While an action is being applied, skip regular sampling --
            # the pre/post samples taken inside _apply_action already
            # cover this window, tagged distinctly.
            if self.action_in_progress:
                continue

            self.sample(tag="observation")

            decision = agent_decide_fn(self.samples)

            if decision.get("action") != "noop":
                self._apply_action(decision, actuator)

            time.sleep(self.sample_interval)

            if len(self.samples) >= max_steps:
                break

    def _apply_action(self, decision, actuator):
        """
        This is the isolation boundary we designed: flag up, take a
        pre-action snapshot, apply the action, take a post-action
        snapshot, log the triple, flag down. No sampling happens for
        the outside world during this window.
        """
        self.action_in_progress = True
        try:
            pre = self.sample(tag="action_pre")
            result = actuator.apply(decision)
            post = self.sample(tag="action_post")
            actuator.log_event(decision, pre, post, result)
        finally:
            # Always clear the flag, even if the actuator raises --
            # otherwise the monitor loop would silently stop sampling
            # forever on any actuator bug.
            self.action_in_progress = False
