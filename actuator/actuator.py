"""
Applies agent decisions to the runtime (gc.collect, gc.set_threshold,
cache eviction) under strict safety bounds. Logs every action as a
structured before/after record with timing.

This is the module where a bug is most dangerous -- it's the one place
code here can actually change interpreter-wide state -- so bounds are
enforced here even though the agent is also expected to be sensible.
"""

import gc as _real_gc
import json
import time


class Actuator:
    def __init__(self, log_path="run_log.jsonl", threshold_bounds=(100, 100_000),
                 gc_module=None, workload_module=None):
        self.log_path = log_path
        self.threshold_bounds = threshold_bounds
        self.gc_module = gc_module if gc_module is not None else _real_gc
        self.workload_module = workload_module
        open(self.log_path, "w").close()

    def set_workload_module(self, workload_module):
        self.workload_module = workload_module

    def apply(self, decision, workload_module=None):
        action = decision.get("action")
        target_workload = workload_module or self.workload_module

        if action == "collect":
            # perf_counter = highest-resolution monotonic clock for short intervals
            t0 = time.perf_counter()
            freed = self.gc_module.collect()
            pause_ms = (time.perf_counter() - t0) * 1000
            return {"objects_collected": freed, "duration_ms": round(pause_ms, 3)}

        elif action == "evict_cache":
            # Human-approved application-level cache eviction
            ratio = decision.get("evict_ratio", 0.5)
            evicted = 0
            if target_workload and hasattr(target_workload, "evict"):
                evicted = target_workload.evict(ratio=ratio)
            t0 = time.perf_counter()
            freed = self.gc_module.collect()
            pause_ms = (time.perf_counter() - t0) * 1000
            return {
                "evicted_items": evicted,
                "objects_collected": freed,
                "duration_ms": round(pause_ms, 3),
            }

        elif action == "tune_threshold":
            requested = decision.get("threshold", 700)
            low, high = self.threshold_bounds
            # Hard clamp -- safety rail the LLM cannot override
            bounded = max(low, min(high, requested))
            self.gc_module.set_threshold(bounded)
            return {"requested_threshold": requested, "applied_threshold": bounded}

        elif action in ("flag_cycle", "flag_leak"):
            # Non-mutating markers: no interpreter state change
            # flag_cycle = unreachable reference cycles
            # flag_leak  = referenced memory (collect() can't help; fix is in the app)
            return {"flagged": True, "kind": action.split("_")[1]}

        elif action == "flag_unbounded_cache":
            return {"flagged": True, "type": "unbounded_cache_leak"}

        else:
            return {"noop": True}

    def log_event(self, decision, pre, post, result):
        record = {
            "decision": decision,
            "pre": {
                "timestamp": pre["timestamp"],
                "traced_current_bytes": pre["traced_current_bytes"],
                "rss_bytes": pre["rss_bytes"],
            },
            "post": {
                "timestamp": post["timestamp"],
                "traced_current_bytes": post["traced_current_bytes"],
                "rss_bytes": post["rss_bytes"],
            },
            "result": result,
        }
        with open(self.log_path, "a") as f:
            f.write(json.dumps(record) + "\n")

    def log_run_summary(self, label, total_observations, total_actions, extra=None):
        """
        Written once at the end of a run. The `extra` kwarg lets the LLM agent
        pass its call counters, fallback count, and latency into the log for
        evaluator.evaluate to pick up and compare against the rule agent.
        """
        summary = {
            "type": "run_summary",
            "label": label,
            "timestamp": time.time(),
            "total_observations": total_observations,
            "total_actions": total_actions,
        }
        if extra:
            summary.update(extra)
        with open(self.log_path, "a") as f:
            f.write(json.dumps(summary) + "\n")
