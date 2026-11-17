"""
Applies agent decisions to the runtime (gc.collect, gc.set_threshold, cache eviction)
under strict safety bounds and logs every action as a structured before/after record.
"""

import gc as _real_gc
import json
import time


class Actuator:
    def __init__(self, log_path="run_log.jsonl", threshold_bounds=(100, 100_000), gc_module=None, workload_module=None):
        self.log_path = log_path
        self.threshold_bounds = threshold_bounds  # (min, max) for gc.set_threshold
        self.gc_module = gc_module if gc_module is not None else _real_gc
        self.workload_module = workload_module

        # Ensure the log file is created up front
        open(self.log_path, "w").close()

    def set_workload_module(self, workload_module):
        self.workload_module = workload_module

    def apply(self, decision, workload_module=None):
        action = decision.get("action")
        target_workload = workload_module or self.workload_module

        if action == "collect":
            freed = self.gc_module.collect()
            return {"objects_collected": freed}

        elif action == "evict_cache":
            ratio = decision.get("evict_ratio", 0.5)
            evicted = 0
            if target_workload and hasattr(target_workload, "evict"):
                evicted = target_workload.evict(ratio=ratio)
            # Sweep to reclaim newly unreferenced objects
            freed = self.gc_module.collect()
            return {
                "evicted_items": evicted,
                "objects_collected": freed,
            }

        elif action == "tune_threshold":
            requested = decision.get("threshold", 700)
            low, high = self.threshold_bounds
            bounded = max(low, min(high, requested))
            self.gc_module.set_threshold(bounded)
            return {"requested_threshold": requested, "applied_threshold": bounded}

        elif action == "flag_unbounded_cache":
            return {"flagged": True, "type": "unbounded_cache_leak"}

        elif action == "flag_cycle":
            return {"flagged": True, "type": "cyclic_leak"}

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

    def log_run_summary(self, label, total_observations, total_actions):
        summary = {
            "type": "run_summary",
            "label": label,
            "timestamp": time.time(),
            "total_observations": total_observations,
            "total_actions": total_actions,
        }
        with open(self.log_path, "a") as f:
            f.write(json.dumps(summary) + "\n")
