"""
Applies agent decisions to the real interpreter (gc.collect,
gc.set_threshold) and logs every action as a structured before/after
record. This is the module where a bug is most dangerous -- it's the one
place code here can actually change interpreter-wide state -- so bounds
are enforced here even though the agent is also expected to be sensible.
"""

import gc as _real_gc
import json
import time


class Actuator:
    def __init__(self, log_path="run_log.jsonl", threshold_bounds=(100, 100_000), gc_module=None):
        self.log_path = log_path
        self.threshold_bounds = threshold_bounds  # (min, max) for gc.set_threshold

        # Same backend-swap pattern as Monitor: defaults to real gc,
        # accepts custom_gc (or anything else shaped the same way).
        self.gc_module = gc_module if gc_module is not None else _real_gc

        # Always create the file up front, even if this run never takes
        # an action. Previously the file was only created on the first
        # log_event() call, so a workload that correctly triggers zero
        # interventions (e.g. normal_churn) produced no file at all --
        # which looked like a bug/missing run rather than a clean result.
        open(self.log_path, "w").close()

    def apply(self, decision):
        action = decision.get("action")

        if action == "collect":
            freed = self.gc_module.collect()
            return {"objects_collected": freed}

        elif action == "tune_threshold":
            requested = decision.get("threshold", 700)
            low, high = self.threshold_bounds
            # Hard clamp -- this is the safety rail, not a suggestion.
            # An LLM-driven decision that requests threshold=1 (which
            # would make GC run almost continuously) gets clamped here
            # regardless of what the agent asked for.
            bounded = max(low, min(high, requested))
            self.gc_module.set_threshold(bounded)
            return {"requested_threshold": requested, "applied_threshold": bounded}

        elif action == "flag_cycle":
            # No interpreter state change -- just a marker for the log/
            # dashboard that the agent believes a cycle is present.
            return {"flagged": True}

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
        """
        Written once at the end of a run, regardless of whether any
        actions fired. This guarantees every workload's log file has at
        least one line -- proof the run happened, even for a baseline
        like normal_churn that should trigger zero interventions.
        """
        summary = {
            "type": "run_summary",
            "label": label,
            "timestamp": time.time(),
            "total_observations": total_observations,
            "total_actions": total_actions,
        }
        with open(self.log_path, "a") as f:
            f.write(json.dumps(summary) + "\n")
