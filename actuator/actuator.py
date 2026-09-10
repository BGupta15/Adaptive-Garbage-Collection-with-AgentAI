"""
Applies agent decisions to the real interpreter (gc.collect,
gc.set_threshold) and logs every action as a structured before/after
record. This is the module where a bug is most dangerous -- it's the one
place code here can actually change interpreter-wide state -- so bounds
are enforced here even though the agent is also expected to be sensible.
"""

import gc
import json


class Actuator:
    def __init__(self, log_path="run_log.jsonl", threshold_bounds=(100, 100_000)):
        self.log_path = log_path
        self.threshold_bounds = threshold_bounds  # (min, max) for gc.set_threshold

    def apply(self, decision):
        action = decision.get("action")

        if action == "collect":
            freed = gc.collect()
            return {"objects_collected": freed}

        elif action == "tune_threshold":
            requested = decision.get("threshold", 700)
            low, high = self.threshold_bounds
            # Hard clamp -- this is the safety rail, not a suggestion.
            # An LLM-driven decision that requests threshold=1 (which
            # would make GC run almost continuously) gets clamped here
            # regardless of what the agent asked for.
            bounded = max(low, min(high, requested))
            gc.set_threshold(bounded)
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
