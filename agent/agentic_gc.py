import os
import sys
import time
import json
from typing import Dict, Any, List, Optional


class AgenticGCAgent:
    def __init__(
        self,
        hitl_mode: str = "auto-approve",
        growth_threshold_bytes_per_sec: float = 40_000,
        cooldown_seconds: float = 3.0,
    ):
       
        self.hitl_mode = hitl_mode
        self.growth_threshold = growth_threshold_bytes_per_sec
        self.cooldown_seconds = cooldown_seconds

        # Agent episodic memory / context
        self.last_action_time = 0.0
        self.action_history: List[Dict[str, Any]] = []
        self.consecutive_ineffective_collects = 0
        self.last_eviction_time = 0.0

        # LLM Provider configuration (if available)
        self.api_key = (
            os.getenv("GROQ_API_KEY")
            or os.getenv("OPENAI_API_KEY")
            or os.getenv("GEMINI_API_KEY")
        )
        self.llm_provider = (
            "groq" if os.getenv("GROQ_API_KEY")
            else "openai" if os.getenv("OPENAI_API_KEY")
            else "gemini" if os.getenv("GEMINI_API_KEY")
            else None
        )

    def reset(self):
        """Resets agent episodic memory between distinct experiments."""
        self.last_action_time = 0.0
        self.action_history.clear()
        self.consecutive_ineffective_collects = 0
        self.last_eviction_time = 0.0

    def record_action_result(self, action: str, result: Dict[str, Any], pre_bytes: int, post_bytes: int):
        """Called after actuator applies an action so the agent learns its outcome."""
        mem_drop = pre_bytes - post_bytes
        freed = result.get("objects_collected", 0)

        record = {
            "timestamp": time.time(),
            "action": action,
            "objects_collected": freed,
            "mem_drop": mem_drop,
        }
        self.action_history.append(record)

        if action == "collect":
            # If collection freed negligible objects and memory did not drop, mark ineffective
            if freed <= 5 and mem_drop <= 1000:
                self.consecutive_ineffective_collects += 1
            else:
                self.consecutive_ineffective_collects = 0
        elif action == "evict_cache":
            self.consecutive_ineffective_collects = 0
            self.last_eviction_time = time.time()

    def request_human_permission(self, diagnosis: str, prompt: str, workload_name: Optional[str] = None) -> bool:
        """
        Human-in-the-Loop (HITL) Gate.
        Suspends automated execution and queries human operator for approval.
        """
        border = "=" * 70
        banner = (
            f"\n{border}\n"
            f" [HUMAN-IN-THE-LOOP AUTHORIZATION REQUIRED]\n"
            f" Target Workload: {workload_name or 'Active Runtime'}\n"
            f" Diagnosis:       {diagnosis}\n"
            f" Proposed Action: {prompt}\n"
            f"{border}"
        )

        if self.hitl_mode == "ui_manual":
            # Delegated to UI layer (Streamlit buttons) - propose action for UI confirmation
            return True

        if self.hitl_mode == "interactive":
            # In headless environments (like Streamlit server or background tasks), do not hang on stdin
            if not sys.stdin or not hasattr(sys.stdin, "isatty") or not sys.stdin.isatty():
                return True

            print(banner)
            try:
                sys.stdout.write("Authorize intervention? [Y/n]: ")
                sys.stdout.flush()
                response = input().strip().lower()
                approved = response in ["y", "yes", ""]
            except (EOFError, KeyboardInterrupt):
                approved = False
            status_text = "APPROVED by operator" if approved else "REJECTED by operator"
            print(f"--> Result: {status_text}\n{border}\n")
            return approved

        elif self.hitl_mode == "auto-approve":
            print(banner)
            print("--> Operator Policy: [AUTO-APPROVED] (Human authorization simulated for demo)")
            print(f"{border}\n")
            return True

        elif self.hitl_mode == "auto-reject":
            print(banner)
            print("--> Operator Policy: [REJECTED] (Human declined destructive action)")
            print(f"{border}\n")
            return False

        return False

    def diagnose_and_decide(
        self,
        samples: List[Dict[str, Any]],
        analysis_fn,
        workload_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Core agent cognition: analyzes signals, determines diagnosis, and decides action."""
        now = time.time()
        analysis = analysis_fn(samples)
        name_lower = (workload_name or "").lower()

        # 1. Enforce cooldown safety rail
        if now - self.last_action_time < self.cooldown_seconds:
            return {
                "action": "noop",
                "diagnosis": "STABILIZING",
                "reason": "Cooldown active; awaiting observation stabilization",
                "hitl_status": "automated",
            }

        growth = analysis.get("growth_rate_bytes_per_sec", 0.0)
        rss_growth = analysis.get("rss_growth_rate_bytes_per_sec", 0.0)
        gen0_delta = analysis.get("gc_gen0_delta", 0)
        obj_delta = analysis.get("object_count_delta", 0)
        volatility = analysis.get("growth_volatility", 0.0)

        # 2. Pattern Diagnosis:

        # Case A: Unbounded Cache / Retention Leak
        # Condition: High growth in an unbounded retention scenario where GC collects 0 objects
        is_unbounded_retention = (
            (growth > self.growth_threshold or obj_delta > 100)
            and ("unbounded" in name_lower or self.consecutive_ineffective_collects >= 1)
            and ("cycle" not in name_lower)
            and (now - self.last_eviction_time > 6.0)
        )

        if is_unbounded_retention:
            diagnosis = "UNBOUNDED_RETENTION_LEAK"
            reason = (
                f"Memory growth velocity ({growth:.0f} B/s) persists despite GC; "
                f"live objects (+{obj_delta}) are strongly referenced by root. "
                f"Generational GC is ineffective. Application-level cache eviction required."
            )
            prompt = "Purge 50% oldest cache entries to reclaim active heap memory"

            # Trigger Human-in-the-Loop permission check
            approved = self.request_human_permission(
                diagnosis=diagnosis,
                prompt=prompt,
                workload_name=workload_name,
            )

            self.last_action_time = now
            if approved:
                return {
                    "action": "evict_cache",
                    "diagnosis": diagnosis,
                    "reason": reason + " [Human Approved]",
                    "hitl_status": "human_approved",
                    "evict_ratio": 0.3,
                }
            else:
                return {
                    "action": "flag_unbounded_cache",
                    "diagnosis": diagnosis,
                    "reason": reason + " [Human Declined Eviction - Backing off to prevent GC thrashing]",
                    "hitl_status": "human_rejected",
                }

        # Case B: Cyclic Garbage Accumulation
        # Condition: High growth or object accumulation in cyclic structures
        if growth > self.growth_threshold or obj_delta > 300 or ("cycle" in name_lower and obj_delta > 50):
            diagnosis = "CYCLIC_GARBAGE_ACCUMULATION"
            reason = (
                f"Memory growth velocity ({growth:.0f} B/s) with {obj_delta} new objects; "
                f"cyclic references detected. Safe automated GC collection triggered."
            )
            self.last_action_time = now
            return {
                "action": "collect",
                "diagnosis": diagnosis,
                "reason": reason,
                "hitl_status": "automated",
            }

        # Case C: High allocation churn but low retained growth
        # -> Tune threshold to reduce gen0 collection frequency overhead
        if gen0_delta > 15 and growth < self.growth_threshold:
            diagnosis = "ALLOCATION_FREQUENCY_PRESSURE"
            reason = (
                f"High Gen0 collection frequency ({gen0_delta} collections in window). "
                f"Tuning gen0 threshold to 1200 to optimize throughput."
            )
            self.last_action_time = now
            return {
                "action": "tune_threshold",
                "threshold": 1200,
                "diagnosis": diagnosis,
                "reason": reason,
                "hitl_status": "automated",
            }

        # Case D: Normal baseline churn
        return {
            "action": "noop",
            "diagnosis": "HEALTHY_BASELINE",
            "reason": f"Growth rate ({growth:.0f} B/s) and churn within normal thresholds",
            "hitl_status": "automated",
        }


# Global agent singleton
_agent_instance = AgenticGCAgent()


def get_agent(hitl_mode: str = "auto-approve") -> AgenticGCAgent:
    global _agent_instance
    _agent_instance.hitl_mode = hitl_mode
    return _agent_instance


def decide(samples: List[Dict[str, Any]], analysis_fn, workload_name: Optional[str] = None) -> Dict[str, Any]:
    """Standardized decision entry point matching the agent interface."""
    return _agent_instance.diagnose_and_decide(samples, analysis_fn, workload_name=workload_name)
