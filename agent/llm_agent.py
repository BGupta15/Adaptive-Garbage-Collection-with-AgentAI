"""
LLM decision layer. Same contract as agent/rule_based.py:

    agent.decide(samples, analysis_fn=None) -> {"action": ..., "reason": ...}

so it drops into Monitor.run() unchanged. The pipeline per call:

  stats features -> cheap gate -> (cooldown / call budget) -> LLM call ->
  schema validation -> hard rails (agent/guard.py) -> decision

The LLM never sees raw samples, only the features from analysis/stats.py,
the outcomes of its own past actions, and top allocation sites. If the API
fails or returns garbage twice, we fall back to the rule-based agent, so a
flaky network can't break a run.
"""

import json
import os
import time

from analysis.analysis import analyze
from analysis.stats import (action_history, futile_streak, suspicion_score,
                            top_allocation_sites, window_features)
from agent import rule_based
from agent.guard import MUTATING, DecisionError, enforce_rails, validate

GATE = 0.35               # suspicion score needed before we even consider an LLM call

SYSTEM_PROMPT = """You are the decision layer of an adaptive garbage-collection agent for a running Python process.
You receive statistical features of recent memory behaviour and the measured outcomes of your own earlier actions. Choose ONE action.

Actions:
- "collect": run a full gc collection now. Only helps if memory is unreachable garbage (e.g. reference cycles).
- "tune_threshold": set the gen0 threshold (include integer "threshold"). Use sparingly; it persists.
- "flag_cycle": no state change; report that unreachable reference cycles are accumulating.
- "flag_leak": no state change; report memory that is still REFERENCED and growing (collect cannot free it; the fix is in the program, e.g. cache eviction).
- "noop": nothing is wrong, or evidence is insufficient.

Feature meanings:
- trend_r2: how linear the growth is (1 = perfectly steady, 0 = no trend).
- floor_ratio: share of the growth that persists between collections (~1 = memory is retained, ~0 = it is reclaimed).
- window_growth_pct: growth over the window as % of average level.
- sawtooth_amplitude_pct: size of swings between low and high points.
- gc_collected_in_window: objects the collector already reclaimed on its own in this window.
- volatility_cv: std-dev of step deltas relative to mean delta (high = bursty/noisy).
- recent_actions: what your previous actions actually freed. reclaim_ratio < 0.05 means the action was futile.

Rules: Base the decision on the numbers, not on guesses. Do not repeat an action that recent_actions shows was futile. Prefer "noop" when the evidence is weak. Healthy bursts that are freed without the collector (large swings, gc_collected_in_window near 0, no persistent floor rise) are not a problem.

Respond with ONLY a JSON object:
{"action": "...", "hypothesis": "cycle_garbage|unbounded_retention|bursty|healthy|unknown", "confidence": 0.0-1.0, "reason": "<one sentence citing specific numbers>", "recommendation": "<optional short fix suggestion>", "threshold": <int, only for tune_threshold>}"""


def make_groq_chat(model=None, timeout=10.0):
    """
    Returns chat(system, user) -> str using the Groq API. Imported lazily so
    the rest of the project runs without the SDK installed.
    """
    from groq import Groq
    try:
        from dotenv import load_dotenv
        load_dotenv()                      # reads GROQ_API_KEY from .env
    except ImportError:
        pass
    client = Groq(api_key=os.environ["GROQ_API_KEY"], timeout=timeout, max_retries=0)
    model = model or os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")

    def chat(system, user):
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            temperature=0,
            max_tokens=300,
            response_format={"type": "json_object"},   # forces syntactically valid JSON
        )
        return resp.choices[0].message.content

    return chat


class LLMAgent:
    def __init__(self, chat_fn, cooldown_s=3.0, review_interval_s=2.0, window=20):
        """
        cooldown_s:        min seconds between state-changing actions
        review_interval_s: min seconds between LLM calls (the monitor samples
                           every ~0.1s; we must not call the API that often)
        """
        self.chat_fn = chat_fn
        self.cooldown_s = cooldown_s
        self.review_interval_s = review_interval_s
        self.window = window
        self._last_llm_call = float("-inf")
        self._last_mutation = float("-inf")
        self._actions_taken = 0
        self._recent_decisions = []        # the agent's own short-term memory
        self._counters = {"llm_calls": 0, "fallbacks": 0, "guard_overrides": 0,
                          "gated_out": 0}
        self._latency_ms = 0.0

    def decide(self, samples, analysis_fn=None):
        feats = window_features(samples, self.window)
        score = suspicion_score(feats)

        if score < GATE:
            self._counters["gated_out"] += 1
            return {"action": "noop", "reason": f"below suspicion gate ({score})",
                    "source": "gate"}

        now = time.time()
        if now - self._last_llm_call < self.review_interval_s:
            return {"action": "noop", "reason": "waiting for next review", "source": "gate"}
        self._last_llm_call = now

        history = action_history(samples)
        streak = futile_streak(history)
        decision = self._ask_llm(feats, history)
        if decision is None:                                   # API/parse failure
            self._counters["fallbacks"] += 1
            fb = rule_based.decide(samples, analyze)
            fb["source"] = "rule_fallback"
            return fb

        decision = enforce_rails(decision, history, streak, self._actions_taken)
        if decision.get("guard"):
            self._counters["guard_overrides"] += 1
        if decision["action"] in MUTATING:
            if now - self._last_mutation < self.cooldown_s:
                decision = {**decision, "proposed_action": decision["action"],
                            "action": "noop", "guard": "cooldown",
                            "reason": f"[guard: cooldown active] {decision['reason']}"}
                self._counters["guard_overrides"] += 1
            else:
                self._last_mutation = now
                self._actions_taken += 1

        decision["source"] = "llm"
        decision["suspicion"] = score
        # Keep the evidence the model saw next to its answer (explainability).
        decision["features"] = {k: feats[k] for k in
                                ("traced_slope_bps", "trend_r2", "floor_ratio",
                                 "window_growth_pct", "sawtooth_amplitude_pct")}
        self._recent_decisions = (self._recent_decisions + [
            {"action": decision["action"], "hypothesis": decision["hypothesis"]}])[-3:]
        return decision

    # ------------------------------------------------------------------
    def _ask_llm(self, feats, history):
        payload = {
            "features": feats,
            "recent_actions": history[-3:],
            "your_previous_decisions": self._recent_decisions,
            "top_allocation_sites": top_allocation_sites(3),
        }
        user_msg = json.dumps(payload)

        for attempt in range(2):                # one retry, then give up -> fallback
            t0 = time.perf_counter()
            try:
                text = self.chat_fn(SYSTEM_PROMPT, user_msg)
                self._counters["llm_calls"] += 1
                self._latency_ms += (time.perf_counter() - t0) * 1000
                return validate(json.loads(text))
            except (DecisionError, json.JSONDecodeError, TypeError):
                continue                        # bad output -> retry once
            except Exception:                   # network/auth/rate limit
                break
        return None

    def stats(self):
        out = dict(self._counters)
        out["avg_llm_latency_ms"] = round(self._latency_ms / max(1, out["llm_calls"]), 1)
        return out
