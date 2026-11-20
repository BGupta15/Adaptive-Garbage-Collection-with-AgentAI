import json
import pytest
from agent.llm_agent import LLMAgent
from tests.test_stats import mk

LEAK = mk([100_000 + 20_000 * i for i in range(30)])
FLAT = mk([500_000] * 30)
OK = json.dumps({"action": "flag_leak", "hypothesis": "unbounded_retention", "confidence": 0.9,
                 "reason": "r2 ~1 and floor_ratio ~1"})


def agent(chat):
    return LLMAgent(chat, cooldown_s=0, review_interval_s=0,
                    observations_between_reviews=1, required_suspicious_reviews=1)


def test_gate_skips_llm_on_healthy():
    calls = []
    d = agent(lambda s, u: calls.append(1) or OK).decide(FLAT)
    assert d["action"] == "noop" and d["source"] == "gate" and not calls


def test_llm_decision_used_and_features_logged():
    d = agent(lambda s, u: OK).decide(LEAK)
    assert d["action"] == "flag_leak" and d["source"] == "llm" and "features" in d


def test_bad_json_retries_then_falls_back_to_rules():
    n = []
    a = agent(lambda s, u: n.append(1) or "not json {{{")
    d = a.decide(LEAK)
    assert len(n) == 2 and d["source"] == "rule_fallback" and a.stats()["fallbacks"] == 1


def test_exception_in_client_falls_back():
    def boom(s, u): raise TimeoutError("network")
    assert agent(boom).decide(LEAK)["source"] == "rule_fallback"


def test_prompt_injection_style_action_rejected():
    bad = json.dumps({"action": "os.system", "reason": "x", "confidence": 1})
    assert agent(lambda s, u: bad).decide(LEAK)["source"] == "rule_fallback"


def test_cycle_signal_overrides_llm_noop():
    from tests.test_stats import mk
    values = [100_000 + 5_000 * i for i in range(30)]
    samples = mk(values, objs=[10 * i for i in range(30)])
    noop = json.dumps({"action": "noop", "hypothesis": "unknown",
                       "confidence": 0.9, "reason": "model saw no issue"})
    d = LLMAgent(lambda s, u: noop, cooldown_s=0, review_interval_s=0,
                 observations_between_reviews=1,
                 required_suspicious_reviews=1).decide(
                     samples, workload_name="cycle_generator")
    assert d["action"] == "collect"
    assert d["hybrid_override"] == "cycle_signal"


def test_llm_can_use_slow_leak_cleanup_adapter():
    raw = json.dumps({
        "action": "release_retained",
        "hypothesis": "unbounded_retention",
        "confidence": 0.95,
        "reason": "Persistent rising floor; use the approved workload cleanup hook.",
        "release_ratio": 0.5,
    })
    seen = {}

    def chat(system, user):
        seen["system"] = system
        seen["user"] = json.loads(user)
        return raw

    d = LLMAgent(chat, cooldown_s=0, review_interval_s=0,
                 observations_between_reviews=1,
                 required_suspicious_reviews=1).decide(
                     LEAK, workload_name="slow_leak")
    assert d["action"] == "release_retained"
    assert d["release_ratio"] == 0.5
    assert seen["user"]["retention_cleanup_supported"] is True
    assert "release_retained" in seen["system"]


def test_llm_cleanup_action_is_rejected_for_unsupported_workload():
    raw = json.dumps({
        "action": "release_retained",
        "hypothesis": "unbounded_retention",
        "confidence": 0.95,
        "reason": "Attempt cleanup.",
        "release_ratio": 0.5,
    })
    a = LLMAgent(lambda system, user: raw, cooldown_s=0, review_interval_s=0,
                 observations_between_reviews=1,
                 required_suspicious_reviews=1)
    d = a.decide(LEAK, workload_name="unbounded_cache")
    assert d["action"] == "flag_leak"
    assert a.stats()["guard_overrides"] == 1
