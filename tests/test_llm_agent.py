import json
import pytest
from agent.llm_agent import LLMAgent
from tests.test_stats import mk

LEAK = mk([100_000 + 20_000 * i for i in range(30)])
FLAT = mk([500_000] * 30)
OK = json.dumps({"action": "flag_leak", "hypothesis": "unbounded_retention", "confidence": 0.9,
                 "reason": "r2 ~1 and floor_ratio ~1"})


def agent(chat):
    return LLMAgent(chat, cooldown_s=0, review_interval_s=0)


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
