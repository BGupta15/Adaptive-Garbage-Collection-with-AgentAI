import pytest
from agent.guard import validate, enforce_rails

GOOD = {"action": "collect", "reason": "r", "hypothesis": "cycle_garbage", "confidence": 0.9}


def test_validate_accepts_good():
    assert validate(GOOD)["action"] == "collect"


@pytest.mark.parametrize("bad", [
    {**GOOD, "action": "rm -rf"},
    {**GOOD, "reason": ""},
    {**GOOD, "confidence": 1.5},
    {**GOOD, "confidence": True},
    {**GOOD, "hypothesis": "aliens"},
    {**GOOD, "action": "tune_threshold"},            # missing threshold
    {**GOOD, "action": "tune_threshold", "threshold": "7"},
    "not a dict",
])
def test_validate_rejects_bad(bad):
    with pytest.raises(ValueError):
        validate(bad)


def test_low_confidence_mutation_downgraded():
    d = enforce_rails({**GOOD, "confidence": 0.3}, [], 0, 0)
    assert d["action"] == "noop" and "guard" in d


def test_futile_collect_escalates_to_flag_leak():
    d = enforce_rails(GOOD, [], futile_streak=2, actions_taken=1)
    assert d["action"] == "flag_leak"


def test_max_actions_blocks_mutation_but_not_flags():
    assert enforce_rails(GOOD, [], 0, actions_taken=10)["action"] == "noop"
    flag = {**GOOD, "action": "flag_leak"}
    assert enforce_rails(flag, [], 0, actions_taken=10)["action"] == "flag_leak"
