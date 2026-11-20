import json

from evaluator.evaluate import print_table, summarize_run


def _write_run(tmp_path, label, agent, records):
    suffix = "" if agent == "rule" else f"_{agent}"
    path = tmp_path / f"{label}{suffix}_log.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")
    return path


def test_healthy_workload_remediation_is_not_applicable(tmp_path):
    _write_run(tmp_path, "normal_churn", "rule", [
        {"type": "run_summary", "total_observations": 100, "total_actions": 0}
    ])
    row = summarize_run("normal_churn", "rule", str(tmp_path))
    assert row["detection_correct"] is True
    assert row["false_positive"] is False
    assert row["remediation_success"] is None


def test_slow_leak_cleanup_counts_as_detection_and_remediation(tmp_path):
    _write_run(tmp_path, "slow_leak", "rule", [
        {"decision": {"action": "release_retained"},
         "pre": {"traced_current_bytes": 100_000},
         "post": {"traced_current_bytes": 40_000},
         "result": {"duration_ms": 2.5}},
        {"type": "run_summary", "total_observations": 100, "total_actions": 1}
    ])
    row = summarize_run("slow_leak", "rule", str(tmp_path))
    assert row["detection_correct"] is True
    assert row["remediation_success"] is True
    assert row["bytes_freed"] == 60_000


def test_healthy_table_displays_remediation_as_na(tmp_path, capsys):
    row = {
        "label": "normal_churn", "agent": "rule",
        "detection_correct": True, "remediation_success": None,
        "mutating_actions": 0, "wasted_actions": 0,
        "false_positive": False, "missed_leak": False,
        "bytes_freed": 0, "total_pause_ms": 0.0,
    }
    print_table([row])
    assert "N/A" in capsys.readouterr().out
