"""
Runs each synthetic workload through the full monitor -> analyze ->
decide -> act -> log loop, one workload at a time, and writes a JSONL
log of every actuator action for each run.

Usage:
    python main.py
"""

import functools

from workloads import cycle_generator, unbounded_cache, normal_churn
from monitor.monitor import Monitor
from analysis.analysis import analyze
from agent.rule_based import decide
from actuator.actuator import Actuator


def run_experiment(workload_module, label, iterations=100, sample_interval=0.1):
    print(f"--- Running {label} ---")

    monitor = Monitor(sample_interval=sample_interval)
    actuator = Actuator(log_path=f"{label}_log.jsonl")
    agent_decide = functools.partial(decide, analysis_fn=analyze)

    monitor.run(
        workload_gen=workload_module.run(iterations=iterations),
        agent_decide_fn=agent_decide,
        actuator=actuator,
        max_steps=iterations,
    )

    obs = [s for s in monitor.samples if s["tag"] == "observation"]
    actions = [s for s in monitor.samples if s["tag"] == "action_pre"]

    print(f"  observation samples: {len(obs)}")
    print(f"  actuator actions taken: {len(actions)}")
    print(f"  log written to {label}_log.jsonl")
    print()


if __name__ == "__main__":
    # Keep iterations small for a first run -- raise once you've confirmed
    # each workload produces the signal you expect.
    run_experiment(cycle_generator, "cycle_generator", iterations=100)
    run_experiment(unbounded_cache, "unbounded_cache", iterations=100)
    run_experiment(normal_churn, "normal_churn", iterations=100)
