"""
Runs the same monitor -> analyze -> decide -> act -> log pipeline as
main.py, but pointed at OUR OWN mark-sweep collector (custom_gc) instead
of the real interpreter's gc module. Kept as a separate entry point from
main.py so both backends stay independently runnable and comparable --
you can show "here's the pipeline against real Python gc" and "here's
the same pipeline against a collector we implemented ourselves" side by
side.

Usage:
    python main_custom_gc.py
"""

import functools

import custom_gc
from workloads import cycle_generator_custom, unbounded_cache_custom, normal_churn_custom
from monitor.monitor import Monitor
from analysis.analysis import analyze
from agent.rule_based import decide
from actuator.actuator import Actuator


def run_experiment(workload_module, label, iterations=100, sample_interval=0.05):
    print(f"--- Running {label} (custom_gc backend) ---")

    # Reset the shared heap before each experiment -- otherwise leftover
    # objects from the previous workload would still be sitting on the
    # heap and confound this workload's numbers.
    custom_gc.get_heap().reset()

    monitor = Monitor(sample_interval=sample_interval, gc_module=custom_gc)
    actuator = Actuator(log_path=f"{label}_log.jsonl", gc_module=custom_gc)
    agent_decide = functools.partial(decide, analysis_fn=analyze)

    monitor.run(
        workload_gen=workload_module.run(iterations=iterations),
        agent_decide_fn=agent_decide,
        actuator=actuator,
        max_steps=iterations,
    )

    obs = [s for s in monitor.samples if s["tag"] == "observation"]
    actions = [s for s in monitor.samples if s["tag"] == "action_pre"]
    heap_size_now = len(custom_gc.get_heap().objects)

    actuator.log_run_summary(label, len(obs), len(actions))

    print(f"  observation samples: {len(obs)}")
    print(f"  actuator actions taken: {len(actions)}")
    print(f"  live objects remaining on heap: {heap_size_now}")
    print(f"  log written to {label}_log.jsonl")
    print()


if __name__ == "__main__":
    run_experiment(cycle_generator_custom, "cycle_generator_custom", iterations=100)
    run_experiment(unbounded_cache_custom, "unbounded_cache_custom", iterations=100)
    run_experiment(normal_churn_custom, "normal_churn_custom", iterations=100)
