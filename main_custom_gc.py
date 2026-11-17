"""
Runs the same Agentic GC pipeline as main.py, but pointed at OUR OWN
mark-sweep collector (custom_gc) instead of the real interpreter's gc module.

Usage:
    python main_custom_gc.py
    python main_custom_gc.py --hitl interactive
"""

import argparse
import functools

import custom_gc
from workloads import cycle_generator_custom, unbounded_cache_custom, normal_churn_custom
from monitor.monitor import Monitor
from analysis.analysis import analyze
from agent.agentic_gc import get_agent, decide
from actuator.actuator import Actuator


def run_experiment(workload_module, label, agent, iterations=100, sample_interval=0.04):
    print(f"\n=======================================================")
    print(f"   RUNNING CUSTOM GC EXPERIMENT: {label.upper()}")
    print(f"=======================================================")

    # Reset the custom heap between experiments
    custom_gc.get_heap().reset()
    agent.reset()

    monitor = Monitor(sample_interval=sample_interval, gc_module=custom_gc)
    actuator = Actuator(
        log_path=f"{label}_log.jsonl",
        gc_module=custom_gc,
        workload_module=workload_module,
    )
    agent_decide = functools.partial(decide, analysis_fn=analyze)

    monitor.run(
        workload_gen=workload_module.run(iterations=iterations),
        agent_decide_fn=agent_decide,
        actuator=actuator,
        max_steps=iterations,
        workload_module=workload_module,
        workload_name=label,
    )

    obs = [s for s in monitor.samples if s["tag"] == "observation"]
    actions = [s for s in monitor.samples if s["tag"] == "action_pre"]
    heap_size_now = len(custom_gc.get_heap().objects)

    actuator.log_run_summary(label, len(obs), len(actions))

    print(f"\n[Completed {label}]")
    print(f"  Total observation samples:      {len(obs)}")
    print(f"  Agent interventions taken:      {len(actions)}")
    print(f"  Live objects remaining on heap: {heap_size_now}")
    print(f"  Audit log recorded:             {label}_log.jsonl")


def main():
    parser = argparse.ArgumentParser(description="Adaptive Garbage Collection with Custom Mark-Sweep & Agent AI")
    parser.add_argument(
        "--hitl",
        choices=["interactive", "auto-approve", "auto-reject"],
        default="auto-approve",
        help="Human-in-the-Loop policy",
    )
    parser.add_argument(
        "--workload",
        choices=["all", "cycle_generator_custom", "unbounded_cache_custom", "normal_churn_custom"],
        default="all",
        help="Specific workload to execute",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=100,
        help="Number of iterations per workload run (default: 100)",
    )
    args = parser.parse_args()

    agent = get_agent(hitl_mode=args.hitl)
    print(f"Agentic Custom GC runtime initialized. Human-in-the-Loop policy: [{args.hitl.upper()}]")

    workload_map = {
        "cycle_generator_custom": (cycle_generator_custom, "cycle_generator_custom"),
        "unbounded_cache_custom": (unbounded_cache_custom, "unbounded_cache_custom"),
        "normal_churn_custom": (normal_churn_custom, "normal_churn_custom"),
    }

    if args.workload == "all":
        for mod, name in workload_map.values():
            run_experiment(mod, name, agent, iterations=args.iterations)
    else:
        mod, name = workload_map[args.workload]
        run_experiment(mod, name, agent, iterations=args.iterations)


if __name__ == "__main__":
    main()
