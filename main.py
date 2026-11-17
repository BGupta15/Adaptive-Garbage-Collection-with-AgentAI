"""
Runs each synthetic workload through the full Agentic GC loop:
Monitor -> Analyze -> Agentic Diagnosis & Decision (with HITL Gate) -> Actuate -> Log

Usage:
    # Run with simulated human approval for demonstration (default)
    python main.py

    # Run with live interactive human approval prompts in the terminal
    python main.py --hitl interactive

    # Run single workload
    python main.py --workload unbounded_cache --hitl interactive
"""

import argparse
import functools

from workloads import cycle_generator, unbounded_cache, normal_churn
from monitor.monitor import Monitor
from analysis.analysis import analyze
from agent.agentic_gc import get_agent, decide
from actuator.actuator import Actuator


def run_experiment(workload_module, label, agent, iterations=100, sample_interval=0.08):
    print(f"\n=======================================================")
    print(f"   RUNNING EXPERIMENT: {label.upper()}")
    print(f"=======================================================")

    agent.reset()
    monitor = Monitor(sample_interval=sample_interval)
    actuator = Actuator(log_path=f"{label}_log.jsonl", workload_module=workload_module)
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

    actuator.log_run_summary(label, len(obs), len(actions))

    print(f"\n[Completed {label}]")
    print(f"  Total observation samples:  {len(obs)}")
    print(f"  Agent interventions taken:  {len(actions)}")
    print(f"  Audit log recorded:         {label}_log.jsonl")


def main():
    parser = argparse.ArgumentParser(description="Adaptive Garbage Collection with Agent AI")
    parser.add_argument(
        "--hitl",
        choices=["interactive", "auto-approve", "auto-reject"],
        default="auto-approve",
        help="Human-in-the-Loop policy: 'interactive' prompts the operator; 'auto-approve' simulates approval for hands-free demos.",
    )
    parser.add_argument(
        "--workload",
        choices=["all", "cycle_generator", "unbounded_cache", "normal_churn"],
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
    print(f"Agentic GC runtime initialized. Human-in-the-Loop policy: [{args.hitl.upper()}]")

    workload_map = {
        "cycle_generator": (cycle_generator, "cycle_generator"),
        "unbounded_cache": (unbounded_cache, "unbounded_cache"),
        "normal_churn": (normal_churn, "normal_churn"),
    }

    if args.workload == "all":
        for mod, name in workload_map.values():
            run_experiment(mod, name, agent, iterations=args.iterations)
    else:
        mod, name = workload_map[args.workload]
        run_experiment(mod, name, agent, iterations=args.iterations)


if __name__ == "__main__":
    main()
