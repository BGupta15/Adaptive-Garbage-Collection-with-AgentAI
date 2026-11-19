"""
Runs each synthetic workload through the full Agentic GC loop:
Monitor -> Analyze -> Decide (Agentic/LLM/Rule) -> Actuate -> Log

Usage:
    python main.py                              # agentic agent, all workloads, auto-approve HITL
    python main.py --agent llm                  # LLM agent (needs GROQ_API_KEY in .env)
    python main.py --agent both                 # run rule + llm, then print comparison table
    python main.py --agent agentic --hitl interactive   # live terminal HITL prompts
    python main.py --workload unbounded_cache   # single workload
    python main.py --iterations 50              # fewer steps for quick dev runs
"""

import argparse
import functools

from workloads import cycle_generator, unbounded_cache, normal_churn, slow_leak, bursty
from monitor.monitor import Monitor
from analysis.analysis import analyze
from agent.agentic_gc import get_agent, decide as agentic_decide
from agent.rule_based import decide as rule_decide
from actuator.actuator import Actuator

WORKLOADS = {
    "cycle_generator": cycle_generator,
    "unbounded_cache": unbounded_cache,
    "normal_churn": normal_churn,
    "slow_leak": slow_leak,
    "bursty": bursty,
}


def run_experiment(workload_module, label, agent_name="agentic", chat_fn=None,
                   hitl_mode="auto-approve", iterations=100, sample_interval=0.08):
    print(f"\n=======================================================")
    print(f"   RUNNING EXPERIMENT: {label.upper()} [{agent_name}]")
    print(f"=======================================================")

    suffix = "" if agent_name in ("rule", "agentic") else f"_{agent_name}"
    log_path = f"{label}{suffix}_log.jsonl"

    monitor = Monitor(sample_interval=sample_interval)
    actuator = Actuator(log_path=log_path, workload_module=workload_module)

    llm_agent_obj = None
    if agent_name == "llm":
        from agent.llm_agent import LLMAgent
        llm_agent_obj = LLMAgent(chat_fn)
        agent_decide_fn = llm_agent_obj.decide
    elif agent_name == "rule":
        agent_decide_fn = functools.partial(rule_decide, analysis_fn=analyze)
    else:  # agentic (default)
        agent = get_agent(hitl_mode=hitl_mode)
        agent.reset()
        agent_decide_fn = functools.partial(agentic_decide, analysis_fn=analyze)

    monitor.run(
        workload_gen=workload_module.run(iterations=iterations),
        agent_decide_fn=agent_decide_fn,
        actuator=actuator,
        max_steps=iterations,
        workload_module=workload_module,
        workload_name=label,
    )

    obs = [s for s in monitor.samples if s["tag"] == "observation"]
    actions = [s for s in monitor.samples if s["tag"] == "action_pre"]
    extra = llm_agent_obj.stats() if llm_agent_obj else None
    actuator.log_run_summary(label, len(obs), len(actions), extra=extra)

    print(f"\n[Completed {label}]")
    print(f"  Total observation samples:  {len(obs)}")
    print(f"  Agent interventions taken:  {len(actions)}")
    if extra:
        print(f"  LLM stats:                  {extra}")
    print(f"  Audit log recorded:         {log_path}")


def main():
    parser = argparse.ArgumentParser(description="Adaptive Garbage Collection with Agent AI")
    parser.add_argument(
        "--agent",
        choices=["agentic", "rule", "llm", "both"],
        default="agentic",
        help="Decision layer: 'agentic' (our HITL engine), 'rule' (original rule-based), 'llm' (Groq LLM), 'both' (rule+llm comparison)",
    )
    parser.add_argument(
        "--hitl",
        choices=["interactive", "auto-approve", "auto-reject"],
        default="auto-approve",
        help="Human-in-the-Loop policy for the agentic agent",
    )
    parser.add_argument(
        "--workload",
        choices=["all"] + list(WORKLOADS.keys()),
        default="all",
        help="Specific workload to execute",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=100,
        help="Number of iterations per workload run",
    )
    args = parser.parse_args()

    chat_fn = None
    if args.agent in ("llm", "both"):
        from agent.llm_agent import make_groq_chat
        chat_fn = make_groq_chat()

    agents = ["rule", "llm"] if args.agent == "both" else [args.agent]
    workloads = WORKLOADS if args.workload == "all" else {args.workload: WORKLOADS[args.workload]}

    print(f"Agentic GC runtime | Agent: [{args.agent}] | HITL policy: [{args.hitl}]")

    for agent_name in agents:
        for label, mod in workloads.items():
            run_experiment(mod, label, agent_name=agent_name, chat_fn=chat_fn,
                           hitl_mode=args.hitl, iterations=args.iterations)

    if args.agent == "both":
        from evaluator.evaluate import compare, print_table
        rows = compare()
        if rows:
            print("\n--- Agent Comparison Table ---")
            print_table(rows)


if __name__ == "__main__":
    main()
