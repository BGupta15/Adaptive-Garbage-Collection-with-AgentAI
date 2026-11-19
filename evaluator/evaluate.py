"""
Phase V evaluator: turns the JSONL action logs into comparable metrics.

Log naming: <label>_log.jsonl (rule/agentic agent), <label>_llm_log.jsonl (LLM agent).

CLI:
    python -m evaluator.evaluate                # table: rule/agentic vs llm per workload
    python -m evaluator.evaluate --report       # + LLM-written analysis (needs GROQ_API_KEY)
"""

import argparse
import json
import os

# What each workload SHOULD be diagnosed as -- known because we wrote them.
GROUND_TRUTH = {
    "cycle_generator": "cycle_garbage",
    "unbounded_cache": "unbounded_retention",
    "normal_churn": "healthy",
    "slow_leak": "unbounded_retention",
    "bursty": "healthy",
}
MUTATING = {"collect", "tune_threshold"}
FUTILE_RATIO = 0.05


def log_path(label, agent, directory="."):
    suffix = "" if agent == "rule" else f"_{agent}"
    return os.path.join(directory, f"{label}{suffix}_log.jsonl")


def load_log(path):
    actions, summary = [], None
    if not os.path.exists(path):
        return actions, summary
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("type") == "run_summary":
                summary = rec
            else:
                actions.append(rec)
    return actions, summary


def summarize_run(label, agent="rule", directory="."):
    actions, summary = load_log(log_path(label, agent, directory))
    if summary is None:
        return None
    truth = GROUND_TRUTH.get(label)

    freed_bytes = pause_ms = wasted = mutating = 0
    flags = []
    for a in actions:
        act = a["decision"]["action"]
        pre, post = a["pre"]["traced_current_bytes"], a["post"]["traced_current_bytes"]
        if act in MUTATING:
            mutating += 1
            freed = pre - post
            freed_bytes += freed
            pause_ms += a["result"].get("duration_ms", 0.0)
            if pre and freed / pre < FUTILE_RATIO:
                wasted += 1
        elif act.startswith("flag_"):
            flags.append(act)
        elif act == "evict_cache":
            # Count evictions as correct identification for unbounded_retention
            flags.append("flag_leak")

    correct_flag = ((truth == "unbounded_retention" and
                     ("flag_leak" in flags or "evict_cache" in [a["decision"]["action"] for a in actions])) or
                    (truth == "cycle_garbage" and ("flag_cycle" in flags or
                                                   mutating > wasted)))
    return {
        "label": label, "agent": agent, "ground_truth": truth,
        "observations": summary["total_observations"],
        "mutating_actions": mutating,
        "wasted_actions": wasted,
        "false_positive": truth == "healthy" and mutating > 0,
        "missed_leak": truth == "unbounded_retention" and not correct_flag,
        "correct_flag": correct_flag,
        "bytes_freed": freed_bytes,
        "total_pause_ms": round(pause_ms, 3),
        "bytes_freed_per_ms": round(freed_bytes / pause_ms, 1) if pause_ms else None,
        "llm": {k: summary[k] for k in ("llm_calls", "fallbacks", "guard_overrides",
                                        "avg_llm_latency_ms") if k in summary} or None,
    }


def compare(labels=None, directory="."):
    labels = labels or list(GROUND_TRUTH)
    rows = []
    for label in labels:
        for agent in ("rule", "llm"):
            r = summarize_run(label, agent, directory)
            if r:
                rows.append(r)
    return rows


def print_table(rows):
    cols = ["label", "agent", "mutating_actions", "wasted_actions", "false_positive",
            "missed_leak", "correct_flag", "bytes_freed", "total_pause_ms"]
    widths = [max(len(c), *(len(str(r[c])) for r in rows)) for c in cols]
    print("  ".join(c.ljust(w) for c, w in zip(cols, widths)))
    for r in rows:
        print("  ".join(str(r[c]).ljust(w) for c, w in zip(cols, widths)))


def narrate(rows, chat_fn):
    """LLM-written performance analysis of the table."""
    system = ("You analyse benchmark results of a GC-management agent comparing a rule-based "
              "agent with an LLM agent. Use ONLY the numbers given. Say where each agent did "
              "better or worse, point out wasted actions and missed leaks, and state any "
              "caveat about sample size. Max 150 words, plain text.")
    return chat_fn(system, json.dumps(rows))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()
    rows = compare()
    if not rows:
        raise SystemExit("No logs found. Run `python main.py --agent rule` and `--agent llm` first.")
    print_table(rows)
    if args.report:
        from agent.llm_agent import make_groq_chat
        print("\n" + narrate(rows, make_groq_chat()))
