# gc-agent

Agentic memory management / adaptive GC for Python. A monitor watches a
running workload's memory and GC behavior, an agent decides whether to
intervene, and an actuator applies that decision under hard safety bounds
-- then an evaluator-style before/after log lets you check whether the
intervention actually helped.

Currently implemented: **Phases I-III** (workloads, monitoring, analysis)
plus a working **actuator with safety rails**, driven by a **rule-based
stand-in agent** so the whole loop runs end-to-end without needing an LLM
API key yet. The real LLM/LangGraph agent (Phase IV) is designed to slot
in as a drop-in replacement -- see "Extending" below.

## How it works

```
workload (generator, yields periodically)
        |
        v
   Monitor.run()  -- samples gc/tracemalloc/psutil each step
        |
        v
   analysis.analyze()  -- growth rate, gen0 GC frequency
        |
        v
   agent.decide()  -- returns {"action": ..., "reason": ...}
        |
        v
   Actuator.apply()  -- gc.collect() / gc.set_threshold() / flag, bounded
        |
        v
   Actuator.log_event()  -- pre/post snapshot pair -> <label>_log.jsonl
```

### The in-process design

Monitor and workload run in the same process, same thread. Workloads are
generators that `yield` after each unit of work so the monitor gets a
turn to sample in between -- a workload that ran to completion in one
blocking call would give the monitor nothing to observe mid-flight.

### The safety-guard design

When the agent decides on a non-noop action, `Monitor._apply_action()`
sets an `action_in_progress` flag, takes a pre-action snapshot, applies
the action, takes a post-action snapshot, logs the triple, then clears
the flag. Regular sampling is skipped while the flag is up. This is a
single-threaded state guard, not a concurrency lock -- its only job is to
stop an intervention's effects from being misread as ordinary workload
behavior in the logs.

Actual safety rails enforced in code:
- **Cooldown** (`agent/rule_based.py`): won't act again within
  `MIN_SECONDS_BETWEEN_ACTIONS` of the last action.
- **Threshold clamping** (`actuator/actuator.py`): any `tune_threshold`
  request is clamped to `threshold_bounds`, regardless of what the agent
  asks for.

## Project layout

```
gc-agent/
├── workloads/
│   ├── cycle_generator.py   # continuously creates reference cycles
│   ├── unbounded_cache.py   # dict that grows forever, no eviction
│   └── normal_churn.py      # baseline: create+drop, no cycles/leaks
├── monitor/monitor.py        # sampling loop + action-isolation guard
├── analysis/analysis.py      # growth rate, gc frequency signals
├── agent/rule_based.py       # rule-based decision stand-in
├── actuator/actuator.py      # applies decisions, enforces bounds, logs
├── main.py                   # runs all three workloads as experiments
├── requirements.txt
├── .env.example               # for the future LLM API key
└── .gitignore
```

## Running it

```bash
pip install -r requirements.txt
python main.py
```

This runs all three workloads (100 steps each) and writes
`cycle_generator_log.jsonl`, `unbounded_cache_log.jsonl`, and
`normal_churn_log.jsonl` -- each line is one actuator action with its
pre/post memory snapshot and result.

You can also run any single workload standalone to sanity-check it:

```bash
python workloads/cycle_generator.py
```

### What a healthy result looks like

- `normal_churn` should trigger **zero or very few** actuator actions.
- `cycle_generator` and `unbounded_cache` should trigger **some**.
- For `unbounded_cache` specifically, don't be surprised if a `collect()`
  call frees few or zero objects on later actions -- the cache's entries
  are still *referenced* (live), not garbage, so collection legitimately
  can't free them. That's a real, useful finding, not a bug: it's
  evidence that different leak types need different remedies (eviction
  policy vs. cycle collection).

## Tuning

`agent/rule_based.py` has two constants that are placeholder guesses, not
measured values:

```python
GROWTH_THRESHOLD_BYTES_PER_SEC = 50_000
MIN_SECONDS_BETWEEN_ACTIONS = 5
```

Run `main.py`, look at the `growth_rate_bytes_per_sec` values implied by
the logs (or add a print in `analysis.analyze()` temporarily), and pick
thresholds that actually separate your three workloads instead of firing
on all of them or none of them.

`tune_threshold` is implemented in the actuator but never emitted by
`rule_based.py` yet -- deciding when threshold-tuning is the right call
versus a plain `collect()` is an open design decision.

## Extending: swapping in the real LLM agent (Phase IV)

Write a new module, e.g. `agent/llm_agent.py`, with the same signature
and return schema as `agent/rule_based.py`:

```python
def decide(samples, analysis_fn):
    ...
    return {"action": "collect" | "tune_threshold" | "flag_cycle" | "noop",
            "reason": "<string>"}
```

Then in `main.py`, change one import:

```python
from agent.rule_based import decide      # before
from agent.llm_agent import decide       # after
```

Nothing else needs to change -- the monitor, actuator, and logging don't
know or care whether the decision came from a rule or an LLM call.
Recommended for that module: strict JSON-schema parsing with retry logic,
and a fallback to the rule-based `decide()` if the LLM call fails or the
response doesn't parse, so a flaky API doesn't break a live demo.

## Not yet built

- **Phase IV**: real LangGraph + LLM agent (see above)
- **Phase V evaluator**: the raw before/after data is already logged;
  a dedicated module to summarize/compare across runs is still open
- **Phase VI**: FastAPI service + Streamlit dashboard
- **Reference cycle visualization** via `objgraph`
- Persistent run history across restarts (currently flat JSONL per run;
  SQLite was the suggested upgrade)
