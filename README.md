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

Also includes a **from-scratch mark-sweep garbage collector**
(`custom_gc/`) as an alternative to real Python `gc` -- see "Custom GC
backend" below. This exists to answer a fair critique of the original
scope: monitoring/reacting to Python's *already-built* collector isn't
the same as *implementing* one. `custom_gc` is a real, from-scratch
implementation of the algorithm, verified to correctly distinguish
reference cycles from live reachable data (see that section for the
actual numbers from a test run).

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
- **Phase VI**: FastAPI service (the dashboard currently reads log files
  directly, so this isn't blocking anything)
- **Reference cycle visualization** via `objgraph`
- Persistent run history across restarts (currently flat JSONL per run;
  SQLite was the suggested upgrade)

## Custom GC backend (`custom_gc/`)

A real mark-sweep collector implemented from scratch: `custom_gc/heap.py`
defines `GCObject` (id, name, outgoing references, a mark flag) and
`Heap` (tracks every live object, a set of roots, and runs mark-then-
sweep collection on demand). `custom_gc/__init__.py` exposes a single
shared `Heap` through module-level functions shaped exactly like the
real `gc` module (`get_count`, `get_stats`, `get_objects`, `collect`,
`set_threshold`) -- this is what lets `Monitor` and `Actuator` work
against it with zero changes to their own logic, just a different module
passed in.

**Why real Python's own `gc` is disabled while this is in use:**
`GCObject` instances are still real Python objects. Two of them
referencing each other is coincidentally *also* a real cycle in
CPython's own object graph, meaning the real interpreter's collector
could silently free it in the background -- which would make our own
`collect()` results meaningless (e.g. "we called collect() and froze 0
objects" because real gc already did it first). `custom_gc/heap.py`
calls `gc.disable()` at import time so every collection in this backend
is attributable to *our* algorithm, not CPython's.

**Run it:**
```bash
python main_custom_gc.py
```

**Verified correctness, from an actual test run:**

| Workload | `collect()` froze | Why that's correct |
|---|---|---|
| `cycle_generator_custom` | 200 objects | Two-object cycles, unreachable from any root -- mark phase never reaches them, sweep removes them |
| `unbounded_cache_custom` | 0 objects | Every entry is reachable via `cache_root` (a root) -- mark phase correctly reaches all of them, so sweep removes nothing, even though memory is genuinely growing |
| `normal_churn_custom` | 102 objects | Previous batch's root was explicitly removed before the next batch started -- correctly reclaimed |

The `unbounded_cache_custom` result (0 objects freed, despite rising
memory) is the important one to point out in a demo -- it's proof the
collector isn't just deleting everything it's shown, it's actually doing
reachability analysis correctly.

**Design notes / things you decide, not fixed here:**
- All three `*_custom.py` workloads share ONE heap (`custom_gc`'s
  module-level singleton, matching how a single real process has one
  heap). `main_custom_gc.py` calls `heap.reset()` between experiments
  so one workload's leftovers don't confound the next one's numbers --
  remove that reset if you specifically want to demo a heap accumulating
  across phases instead.
- This is a single-generation collector (no generational promotion) --
  `get_count()`/`get_stats()` return `gen1`/`gen2` as always-zero purely
  for interface parity with real `gc`, not because generations exist
  here. Whether to actually implement generations is a scope decision,
  not something assumed for you.
- `tracemalloc`/`psutil` numbers in samples still reflect the REAL
  Python process throughout (they always have -- there's no way around
  this without leaving Python), so with `custom_gc` they show the
  overhead of `GCObject` instances themselves, not a separate simulated
  memory space. Worth being able to explain this distinction if asked.
