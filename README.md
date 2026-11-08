# stateful-agent

Stateful long-running agent - project #5 from the TOP 1% AI projects list.
Durable, checkpointed agent runs that survive crashes and respect budgets.

## What it does

- **checkpoint**: atomic JSON snapshots of the full agent state (task, steps,
  partial answer, tokens spent). Writes use tmp-file + rename so a crash
  mid-write never leaves a torn checkpoint; ids are path-sanitized.
- **runner**: drives any step-policy callable, persisting after every
  committed step and enforcing step/token budgets. A crashed run resumes
  exactly where it stopped - no repeated work, no lost progress.

## Why

Long-running agents fail in boring ways: timeouts, OOM kills, restarts. The
difference between a demo and a dependable worker is whether hour 9 of a
10-hour job survives a reboot. This is that reliability layer, deliberately
model-agnostic: the policy can be MiniAgent (see `agent-lab`), a scripted
plan, or a real LLM loop when one is authorized.

## Usage

```python
from stateagent.checkpoint import CheckpointStore
from stateagent.runner import run_checkpointed

store = CheckpointStore("runs/")
state = run_checkpointed(store, "job-7", "summarize the corpus", policy, max_steps=50)
```

## Test

```bash
python -m unittest discover -s tests
```

7 tests cover snapshot round-trips, missing/sanitized ids, run-to-completion,
crash-resume without re-execution, step budgets, and no-rerun of finished
jobs.
