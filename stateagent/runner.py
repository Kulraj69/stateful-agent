"""Checkpointed runner: drive any step-callable with durable save/resume.

The wrapped callable decides the next step from (task, steps-so-far); the
runner persists state after every step and enforces a step/token budget, so
a long task survives crashes and restarts exactly where it stopped.
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
for _sib in ("../../trajectory-eval",):
    _p = os.path.realpath(os.path.join(_HERE, _sib))
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.append(_p)

from teval.schema import Step

from .checkpoint import AgentState, CheckpointStore


def run_checkpointed(store: CheckpointStore, traj_id: str, task: str, policy,
                     max_steps: int = 50, max_tokens: int = 100_000) -> AgentState:
    """Run (or resume) a trajectory to completion or budget exhaustion.

    policy: callable(task, steps) -> Step | None; None means the run is done.
    Returns the final AgentState (done=True on natural completion).
    """
    state = store.load(traj_id) or AgentState(traj_id=traj_id, task=task)
    if state.done:
        return state

    while len(state.steps) < max_steps and state.tokens_used < max_tokens:
        step = policy(state.task, [Step.from_dict(s) for s in state.steps])
        if step is None:
            state.done = True
            break
        state.steps.append(dataclasses_asdict_step(step))
        state.tokens_used += step.tokens
        if step.kind == "answer":
            state.final_answer = step.content
            state.done = True
        store.save(state)  # checkpoint after every committed step
    store.save(state)
    return state


def dataclasses_asdict_step(step) -> dict:
    d = {"kind": step.kind, "content": step.content, "tokens": step.tokens}
    if step.tool_call is not None:
        d["tool_call"] = {"name": step.tool_call.name, "arguments": step.tool_call.arguments}
    return d
