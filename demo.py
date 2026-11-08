"""Reproduces the demo video's real output: crash mid-run, resume from checkpoint."""
from stateagent.checkpoint import CheckpointStore
from stateagent.runner import run_checkpointed
from teval.schema import Step, ToolCall

PLAN = [
    Step(kind="thought", content="Need date first, then search", tokens=30),
    Step(kind="tool_call", content="", tokens=25, tool_call=ToolCall(name="calendar.today", arguments={})),
    Step(kind="tool_call", content="", tokens=60, tool_call=ToolCall(name="flights.search", arguments={"from": "BLR", "to": "DEL"})),
    Step(kind="answer", content="Cheapest: IndiGo 06:10 INR 4,320", tokens=28),
]
calls = {"n": 0}
def policy(task, steps):
    calls["n"] += 1
    i = len(steps)
    if i >= len(PLAN):
        return None
    if calls["n"] == 4:
        raise RuntimeError("SIMULATED CRASH: worker killed (SIGKILL)")
    return PLAN[i]

if __name__ == "__main__":
    store = CheckpointStore("/tmp/sa-demo")
    try:
        run_checkpointed(store, "booking-run", "Book cheapest BLR-DEL flight", policy)
    except RuntimeError as e:
        print("crash:", e)
    st = store.load("booking-run")
    print(f"checkpoint on disk: {len(st.steps)}/4 steps committed, done={st.done}")
    st2 = run_checkpointed(store, "booking-run", "Book cheapest BLR-DEL flight", policy)
    print(f"resumed: completed {len(st2.steps)}/4 steps, done={st2.done}")
    print("final_answer:", st2.final_answer)
    print("tokens_used:", st2.tokens_used)
