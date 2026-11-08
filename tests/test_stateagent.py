import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.realpath(os.path.join(_HERE, "..", "..", "trajectory-eval")))

from stateagent.checkpoint import AgentState, CheckpointStore
from stateagent.runner import run_checkpointed

from teval.schema import Step, ToolCall


def policy_three_steps(task, steps):
    n = len(steps)
    if n == 0:
        return Step(kind="tool_call", tool_call=ToolCall(name="search", arguments={"q": task}), tokens=10)
    if n == 1:
        return Step(kind="tool_result", content="result for " + task, tokens=0)
    if n == 2:
        return Step(kind="answer", content="done [tool:search]", tokens=5)
    return None


class TestCheckpointStore(unittest.TestCase):
    def test_save_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as td:
            store = CheckpointStore(td)
            st = AgentState(traj_id="r1", task="t", steps=[{"kind": "thought", "content": "x"}],
                            tokens_used=3)
            store.save(st)
            loaded = store.load("r1")
            self.assertEqual(loaded.task, "t")
            self.assertEqual(loaded.tokens_used, 3)

    def test_load_missing_returns_none(self):
        with tempfile.TemporaryDirectory() as td:
            self.assertIsNone(CheckpointStore(td).load("ghost"))

    def test_ids_are_sanitized(self):
        with tempfile.TemporaryDirectory() as td:
            store = CheckpointStore(td)
            store.save(AgentState(traj_id="../evil/../x", task="t"))
            names = store.list()
            self.assertEqual(len(names), 1)
            self.assertNotIn("/", names[0])
            self.assertNotIn("..", names[0])


class TestRunner(unittest.TestCase):
    def test_runs_to_completion(self):
        with tempfile.TemporaryDirectory() as td:
            st = run_checkpointed(CheckpointStore(td), "r1", "task", policy_three_steps)
            self.assertTrue(st.done)
            self.assertEqual(st.final_answer, "done [tool:search]")
            self.assertEqual(st.tokens_used, 15)

    def test_resume_continues_not_restarts(self):
        with tempfile.TemporaryDirectory() as td:
            store = CheckpointStore(td)
            calls = []
            def counting_policy(task, steps):
                calls.append(len(steps))
                if len(calls) == 2:
                    raise RuntimeError("simulated crash")
                return policy_three_steps(task, steps)
            with self.assertRaises(RuntimeError):
                run_checkpointed(store, "r1", "task", counting_policy)
            # resume with the good policy: should not redo step 0
            st = run_checkpointed(store, "r1", "task", policy_three_steps)
            self.assertTrue(st.done)
            self.assertEqual(len(st.steps), 3)

    def test_max_steps_budget(self):
        with tempfile.TemporaryDirectory() as td:
            def forever(task, steps):
                return Step(kind="thought", content="thinking", tokens=1)
            st = run_checkpointed(CheckpointStore(td), "r1", "task", forever, max_steps=3)
            self.assertFalse(st.done)
            self.assertEqual(len(st.steps), 3)

    def test_completed_run_is_not_rerun(self):
        with tempfile.TemporaryDirectory() as td:
            store = CheckpointStore(td)
            run_checkpointed(store, "r1", "task", policy_three_steps)
            def boom(task, steps):
                raise AssertionError("should not be called")
            st = run_checkpointed(store, "r1", "task", boom)
            self.assertTrue(st.done)


if __name__ == "__main__":
    unittest.main()
