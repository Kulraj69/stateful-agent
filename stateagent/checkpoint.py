"""Durable agent state: atomic JSON snapshots + resume.

A checkpoint captures everything a run needs to continue: task, completed
steps, partial answer, budget spent. Writes are atomic (tmp + rename) so a
crash mid-write never leaves a torn checkpoint.
"""

import dataclasses
import json
import os
import tempfile
from dataclasses import dataclass, field


@dataclass
class AgentState:
    traj_id: str
    task: str
    steps: list = field(default_factory=list)
    final_answer: str = ""
    tokens_used: int = 0
    done: bool = False
    metadata: dict = field(default_factory=dict)


class CheckpointStore:
    def __init__(self, root: str):
        self.root = root
        os.makedirs(root, exist_ok=True)

    def _path(self, traj_id: str) -> str:
        safe = traj_id.replace("/", "_").replace("..", "_")
        return os.path.join(self.root, f"{safe}.json")

    def save(self, state: AgentState) -> str:
        path = self._path(state.traj_id)
        fd, tmp = tempfile.mkstemp(dir=self.root, suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(dataclasses.asdict(state), f, indent=2)
        os.replace(tmp, path)  # atomic on POSIX
        return path

    def load(self, traj_id: str) -> AgentState | None:
        path = self._path(traj_id)
        if not os.path.exists(path):
            return None
        with open(path, encoding="utf-8") as f:
            return AgentState(**json.load(f))

    def list(self) -> list:
        return sorted(f[:-5] for f in os.listdir(self.root) if f.endswith(".json"))
