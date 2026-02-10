# sim/loop_state.py
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class LoopState:
    idx: int = 0
    run_id: str = ""
    strategy_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"idx": int(self.idx), "run_id": self.run_id, "strategy_hash": self.strategy_hash}

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "LoopState":
        return LoopState(
            idx=int(d.get("idx", 0)),
            run_id=str(d.get("run_id", "")),
            strategy_hash=str(d.get("strategy_hash", "")),
        )


class LoopStateStore:
    def __init__(self, path: str):
        self.path = path

    def load(self) -> Optional[LoopState]:
        if not os.path.exists(self.path):
            return None
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                d = json.load(f)
            if not isinstance(d, dict):
                return None
            return LoopState.from_dict(d)
        except Exception:
            return None

    def save(self, state: LoopState) -> None:
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(state.to_dict(), f, indent=2)
