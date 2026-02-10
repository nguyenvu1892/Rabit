# persistence/state_bundle.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class CoreStateBundle:
    run_id: str
    strategy_hash: str

    rl_weights: Dict[str, Any]
    trade_memory: Any
    session_guard_state: Dict[str, Any]
    extended_state: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "strategy_hash": self.strategy_hash,
            "rl_weights": self.rl_weights,
            "trade_memory": self.trade_memory,
            "session_guard_state": self.session_guard_state,
            "extended_state": self.extended_state,
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "CoreStateBundle":
        return CoreStateBundle(
            run_id=str(d.get("run_id", "")),
            strategy_hash=str(d.get("strategy_hash", "")),
            rl_weights=d.get("rl_weights", {}) or {},
            trade_memory=d.get("trade_memory", None),
            session_guard_state=d.get("session_guard_state", {}) or {},
            extended_state=d.get("extended_state", {}) or {},
        )
