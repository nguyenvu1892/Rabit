# brain/trade_memory.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict


def _freeze(x: Any) -> Any:
    """Make x hashable (for dict key)."""
    if isinstance(x, dict):
        return tuple(sorted((k, _freeze(v)) for k, v in x.items()))
    if isinstance(x, (list, tuple)):
        return tuple(_freeze(v) for v in x)
    if isinstance(x, set):
        return tuple(sorted(_freeze(v) for v in x))
    return x


@dataclass
class TradeMemory:
    """
    Aggregated memory for outcomes by a snapshot key.
    Each entry:
      {"wins": int, "losses": int, "total_pnl": float, "samples": int}
    """
    memory: Dict[Any, Dict[str, Any]] = field(default_factory=dict)

    def build_key(self, trade_features: Dict[str, Any]) -> Any:
        # Use full trade_features as snapshot key (stable + hashable)
        return _freeze(trade_features)

    # Backward compat: some older code may call _build_key
    def _build_key(self, trade_features: Dict[str, Any]) -> Any:
        return self.build_key(trade_features)

    def get_stats(self, trade_features: Dict[str, Any]) -> Dict[str, Any]:
        # Backward compat: if loaded old data where memory became list, fix it
        if not isinstance(self.memory, dict):
            self.memory = {}

        key = self.build_key(trade_features)
        data = self.memory.get(key)
        if not data or data.get("samples", 0) <= 0:
            return {"avg_pnl": 0.0, "samples": 0, "wins": 0, "losses": 0}

        samples = int(data.get("samples", 0))
        total_pnl = float(data.get("total_pnl", 0.0))
        wins = int(data.get("wins", 0))
        losses = int(data.get("losses", 0))

        avg_pnl = total_pnl / samples if samples > 0 else 0.0
        return {
            "avg_pnl": float(avg_pnl),
            "samples": samples,
            "wins": wins,
            "losses": losses,
        }

    def record(self, snapshot: Dict[str, Any], outcome: Dict[str, Any]) -> None:
        if not isinstance(self.memory, dict):
            self.memory = {}

        key = self.build_key(snapshot)

        entry = self.memory.get(key)
        if entry is None:
            entry = {"wins": 0, "losses": 0, "total_pnl": 0.0, "samples": 0}
            self.memory[key] = entry

        pnl = float(outcome.get("pnl", 0.0))
        win = bool(outcome.get("win", pnl > 0))

        entry["samples"] += 1
        entry["total_pnl"] += pnl
        if win:
            entry["wins"] += 1
        else:
            entry["losses"] += 1
