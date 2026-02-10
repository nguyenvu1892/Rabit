from __future__ import annotations

from typing import Any, Dict, List, Tuple


class ContextMemory:
    """
    Lưu thống kê theo "context key" (hashable) để ContextIntelligence dùng lại.
    """

    def __init__(self) -> None:
        # key -> aggregate
        self.memory: Dict[Tuple[Any, ...], Dict[str, float]] = {}
        # key -> raw history (optional, để debug)
        self.history: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = {}

    @staticmethod
    def _freeze(x: Any) -> Any:
        """Convert object (dict/list/set/tuple) thành dạng hashable để làm key."""
        if isinstance(x, dict):
            return tuple(sorted((k, ContextMemory._freeze(v)) for k, v in x.items()))
        if isinstance(x, (list, tuple)):
            return tuple(ContextMemory._freeze(v) for v in x)
        if isinstance(x, set):
            return tuple(sorted(ContextMemory._freeze(v) for v in x))
        return x

    def build_key(self, trade_features: Dict[str, Any]) -> Tuple[Any, ...]:
        return tuple(sorted((k, self._freeze(v)) for k, v in trade_features.items()))

    def get_stats(self, trade_features: Dict[str, Any]) -> Dict[str, Any]:
        key = self.build_key(trade_features)
        agg = self.memory.get(key)
        if not agg:
            return {"wins": 0, "losses": 0, "samples": 0, "total_outcome": 0.0}

        return {
            "wins": int(agg.get("wins", 0.0)),
            "losses": int(agg.get("losses", 0.0)),
            "samples": int(agg.get("samples", 0.0)),
            "total_outcome": float(agg.get("total_outcome", 0.0)),
        }

    def store(self, snapshot: Dict[str, Any], score: float, risk_cfg: Dict[str, Any], outcome: float) -> None:
        """
        snapshot: trade_features tại thời điểm vào lệnh (đã freeze để làm key)
        outcome : reward/pnl (âm/dương)
        """
        key = self.build_key(snapshot)

        self.history.setdefault(key, []).append(
            {"snapshot": snapshot, "score": float(score), "risk": dict(risk_cfg or {}), "outcome": float(outcome)}
        )

        agg = self.memory.setdefault(key, {"wins": 0.0, "losses": 0.0, "total_outcome": 0.0, "samples": 0.0})
        agg["samples"] += 1.0
        agg["total_outcome"] += float(outcome)
        if outcome > 0:
            agg["wins"] += 1.0
        else:
            agg["losses"] += 1.0

    def get_strength(self, trade_features: Dict[str, Any]) -> float:
        """Strength ~ winrate (0..1). Default 0.5 nếu chưa có dữ liệu."""
        s = self.get_stats(trade_features)
        n = s.get("samples", 0) or 0
        if n <= 0:
            return 0.5
        return float(s.get("wins", 0)) / float(n)
