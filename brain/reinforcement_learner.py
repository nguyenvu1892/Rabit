from __future__ import annotations
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from brain.weight_store import WeightStore

class ReinforcementLearner:
    def __init__(self, weight_store: Optional["WeightStore"] = None):
        if weight_store is None:
            try:
                from brain.weight_store import WeightStore as _WeightStore
            except Exception as e:
                # IMPORTANT: đừng nuốt lỗi -> phải show nguyên nhân thật (circular import, syntax error, etc.)
                raise ImportError("Failed to import WeightStore in ReinforcementLearner (possible circular import).") from e

            weight_store = _WeightStore()

        self.weights = weight_store

    def update(self, snapshot: Dict[str, Any], outcome: Dict[str, Any], reward: float) -> None:
        expert = (
            snapshot.get("expert")
            or (snapshot.get("decision") or {}).get("expert")
            or outcome.get("expert")
            or "UNKNOWN_EXPERT"
        )
        self.weights.update("expert", str(expert), float(reward))


    def learn(self, snapshot: Dict[str, Any], outcome: Dict[str, Any], reward: float = 0.0) -> None:
        self.update(snapshot, outcome, float(reward))
