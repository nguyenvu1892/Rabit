# brain/reinforcement_learner.py
from __future__ import annotations
from typing import Any, Dict, Optional

from brain.weight_store import WeightStore


class ReinforcementLearner:
    """
    Minimal learner: update expert weights from (snapshot, outcome, reward).
    Compatible with older code that calls .learn(...) too.
    """
    def __init__(self, weight_store: Optional[WeightStore] = None):
        self.weights = weight_store or WeightStore()

    def update(self, snapshot: Dict[str, Any], outcome: Dict[str, Any], reward: float) -> None:
        expert = (
            snapshot.get("expert")
            or (snapshot.get("decision") or {}).get("expert")
            or outcome.get("expert")
            or "UNKNOWN_EXPERT"
        )
        self.weights.update(str(expert), float(reward))

    # backward-compatible alias
    def learn(self, snapshot: Dict[str, Any], outcome: Dict[str, Any], reward: float = 0.0) -> None:
        self.update(snapshot, outcome, float(reward))
