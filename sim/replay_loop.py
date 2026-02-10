# sim/replay_loop.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Sequence, Optional


@dataclass
class ReplayResult:
    steps: int
    decisions: int
    allows: int


class ReplayLoop:
    """
    Minimal SIM loop:
      - maintains a rolling candle window
      - computes features via FeatureSet
      - calls DecisionEngine.evaluate_trade(features)
    """

    def __init__(self, feature_set, decision_engine, window: int = 50):
        self.feature_set = feature_set
        self.decision_engine = decision_engine
        self.window = int(window)

        self._candles: List[Dict[str, Any]] = []

    def step(self, candle: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        self._candles.append(candle)
        if len(self._candles) < 2:
            return None

        if len(self._candles) > self.window:
            self._candles = self._candles[-self.window :]

        trade_features = self.feature_set.compute(self._candles)
        allow, score, risk = self.decision_engine.evaluate_trade(trade_features)

        return {"allow": allow, "score": score, "risk": risk, "features": trade_features}

    def run(self, candles: Sequence[Dict[str, Any]]) -> ReplayResult:
        steps = 0
        decisions = 0
        allows = 0
        for c in candles:
            steps += 1
            out = self.step(c)
            if out is None:
                continue
            decisions += 1
            if out["allow"]:
                allows += 1
        return ReplayResult(steps=steps, decisions=decisions, allows=allows)
