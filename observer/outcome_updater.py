# observer/outcome_updater.py
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from brain.reinforcement_learner import ReinforcementLearner
from brain.trade_memory import TradeMemory

try:
    from brain.weight_store import WeightStore
except Exception:  # pragma: no cover
    WeightStore = None  # type: ignore


class OutcomeUpdater:
    """
    Consumes trade outcomes and updates:
    - ReinforcementLearner (existing behavior)
    - WeightStore (new optional) [5.0.8.1]
    Defensive: missing fields won't crash.
    """

    def __init__(
        self,
        learner: ReinforcementLearner,
        trade_memory: TradeMemory,
        weight_store: Optional["WeightStore"] = None,
        weights_path: Optional[str] = None,
        autosave: bool = True,
    ) -> None:
        self.learner = learner
        self.trade_memory = trade_memory

        self.weight_store = weight_store
        self.weights_path = weights_path
        self.autosave = bool(autosave)

        # Load once if configured
        if self.weight_store is not None and self.weights_path:
            try:
                self.weight_store.path = self.weights_path
                self.weight_store.load_json(self.weights_path)
            except Exception:
                pass

    def _extract_expert_regime(self, outcome: Dict[str, Any]) -> Tuple[str, str]:
        # direct
        expert = outcome.get("expert")
        regime = outcome.get("regime")

        # nested in risk/meta
        if not expert:
            risk = outcome.get("risk") or {}
            if isinstance(risk, dict):
                expert = risk.get("expert") or expert
                regime = risk.get("regime") or regime

        if not expert:
            meta = outcome.get("meta") or {}
            if isinstance(meta, dict):
                expert = meta.get("expert") or expert
                regime = meta.get("regime") or regime

        return str(expert or "UNKNOWN_EXPERT"), str(regime or "UNKNOWN")

    def process_outcome(self, outcome: Dict[str, Any]) -> None:
        # 1) old behavior
        try:
            self.learner.learn_from_outcome(outcome)
        except Exception:
            pass

        # 2) new weight learning
        if self.weight_store is not None:
            try:
                expert, regime = self._extract_expert_regime(outcome)

                win = bool(outcome.get("win", False))
                pnl = float(outcome.get("pnl", 0.0))

                r = self.weight_store.outcome_reward(win=win, pnl=pnl)

                # Update expert weight (main)
                self.weight_store.update("expert", expert, r)

                # Update regime weight (lighter)
                self.weight_store.update("regime", regime, 0.3 * r)

                if self.autosave:
                    self.weight_store.save_json(self.weights_path or self.weight_store.path)
            except Exception:
                pass
