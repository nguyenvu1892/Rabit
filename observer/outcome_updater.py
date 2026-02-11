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
    - WeightStore (optional) [5.0.8.1+]

    5.0.8.5 (stabilize):
    - penalize/attenuate reward for forced trades
    - keep regime updates weaker
    - periodic save/decay/log only (avoid IO spam)
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

        self._outcome_count = 0
        self.save_every = 200
        self.decay_every = 200
        self.decay_rate = 0.001
        self.log_every = 200

        if self.weight_store is not None and self.weights_path:
            try:
                self.weight_store.path = self.weights_path
                self.weight_store.load_json(self.weights_path)
            except Exception:
                pass

    def _extract_expert_regime(self, outcome: Dict[str, Any]) -> Tuple[str, str]:
        expert = outcome.get("expert")
        regime = outcome.get("regime")

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

    def _is_forced(self, outcome: Dict[str, Any]) -> bool:
        if bool(outcome.get("forced", False)):
            return True
        meta = outcome.get("meta") or {}
        if isinstance(meta, dict) and bool(meta.get("forced", False)):
            return True
        return False

    def process_outcome(self, outcome: Dict[str, Any]) -> None:
        # 1) keep old behavior (do not break)
        try:
            self.learner.learn_from_outcome(outcome)
        except Exception:
            pass

        # 2) weight learning
        if self.weight_store is not None:
            try:
                expert, regime = self._extract_expert_regime(outcome)
                win = bool(outcome.get("win", False))
                pnl = float(outcome.get("pnl", 0.0))
                r = float(self.weight_store.outcome_reward(win=win, pnl=pnl))

                # forced trades are "training wheels" => reduce trust signal
                if self._is_forced(outcome):
                    r *= 0.5

                # expert is primary
                self.weight_store.update("expert", expert, r)
                # regime is weaker
                self.weight_store.update("regime", regime, 0.25 * r)

            except Exception:
                pass

        # 3) periodic stabilize actions
        self._outcome_count += 1

        if self.weight_store is not None:
            if self.decay_every > 0 and (self._outcome_count % self.decay_every == 0):
                try:
                    self.weight_store.decay_toward_one("expert", rate=self.decay_rate)
                    self.weight_store.decay_toward_one("regime", rate=self.decay_rate)
                except Exception:
                    pass

            if self.autosave and self.save_every > 0 and (self._outcome_count % self.save_every == 0):
                try:
                    self.weight_store.save_json(self.weights_path or self.weight_store.path)  # type: ignore[arg-type]
                except Exception:
                    pass

            if self.log_every > 0 and (self._outcome_count % self.log_every == 0):
                try:
                    top = self.weight_store.topk("expert", 5)
                    bot = self.weight_store.bottomk("expert", 5)
                    print(f"[weights] outcomes={self._outcome_count} top_expert={top} bottom_expert={bot}")
                except Exception:
                    pass
