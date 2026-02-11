# observer/outcome_updater.py
from __future__ import annotations

from typing import Any, Dict, Optional

from brain.reinforcement_learner import ReinforcementLearner
from brain.trade_memory import TradeMemory

try:
    # optional dependency for 5.0.8.1
    from brain.weight_store import WeightStore
except Exception:  # pragma: no cover
    WeightStore = None  # type: ignore


class OutcomeUpdater:
    """
    Consumes trade outcomes and updates:
      - ReinforcementLearner (existing behavior)
      - WeightStore (new, optional)  [5.0.8.1]

    This is intentionally defensive: if fields are missing, it won't crash.
    """

    def __init__(
        self,
        learner: ReinforcementLearner,
        trade_memory: TradeMemory,
        weight_store: Optional["WeightStore"] = None,
        weights_path: Optional[str] = None,
        autosave: bool = True,
    ):
        self.learner = learner
        self.trade_memory = trade_memory

        self.weight_store = weight_store
        self.weights_path = weights_path
        self.autosave = bool(autosave)

        # If weight_store is provided and path exists, load once.
        if self.weight_store is not None and self.weights_path:
            try:
                self.weight_store.path = self.weights_path
                self.weight_store.load_json(self.weights_path)
            except Exception:
                pass

    def _extract_expert_regime(self, outcome: Dict[str, Any]) -> tuple[Optional[str], Optional[str]]:
        # Try common locations
        expert = outcome.get("expert")
        regime = outcome.get("regime")

        # Sometimes it is inside "risk" or "meta"
        if not expert:
            risk = outcome.get("risk") or {}
            if isinstance(risk, dict):
                expert = expert or risk.get("expert")
                regime = regime or risk.get("regime")

        if not expert:
            meta = outcome.get("meta") or {}
            if isinstance(meta, dict):
                expert = expert or meta.get("expert")
                regime = regime or meta.get("regime")

        if expert is not None:
            expert = str(expert)
        if regime is not None:
            regime = str(regime)

        return expert, regime

    def process_outcome(self, outcome: Dict[str, Any]) -> None:
        """
        outcome is expected to have at least:
          - "pnl" (float) and/or "win" (bool)
        but we handle missing fields.
        """
        # Existing learning behavior
        try:
            self.learner.learn_from_outcome(outcome)
        except Exception:
            pass

        # Weight learning (optional)
        if self.weight_store is not None:
            expert, regime = self._extract_expert_regime(outcome)
            if expert:
                win = outcome.get("win")
                pnl = outcome.get("pnl")

                # Normalize types
                if isinstance(win, str):
                    if win.lower() in ("true", "1", "yes", "y"):
                        win = True
                    elif win.lower() in ("false", "0", "no", "n"):
                        win = False
                    else:
                        win = None

                try:
                    pnl_f = float(pnl) if pnl is not None else None
                except Exception:
                    pnl_f = None

                try:
                    new_w = self.weight_store.update_from_outcome(expert, win=win, pnl=pnl_f, regime=regime)
                    # Attach to outcome for logging/journal downstream (non-breaking)
                    meta = outcome.get("meta")
                    if not isinstance(meta, dict):
                        meta = {}
                        outcome["meta"] = meta
                    meta["weight_after"] = float(new_w)

                    if self.autosave and (self.weights_path or getattr(self.weight_store, "path", None)):
                        self.weight_store.save_json(self.weights_path)
                except Exception:
                    pass
