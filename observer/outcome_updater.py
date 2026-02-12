# observer/outcome_updater.py
from __future__ import annotations

from typing import Any, Dict, Optional

from brain.trade_memory import TradeMemory


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


class OutcomeUpdater:
    """
    Consumes trade outcomes and updates:
    - ReinforcementLearner (if provided)
    - TradeMemory (if provided)
    - WeightStore (expert-regime)
    - MetaController (regime threshold learning)
    """

    def __init__(
        self,
        learner: Any = None,
        trade_memory: Optional[TradeMemory] = None,
        weight_store: Optional[Any] = None,
        meta_controller: Optional[Any] = None,
        weights_path: Optional[str] = None,
        autosave: bool = True,
        **kwargs,  # IMPORTANT: compat across versions
    ) -> None:
        self.learner = learner
        self.trade_memory = trade_memory
        self.weight_store = weight_store
        self.meta_controller = meta_controller
        self.weights_path = weights_path
        self.autosave = bool(autosave)

        self._updates = 0

        # If WeightStore passed without path but weights_path provided
        try:
            if (
                self.weight_store is not None
                and getattr(self.weight_store, "path", None) is None
                and self.weights_path
            ):
                self.weight_store.path = self.weights_path
        except Exception:
            pass

    def on_outcome(self, snapshot: Dict[str, Any]) -> None:
        self._updates += 1

        # 1) learner update (best-effort)
        try:
            if self.learner is not None and hasattr(self.learner, "update_from_snapshot"):
                self.learner.update_from_snapshot(snapshot)
        except Exception:
            pass

        # 2) trade_memory append (best-effort)
        try:
            if self.trade_memory is not None and hasattr(self.trade_memory, "append"):
                self.trade_memory.append(snapshot)
        except Exception:
            pass

        # 3) derive expert/regime
        expert, regime = self._extract_expert_regime(snapshot)

        # 4) reward
        reward = self._reward_from_snapshot(snapshot)

        # optional penalty for forced exploration trades
        forced = bool(snapshot.get("forced", False))
        if forced:
            reward *= 0.50

        # 5) meta_controller update (regime-only)
        try:
            if self.meta_controller is not None and hasattr(self.meta_controller, "on_outcome"):
                self.meta_controller.on_outcome(regime, reward)
        except Exception:
            pass

        # 6) weight_store update (expert-regime)
        if self.weight_store is None:
            return
        if not expert or not regime:
            return

        try:
            self.weight_store.update(expert, regime, reward, autosave=self.autosave, log=False)
        except Exception:
            return

        # optional explicit save
        if self.autosave and self.weights_path:
            try:
                self.weight_store.save(self.weights_path)
            except Exception:
                pass

    # ------------------------
    # Helpers
    # ------------------------
    def _extract_expert_regime(self, snapshot: Dict[str, Any]) -> tuple[Optional[str], Optional[str]]:
        risk_cfg = snapshot.get("risk_cfg") or {}
        meta = snapshot.get("meta") or {}

        expert = risk_cfg.get("expert") or meta.get("expert") or meta.get("expert_name")
        regime = risk_cfg.get("regime") or meta.get("regime")

        if expert is not None:
            expert = str(expert)
        if regime is not None:
            regime = str(regime)

        return expert, regime

    def _reward_from_snapshot(self, snapshot: Dict[str, Any]) -> float:
        pnl = snapshot.get("pnl", None)
        if pnl is not None:
            pnl_f = _safe_float(pnl, 0.0)
            scale = _safe_float(snapshot.get("reward_scale", 1.0), 1.0)
            if scale <= 0:
                scale = 1.0
            mag = min(1.0, abs(pnl_f) / scale)
            if pnl_f > 0:
                return 1.0 * mag
            if pnl_f < 0:
                return -1.0 * mag
            return 0.0

        win = snapshot.get("win", None)
        if isinstance(win, bool):
            return 1.0 if win else -1.0

        outcome = snapshot.get("outcome", None)
        if isinstance(outcome, bool):
            return 1.0 if outcome else -1.0

        return 0.0
