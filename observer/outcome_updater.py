# observer/outcome_updater.py
from __future__ import annotations

from typing import Any, Dict, Optional

from brain.trade_memory import TradeMemory


class OutcomeUpdater:
    """
    Consumes trade outcomes and updates:
      - ReinforcementLearner (existing behavior)
      - WeightStore (5.0.8.x), now with 5.0.8.9: expert-regime pair updates
    """

    def __init__(
        self,
        learner: Any,
        trade_memory: TradeMemory,
        weight_store: Optional[Any] = None,
        weights_path: Optional[str] = None,
        autosave: bool = True,
    ) -> None:
        self.learner = learner
        self.trade_memory = trade_memory
        self.weight_store = weight_store
        self.weights_path = weights_path
        self.autosave = bool(autosave)

        self._updates = 0

        # if WeightStore passed without path but weights_path provided
        try:
            if self.weight_store is not None and getattr(self.weight_store, "path", None) is None and self.weights_path:
                self.weight_store.path = self.weights_path
        except Exception:
            pass

    def on_outcome(self, snapshot: Dict[str, Any]) -> None:
        """
        snapshot is expected to include (best effort):
          - pnl (float)
          - win (bool) OR pnl sign
          - risk_cfg: {"expert": "...", "regime": "..."}  (preferred)
          - meta: may contain {"expert": "..."} etc.
          - forced (bool)
        """
        self._updates += 1

        # 1) keep old learner update (best-effort)
        try:
            if self.learner is not None and hasattr(self.learner, "update_from_snapshot"):
                self.learner.update_from_snapshot(snapshot)
        except Exception:
            pass

        # 2) WeightStore update (expert-regime)
        if self.weight_store is None:
            return

        expert, regime = self._extract_expert_regime(snapshot)
        if not expert or not regime:
            return

        reward = self._reward_from_snapshot(snapshot)

        # optional penalty for forced exploration trades
        forced = bool(snapshot.get("forced", False))
        if forced:
            reward *= 0.50

        # optional scaling by ATR (if present)
        # if you later add atr field, this keeps stable (no crash)
        atr = snapshot.get("atr", None)
        if atr is not None:
            try:
                atr = float(atr)
                if atr > 0:
                    # mild scaling to reduce overreaction under high vol
                    reward = reward / (1.0 + 0.1 * atr)
            except Exception:
                pass

        try:
            self.weight_store.update(
                expert,
                regime,
                reward,
                autosave=self.autosave,
                log=False,
            )
        except Exception:
            # do NOT crash the pipeline
            return

        # optional explicit save path
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
        # prefer pnl if exists
        pnl = snapshot.get("pnl", None)
        if pnl is not None:
            try:
                pnl_f = float(pnl)
                # compress to keep stable:
                # sign(pnl) * min(1, abs(pnl)/scale)
                scale = float(snapshot.get("reward_scale", 1.0))
                if scale <= 0:
                    scale = 1.0
                mag = min(1.0, abs(pnl_f) / scale)
                return (1.0 if pnl_f > 0 else (-1.0 if pnl_f < 0 else 0.0)) * mag
            except Exception:
                pass

        # fallback to win/loss
        win = snapshot.get("win", None)
        if win is None:
            # try outcome field
            win = snapshot.get("outcome", None)
        if isinstance(win, bool):
            return 1.0 if win else -1.0

        return 0.0
    print("DEBUG bucket:", bucket_key)
    print("DEBUG weight before:", weight_store.get(bucket, key))
