# observer/outcome_updater.py
from __future__ import annotations

from typing import Any, Dict, Optional

from brain.trade_memory import TradeMemory


class OutcomeUpdater:
    """
    Consumes trade outcomes and updates:
      - ReinforcementLearner (existing behavior)
      - WeightStore (expert-regime pair updates)

    Snapshot expected (best-effort):
      - pnl: float
      - win: bool (optional)
      - forced: bool (optional)
      - atr: float (optional)
      - conf: float in [0..1] (optional)
      - risk_cfg: {"expert": str, "regime": str} (preferred)
      - meta: may contain "expert"/"regime"
    """

    def __init__(
        self,
        learner: Any,
        trade_memory: TradeMemory,
        weight_store: Optional[Any] = None,
        weights_path: Optional[str] = None,
        autosave: bool = True,
        save_every: int = 25,          # <— avoid IO each trade
        reward_scale: float = 1.0,     # pnl scaling
    ) -> None:
        self.learner = learner
        self.trade_memory = trade_memory
        self.weight_store = weight_store
        self.weights_path = weights_path
        self.autosave = bool(autosave)
        self.save_every = int(save_every)
        self.reward_scale = float(reward_scale)

        self._updates = 0

        # if WeightStore passed without path but weights_path provided
        try:
            if self.weight_store is not None and getattr(self.weight_store, "path", None) is None and self.weights_path:
                self.weight_store.path = self.weights_path
        except Exception:
            pass

    def on_outcome(self, snapshot: Dict[str, Any]) -> None:
        self._updates += 1

        # 1) keep old learner update (best-effort)
        try:
            if self.learner is not None and hasattr(self.learner, "update_from_snapshot"):
                self.learner.update_from_snapshot(snapshot)
        except Exception:
            pass

        # 2) WeightStore update
        if self.weight_store is None:
            return

        expert, regime = self._extract_expert_regime(snapshot)
        if not expert or not regime:
            return

        reward = self._reward_from_snapshot(snapshot)

        # confidence scaling
        conf = snapshot.get("conf", None)
        try:
            conf_f = float(conf) if conf is not None else 1.0
        except Exception:
            conf_f = 1.0
        conf_f = max(0.0, min(1.0, conf_f))

        # optional penalty for forced exploration trades
        forced = bool(snapshot.get("forced", False))
        if forced:
            reward *= 0.50

        # optional scaling by ATR (if present)
        atr = snapshot.get("atr", None)
        if atr is not None:
            try:
                atr = float(atr)
                if atr > 0:
                    reward = reward / (1.0 + 0.1 * atr)
            except Exception:
                pass

        # update expert-regime
        try:
            # WeightStore signature supports conf/autosave/log (ours), but also tolerant if not
            if hasattr(self.weight_store, "update"):
                self.weight_store.update(
                    expert,
                    regime,
                    reward,
                    conf=conf_f,
                    autosave=self.autosave,
                    log=False,
                )
        except TypeError:
            # fallback for older signature: update(bucket,key,delta, autosave=?, log=?)
            try:
                self.weight_store.update(expert, regime, reward, autosave=self.autosave, log=False)
            except Exception:
                return
        except Exception:
            return

        # explicit periodic save (safe)
        if self.autosave and self.weights_path and (self._updates % max(1, self.save_every) == 0):
            try:
                if hasattr(self.weight_store, "save"):
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
                scale = float(snapshot.get("reward_scale", self.reward_scale) or self.reward_scale)
                if scale <= 0:
                    scale = 1.0
                mag = min(1.0, abs(pnl_f) / scale)
                if pnl_f > 0:
                    return 1.0 * mag
                if pnl_f < 0:
                    return -1.0 * mag
                return 0.0
            except Exception:
                pass

        # fallback to win/loss
        win = snapshot.get("win", None)
        if win is None:
            win = snapshot.get("outcome", None)
        if isinstance(win, bool):
            return 1.0 if win else -1.0
        return 0.0
