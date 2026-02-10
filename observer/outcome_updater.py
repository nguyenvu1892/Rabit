from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class OutcomeEvent:
    """A normalized outcome event passed from ShadowRunner."""
    entry_step: int
    exit_step: int
    pnl: float
    win: bool
    meta: Dict[str, Any]


class OutcomeUpdater:
    """
    5.0.8.1
    - Update weights (expert/regime) based on trade outcomes.
    - Keep backward compatible: learner can be None, weight_store can be None.
    """

    def __init__(self, learner: Any = None, weight_store: Any = None) -> None:
        self.learner = learner
        self.weight_store = weight_store

        # Safe default hyperparams (conservative)
        self.lr = 0.05          # learning rate
        self.pnl_clip = 3.0     # clip pnl impact (in "R-like" units; we use sign+small magnitude)
        self.reward_win = 1.0
        self.reward_loss = -1.0

    def on_outcome(self, payload: Dict[str, Any]) -> None:
        """
        payload is typically what Journal logs for an outcome/trade.
        We normalize then update learner + weight_store.
        """
        ev = self._normalize(payload)

        # 1) Update RL learner (if present) - keep as-is
        if self.learner is not None:
            try:
                # If your learner has a known API, keep it. Otherwise ignore safely.
                fn = getattr(self.learner, "on_outcome", None)
                if callable(fn):
                    fn(payload)
            except Exception:
                pass

        # 2) Update weights (core 5.0.8.1)
        self._update_weights(ev)

    # --------------------------
    # internals
    # --------------------------
    def _normalize(self, payload: Dict[str, Any]) -> OutcomeEvent:
        entry_step = int(payload.get("entry_step", payload.get("step", 0)))
        exit_step = int(payload.get("exit_step", entry_step))
        pnl = float(payload.get("pnl", 0.0))
        win = bool(payload.get("win", pnl > 0))

        meta: Dict[str, Any] = {}
        # payload may contain risk/meta nested differently depending on your pipeline
        if isinstance(payload.get("meta"), dict):
            meta.update(payload["meta"])
        if isinstance(payload.get("risk"), dict):
            # risk often carries expert/regime
            meta.setdefault("risk", payload["risk"])

        return OutcomeEvent(
            entry_step=entry_step,
            exit_step=exit_step,
            pnl=pnl,
            win=win,
            meta=meta,
        )

    def _extract_expert_regime(self, ev: OutcomeEvent) -> tuple[str, str]:
        expert = "UNKNOWN_EXPERT"
        regime = "UNKNOWN"

        # Prefer explicit meta keys
        if "expert" in ev.meta:
            expert = str(ev.meta.get("expert") or expert)
        if "regime" in ev.meta:
            regime = str(ev.meta.get("regime") or regime)

        # Fallback: risk dict
        risk = ev.meta.get("risk")
        if isinstance(risk, dict):
            expert = str(risk.get("expert") or expert)
            regime = str(risk.get("regime") or regime)

        return expert, regime

    def _reward(self, ev: OutcomeEvent) -> float:
        # Base reward by win/loss
        r = self.reward_win if ev.win else self.reward_loss

        # Add SMALL pnl influence but clipped (avoid exploding weights)
        pnl = max(-self.pnl_clip, min(self.pnl_clip, ev.pnl))
        # Normalize pnl contribution to [-0.5..0.5] roughly
        pnl_part = 0.5 * (pnl / self.pnl_clip) if self.pnl_clip > 0 else 0.0

        return float(r + pnl_part)

    def _update_weights(self, ev: OutcomeEvent) -> None:
        if self.weight_store is None:
            return

        expert, regime = self._extract_expert_regime(ev)
        reward = self._reward(ev)

        try:
            # WeightStore.update(expert, regime, reward, lr=?)
            upd = getattr(self.weight_store, "update", None)
            if callable(upd):
                try:
                    upd(expert, regime, reward, lr=self.lr)
                except TypeError:
                    # Backward compat: old signature update(expert, reward, lr=?)
                    upd(expert, reward, lr=self.lr)

            # Persist
            save = getattr(self.weight_store, "save", None)
            if callable(save):
                save()
        except Exception:
            # Never break sim loop
            pass
