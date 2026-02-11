# observer/outcome_updater.py
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Optional

from brain.trade_memory import TradeMemory

if TYPE_CHECKING:
    from brain.reinforcement_learner import ReinforcementLearner
    from brain.weight_store import WeightStore


class OutcomeUpdater:
    """
    Consume trade outcomes and update:
      - ReinforcementLearner (existing)
      - WeightStore (5.0.8.x)

    IMPORTANT:
      - No runtime top-level import WeightStore to avoid circular imports.
      - Pass weight_store from shadow_run, or lazily import inside __init__.
    """

    def __init__(
        self,
        learner,
        trade_memory,
        weight_store=None,
        weights_path=None,
        autosave=True,
        # --- new (5.0.8.7) ---
        stabilize_every: int = 25,
        decay_rate: float = 0.002,
        normalize_every: int = 25,
        summary_every: int = 50,
        summary_k: int = 3,
    ):
        self.learner = learner
        self.trade_memory = trade_memory
        self.weight_store = weight_store
        self.weights_path = weights_path
        self.autosave = autosave

        self.stabilize_every = int(stabilize_every)
        self.decay_rate = float(decay_rate)
        self.normalize_every = int(normalize_every)
        self.summary_every = int(summary_every)
        self.summary_k = int(summary_k)

        self._outcome_count = 0

        # lazy init WeightStore to avoid circular import
        self.weight_store = weight_store
        if self.weight_store and (self._outcome_count % self.stabilize_every == 0):
            # stabilize EXPERT bucket
            s1 = self.weight_store.stabilize_bucket(
                "expert",
                min_w=self.w_min,
                max_w=self.w_max,
                decay_rate=self.stabilize_decay,
                target_mean=self.stabilize_target_mean,
            )
            # stabilize REGIME bucket
            s2 = self.weight_store.stabilize_bucket(
                "regime",
                min_w=self.w_min,
                max_w=self.w_max,
                decay_rate=self.stabilize_decay,
                target_mean=self.stabilize_target_mean,
            )

            # log top/bottom (nếu bro đã có log top/bottom ở đoạn summary thì giữ nguyên, còn chưa có thì add):
            top_exp = self.weight_store.topk("expert", k=5)
            bot_exp = self.weight_store.bottomk("expert", k=5)
            top_reg = self.weight_store.topk("regime", k=5)
            bot_reg = self.weight_store.bottomk("regime", k=5)

            print(f"[WEIGHT][STABILIZE] expert {s1} | regime {s2}")
            print(f"[WEIGHT][TOP] expert={top_exp} regime={top_reg}")
            print(f"[WEIGHT][BOT] expert={bot_exp} regime={bot_reg}")

        self.weights_path = weights_path
        self.autosave = bool(autosave)
        self.stabilize_every = 200          # hoặc 100/200 tuỳ bro, 200 khá ổn
        self.stabilize_decay = 0.02         # decay nhẹ về 1.0 (anti-overfit)
        self.stabilize_target_mean = 1.0
        self.w_min = 0.2
        self.w_max = 5.0

        self.forced_penalty = float(forced_penalty)
        self.lr = float(lr)
        self.clamp_min = float(clamp_min)
        self.clamp_max = float(clamp_max)

    def _extract(self, outcome: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize outcome payload. Keep backward compatibility.
        Expect keys may include:
          - pnl (float)
          - win (bool)
          - expert (str)
          - regime (str)
          - forced (bool)
          - atr (float) optional
        """
        pnl = float(outcome.get("pnl", 0.0) or 0.0)
        win = bool(outcome.get("win", pnl > 0))
        expert = str(outcome.get("expert", outcome.get("expert_name", "UNKNOWN_EXPERT")))
        regime = str(outcome.get("regime", "global"))
        forced = bool(outcome.get("forced", False))
        atr = outcome.get("atr", None)
        atr_v = float(atr) if atr is not None else None
        return {"pnl": pnl, "win": win, "expert": expert, "regime": regime, "forced": forced, "atr": atr_v}

    def on_outcome(self, outcome: Dict[str, Any]) -> None:
        self._outcomes_seen += 1
        o = self._extract(outcome)

        # 1) update learner (keep existing behavior)
        if self.learner is not None:
            try:
                # support both method names
                if hasattr(self.learner, "on_outcome"):
                    self.learner.on_outcome(outcome)  # type: ignore[attr-defined]
                elif hasattr(self.learner, "update"):
                    self.learner.update(outcome)      # type: ignore[attr-defined]
            except Exception:
                pass

        # 2) update weight store
        ws = self.weight_store
        if ws is not None:
            self._update_weight(ws, o)

            # periodic decay toward 1.0
            if self.decay_every > 0 and (self._outcomes_seen % self.decay_every == 0):
                try:
                    ws.decay_toward(target=1.0, rate=self.decay_rate, regime=o["regime"])
                    ws.normalize_mean(regime=o["regime"], target_mean=1.0)
                except Exception:
                    pass

            # periodic autosave
            if self.autosave and self.weights_path and self.save_every > 0:
                if self._outcomes_seen % self.save_every == 0:
                    try:
                        ws.save_json(self.weights_path)
                    except Exception:
                        pass

    def _update_weight(self, ws: "WeightStore", o: Dict[str, Any]) -> None:
        """
        Stable update rule:
          - base signal = +1 for win, -1 for loss
          - scale by abs(pnl) (soft) and optional ATR
          - forced trades get smaller update (penalty)
        """
        pnl = float(o["pnl"])
        win = bool(o["win"])
        expert = str(o["expert"])
        regime = str(o["regime"])
        forced = bool(o["forced"])
        atr = o.get("atr", None)

        # direction
        sgn = 1.0 if win else -1.0

        # soft magnitude: cap to avoid explosions
        mag = min(1.0, abs(pnl) / 5.0)  # tweakable
        if mag < 0.05:
            mag = 0.05

        # ATR scaling (optional)
        if atr is not None and atr > 0:
            # normalize atr around ~1.0 scale
            atr_scale = min(2.0, max(0.5, atr / 1.0))
            mag *= atr_scale

        # forced penalty
        if forced:
            mag *= (1.0 - self.forced_penalty)

        delta = self.lr * sgn * mag

        # apply
        try:
            ws.bump(expert=expert, regime=regime, delta=delta, reason="outcome")
        except TypeError:
            # backward fallback if someone changed signature
            ws.bump(expert, delta, regime=regime, reason="outcome")  # type: ignore[misc]
