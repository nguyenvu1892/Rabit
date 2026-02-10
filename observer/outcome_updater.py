# observer/outcome_updater.py
from __future__ import annotations
from typing import Any, Dict, Optional
from brain.weight_store import WeightStore

class OutcomeUpdater:
    """
    - process_outcome(snapshot, outcome) or process_outcome(outcome)
    - calls learner.update/learner.learn with (snapshot, outcome, reward)
    """
    def __init__(self, learner=None, trade_memory=None, weight_store: WeightStore | None = None):
        self.learner = learner
        self.trade_memory = trade_memory
        self.weight_store = weight_store or WeightStore()
    
    def _extract_expert_regime(self, outcome: Dict[str, Any]) -> tuple[Optional[str], Optional[str]]:
        # try multiple layouts (robust)
        # 1) direct
        expert = outcome.get("expert")
        regime = outcome.get("regime")

        # 2) nested risk config (DecisionEngine return)
        if (expert is None) or (regime is None):
            risk = outcome.get("risk_cfg") or outcome.get("risk") or {}
            if isinstance(risk, dict):
                expert = expert or risk.get("expert")
                regime = regime or risk.get("regime")

        # 3) snapshot payload (common in journals)
        if (expert is None) or (regime is None):
            snap = outcome.get("snapshot") or outcome.get("payload") or {}
            if isinstance(snap, dict):
                risk2 = snap.get("risk_cfg") or snap.get("risk") or {}
                if isinstance(risk2, dict):
                    expert = expert or risk2.get("expert")
                    regime = regime or risk2.get("regime")
                expert = expert or snap.get("expert")
                regime = regime or snap.get("regime")

        if isinstance(expert, str):
            expert = expert.strip() or None
        if isinstance(regime, str):
            regime = regime.strip() or None

        return expert, regime

    def on_outcome(self, outcome: Dict[str, Any]) -> None:
        """
        Expected minimal:
          outcome["win"] bool
          outcome["pnl"] float (optional)
          and expert/regime (direct or nested)
        """
        if not isinstance(outcome, dict):
            return

        # 1) RL learner update (existing)
        if self.learner is not None:
            try:
                self.learner.on_outcome(outcome)
            except Exception:
                pass

        # 2) Weight learning (5.0.8.1)
        if self.weight_store is not None:
            try:
                win = bool(outcome.get("win", False))
                pnl = float(outcome.get("pnl", 0.0) or 0.0)

                expert, regime = self._extract_expert_regime(outcome)
                if expert:
                    self.weight_store.update_from_outcome(expert, regime, win=win, pnl=pnl)
                    self.weight_store.save()
            except Exception:
                # never crash runner
                pass

    def _reward(self, outcome: Dict[str, Any]) -> float:
        # normalize reward: +1 win, -1 loss; fallback pnl sign
        if "win" in outcome:
            return 1.0 if bool(outcome["win"]) else -1.0
        pnl = outcome.get("pnl")
        if pnl is None:
            return 0.0
        pnl = float(pnl)
        if pnl > 0:
            return 1.0
        if pnl < 0:
            return -1.0
        return 0.0

    def process_outcome(self, *args):
        if len(args) == 1:
            snapshot = {}
            outcome = args[0] or {}
        elif len(args) == 2:
            snapshot, outcome = args
            snapshot = snapshot or {}
            outcome = outcome or {}
        else:
            raise TypeError("process_outcome expects (outcome) or (snapshot, outcome)")

        reward = self._reward(outcome)

        # record memory (optional)
        if self.trade_memory is not None:
            try:
                self.trade_memory.record(snapshot, outcome)
            except Exception:
                pass
        # --- 5.0.7.9: update weight store from outcome ---
        try:
            expert = str((snapshot.get("risk_cfg") or {}).get("expert") or (snapshot.get("meta") or {}).get("expert") or "UNKNOWN_EXPERT")
            regime = str((snapshot.get("risk_cfg") or {}).get("regime") or (snapshot.get("meta") or {}).get("regime") or "UNKNOWN")

            # reward: đơn giản, an toàn, không pnl-scale mạnh
            win = bool(outcome.get("win", False))
            reward = 1.0 if win else -1.0

            # nếu forced thì giảm ảnh hưởng (tránh học bậy)
            forced = bool((snapshot.get("meta") or {}).get("forced", False))
            if forced:
                reward *= 0.25

            self.weight_store.update(expert, regime, reward)

            # attach for logging/debug (optional)
            outcome.setdefault("meta", {})
            outcome["meta"].update({
                "learn_expert": expert,
                "learn_regime": regime,
                "reward": reward,
                "new_weight": float(self.weight_store.get(expert, regime)),
            })
        except Exception:
            pass
                    
        # learner update (optional)
        if self.learner is not None:
            if hasattr(self.learner, "update"):
                self.learner.update(snapshot, outcome, reward)
            elif hasattr(self.learner, "learn"):
                self.learner.learn(snapshot, outcome, reward)

        return reward
