# brain/decision_engine.py
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple
import random
import inspect

from brain.risk_engine import RiskEngine
from brain.trade_memory import TradeMemory
from brain.context_memory import ContextMemory
from brain.context_intelligence import ContextIntelligence
from brain.regime_detector import detect_regime

# experts live under brain/experts/*
from brain.experts.expert_registry import ExpertRegistry
from brain.experts.expert_gate import ExpertGate
from brain.experts.experts_basic import TrendMAExpert, MeanRevertExpert, BreakoutExpert

# (5.0.7.6+) weight store optional
try:
    from brain.weight_store import WeightStore  # type: ignore
except Exception:
    WeightStore = None  # type: ignore


class DecisionEngine:
    """
    5.0.7.x: DecisionEngine không quyết định trực tiếp nữa -> dùng ExpertGate pick expert.
    Backward-compat: vẫn trả (allow, score, risk_cfg) như cũ.
    """

    def __init__(
        self,
        risk_engine=None,
        trade_memory=None,
        context_intel=None,
        context_memory=None,
        weight_store=None,
        allow_threshold: float = 0.0,
        epsilon: float = 0.0,
        epsilon_cooldown: int = 0,
        seed: int | None = None,
    ) -> None:
        self.risk_engine = risk_engine or RiskEngine()
        self.trade_memory = trade_memory or TradeMemory()

        # --- context intel (constructor hiện tại cần context_memory) ---
        self.context_memory = context_memory or ContextMemory()
        self.context_intel = context_intel or ContextIntelligence(self.context_memory)

        # keep for older paths
        self.allow_threshold = float(allow_threshold)

        # exploration controls (đúng: nhận từ args)
        self.epsilon = float(epsilon)
        self.epsilon_cooldown = int(epsilon_cooldown)
        self._rng = random.Random(seed) if seed is not None else random.Random()

        # --- 5.0.7.6+: weight store optional (không crash nếu module thiếu) ---
        if weight_store is not None:
            self.weight_store = weight_store
        else:
            if WeightStore is not None:
                try:
                    self.weight_store = WeightStore()
                except Exception:
                    self.weight_store = None
            else:
                self.weight_store = None

        # registry + experts
        self.registry = ExpertRegistry()
        self.registry.register(TrendMAExpert())
        self.registry.register(MeanRevertExpert())
        self.registry.register(BreakoutExpert())

        # Gate init: pass only params it actually supports
        gate_kwargs: Dict[str, Any] = {"epsilon": 0.0}  # epsilon injected per-step
        try:
            sig = inspect.signature(ExpertGate.__init__)
            if "epsilon_cooldown" in sig.parameters:
                gate_kwargs["epsilon_cooldown"] = int(self.epsilon_cooldown)
            if "rng" in sig.parameters:
                gate_kwargs["rng"] = self._rng
        except Exception:
            pass

        self.gate = ExpertGate(self.registry, **gate_kwargs)
        self._last_intent_id: Optional[str] = None

    def set_exploration(self, epsilon: float, cooldown: int = 0) -> None:
        """
        Called by ShadowRunner. Keep backward-compat & sync to gate if present.
        """
        self.epsilon = float(epsilon)
        self.epsilon_cooldown = int(cooldown)

        # sync into gate if it supports these fields / methods
        if getattr(self, "gate", None) is None:
            return

        # epsilon
        if hasattr(self.gate, "set_epsilon"):
            try:
                self.gate.set_epsilon(self.epsilon)
            except Exception:
                pass
        else:
            try:
                setattr(self.gate, "epsilon", self.epsilon)
            except Exception:
                pass

        # cooldown
        try:
            if hasattr(self.gate, "epsilon_cooldown"):
                setattr(self.gate, "epsilon_cooldown", self.epsilon_cooldown)
        except Exception:
            pass

    def evaluate_trade(self, trade_features: Dict[str, Any]) -> Tuple[bool, float, Dict[str, Any]]:
        """
        Return: (allow, score, risk_cfg)
        trade_features expects at least:
          - "candles": list/window
        """
        candles = trade_features.get("candles") or trade_features.get("window") or []
        context = detect_regime(candles)

        # keep gate in sync per-step (epsilon may change over time)
        if getattr(self, "gate", None) is not None:
            if hasattr(self.gate, "set_epsilon"):
                try:
                    self.gate.set_epsilon(float(getattr(self, "epsilon", 0.0)))
                except Exception:
                    pass
            if hasattr(self.gate, "tick"):
                try:
                    self.gate.tick()
                except Exception:
                    pass

        best, _all_decisions = self.gate.pick(trade_features, context)

        allow = bool(getattr(best, "allow", False))
        score = float(getattr(best, "score", 0.0))
        expert_name = str(getattr(best, "expert", "UNKNOWN_EXPERT"))

        meta = dict(getattr(best, "meta", {}) or {})
        forced = bool(meta.get("forced", False))
        regime = str(context.get("regime", "UNKNOWN"))

        # attach weight if available (does not break anything)
        if self.weight_store is not None:
            try:
                meta["weight"] = float(self.weight_store.get(expert_name))
            except Exception:
                pass

        risk_cfg: Dict[str, Any] = {
            "expert": expert_name,
            "regime": regime,
            "forced": forced,
            "meta": meta,
            "score": score,
        }

        return allow, score, risk_cfg
