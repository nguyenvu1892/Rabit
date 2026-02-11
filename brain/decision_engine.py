# brain/decision_engine.py
from __future__ import annotations

import random
from typing import Any, Dict, Optional, Tuple

from brain.context_memory import ContextMemory
from brain.context_intelligence import ContextIntelligence
from brain.risk_engine import RiskEngine
from brain.trade_memory import TradeMemory
from brain.weight_store import WeightStore
from brain.regime_detector import detect_regime

from brain.experts.expert_registry import ExpertRegistry
from brain.experts.expert_gate import ExpertGate
from brain.experts.experts_basic import TrendMAExpert, MeanRevertExpert, BreakoutExpert

# Meta layer is optional; do NOT break old runs if meta is unstable/missing
try:
    from brain.meta_controller import MetaController  # type: ignore
except Exception:  # pragma: no cover
    MetaController = None  # type: ignore


class DecisionEngine:
    def __init__(
        self,
        context_memory: Optional[ContextMemory] = None,
        context_intel: Optional[ContextIntelligence] = None,
        risk_engine: Optional[RiskEngine] = None,
        trade_memory: Optional[TradeMemory] = None,
        weight_store: Optional[WeightStore] = None,
        seed: Optional[int] = None,
        allow_threshold: float = 0.0,
        meta_controller: Optional[Any] = None,
        meta_enabled: bool = False,
    ) -> None:
        # --- core components (keep old behavior) ---
        self.context_memory = context_memory if context_memory is not None else ContextMemory()
        self.context_intel = context_intel if context_intel is not None else ContextIntelligence(self.context_memory)
        self.risk_engine = risk_engine if risk_engine is not None else RiskEngine()
        self.trade_memory = trade_memory if trade_memory is not None else TradeMemory()

        # --- weights ---
        self.weight_store = weight_store if weight_store is not None else WeightStore()

        # --- backwards-compat ---
        self.allow_threshold = float(allow_threshold)

        # --- exploration controls (ShadowRunner drives these per-step) ---
        self._rng = random.Random(seed) if seed is not None else random.Random()
        self.epsilon = 0.0
        self.epsilon_cooldown = 0

        # --- expert registry / gate ---
        self.registry = ExpertRegistry()
        self.registry.register(TrendMAExpert())
        self.registry.register(MeanRevertExpert())
        self.registry.register(BreakoutExpert())

        # IMPORTANT: gate must exist right after init
        # Match current ExpertGate signature in repo: (registry, epsilon, epsilon_cooldown, rng, weight_store, soft_threshold)
        self.gate = ExpertGate(
            registry=self.registry,
            epsilon=0.0,
            epsilon_cooldown=0,
            rng=self._rng,
            weight_store=self.weight_store,
        )

        # --- meta layer (optional, do not pass enabled=...) ---
        self.meta_enabled = bool(meta_enabled)
        self.meta = meta_controller
        if self.meta is None and self.meta_enabled and MetaController is not None:
            # keep minimal wiring, don't break if meta expects a mapping later
            try:
                self.meta = MetaController(store=self.weight_store)
            except Exception:
                self.meta = None
                self.meta_enabled = False

        # optional tracking (safe)
        self._last_intent_id: Optional[str] = None

    def set_exploration(self, epsilon: float, cooldown: int = 0) -> None:
        """
        Called by ShadowRunner every step (epsilon may change over time).
        ShadowRunner uses keyword 'cooldown', so keep this signature.
        """
        self.epsilon = float(epsilon)
        self.epsilon_cooldown = int(cooldown)

        # keep gate synced
        if hasattr(self, "gate") and self.gate is not None:
            try:
                self.gate.set_epsilon(self.epsilon)
            except Exception:
                pass
            # some versions store epsilon_cooldown as attribute
            try:
                setattr(self.gate, "epsilon_cooldown", self.epsilon_cooldown)
            except Exception:
                pass

    def evaluate_trade(self, trade_features: Dict[str, Any]) -> Tuple[bool, float, Dict[str, Any]]:
        """
        Return: (allow, score, risk_cfg)
        trade_features expected at least: "candles" or "window"
        """
        candles = trade_features.get("candles") or trade_features.get("window") or []
        context = detect_regime(candles)

        # ensure gate exists (defensive)
        if not hasattr(self, "gate") or self.gate is None:
            # fail-safe: deny, don't crash runner
            return False, 0.0, {"error": "gate_missing"}

        # propagate epsilon to gate per-step
        try:
            self.gate.set_epsilon(self.epsilon)
        except Exception:
            pass

        best, _all_decisions = self.gate.pick(trade_features, context)

        allow = bool(getattr(best, "allow", False))
        score = float(getattr(best, "score", 0.0))
        expert = str(getattr(best, "expert", "UNKNOWN_EXPERT"))
        meta = getattr(best, "meta", None) or {}

        # attach weight snapshot if available (does not break anything)
        try:
            meta.setdefault("weight", float(self.weight_store.get(expert)))
        except Exception:
            pass

        risk_cfg: Dict[str, Any] = {
            "expert": expert,
            "regime": context.get("regime", "UNKNOWN") if isinstance(context, dict) else "UNKNOWN",
            "meta": meta,
        }

        return allow, score, risk_cfg
