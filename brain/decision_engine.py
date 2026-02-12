# brain/decision_engine.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from brain.weight_store import WeightStore


@dataclass
class DecisionContext:
    regime: str = "unknown"
    session: Optional[str] = None
    # you can add more tags later: volatility_bucket, news_flag...


@dataclass
class EngineDecision:
    allow: bool
    score: float
    action: str
    expert: str
    meta: Dict[str, Any]


class DecisionEngine:
    """
    - Builds ExpertRegistry (from experts_basic or your registry module)
    - Uses ExpertGate to pick best decision
    - Applies risk engine gating if provided

    Key fix:
      - NEVER return (None, None) silently -> always produce a safe fallback decision
      - Keep compatibility with existing sim/shadow_runner expectations
    """

    def __init__(
        self,
        risk_engine: Any,
        weight_store: Optional[WeightStore] = None,
        regime_detector: Optional[Any] = None,
        expert_gate: Optional[Any] = None,
        expert_registry: Optional[Any] = None,
    ) -> None:
        self.risk_engine = risk_engine
        self.weight_store = weight_store
        self.regime_detector = regime_detector

        # lazy import to avoid circulars
        self.registry = expert_registry or self._build_registry()
        self.gate = expert_gate or self._build_gate()

    def _build_registry(self) -> Any:
        """
        Default registry builder.
        Works with your current structure:
          - brain/experts/experts_basic.py provides DEFAULT_EXPERTS
          - brain/experts/expert_registry.py provides ExpertRegistry
        """
        try:
            from brain.experts.expert_registry import ExpertRegistry
        except Exception:
            ExpertRegistry = None  # type: ignore

        try:
            from brain.experts.experts_basic import DEFAULT_EXPERTS
        except Exception:
            DEFAULT_EXPERTS = []  # type: ignore

        if ExpertRegistry is None:
            # fallback: registry as a simple list container
            class _SimpleRegistry:
                def __init__(self, experts):
                    self._experts = experts

                def get_all(self):
                    return list(self._experts)

            return _SimpleRegistry(DEFAULT_EXPERTS)

        reg = ExpertRegistry()
        # Support both:
        #   - reg.register(expert)
        #   - reg.add(expert)
        for exp in DEFAULT_EXPERTS:
            try:
                if hasattr(reg, "register"):
                    reg.register(exp)
                elif hasattr(reg, "add"):
                    reg.add(exp)
            except Exception:
                continue
        return reg

    def _build_gate(self) -> Any:
        """
        Default gate builder.
        """
        try:
            from brain.experts.expert_gate import ExpertGate
            return ExpertGate(registry=self.registry, weight_store=self.weight_store)
        except Exception:
            # ultra-safe fallback if gate import broken
            class _FallbackGate:
                def __init__(self, registry, weight_store=None):
                    self.registry = registry
                    self.weight_store = weight_store

                def pick(self, features, context):
                    # return safe fallback
                    return None, []

            return _FallbackGate(self.registry, self.weight_store)

    def _detect_regime(self, trade_features: Dict[str, Any]) -> str:
        if self.regime_detector is None:
            # fallback heuristic
            return str(trade_features.get("regime") or "unknown")
        try:
            r = self.regime_detector.detect(trade_features)
            return str(r or "unknown")
        except Exception:
            return "unknown"

    def evaluate_trade(self, trade_features: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Tuple[bool, float, Dict[str, Any]]:
        """
        Returns (allow, score, risk_cfg)
        """
        ctx = context or {}
        regime = self._detect_regime(trade_features)
        dctx = DecisionContext(regime=regime, session=ctx.get("session"))

        # ExpertGate returns (best_decision, all_decisions)
        best, all_decisions = None, []
        try:
            best, all_decisions = self.gate.pick(trade_features, {"regime": dctx.regime, "session": dctx.session})
        except Exception as e:
            best = None
            all_decisions = []
            ctx["gate_error"] = repr(e)

        # If gate returned None (or empty), create a safe fallback
        if best is None:
            best = EngineDecision(
                allow=True,                 # IMPORTANT: do not deadlock the sim
                score=0.0001,
                action="hold",
                expert="FALLBACK",
                meta={"reason": "no_best_from_gate", "regime": dctx.regime},
            )
        else:
            # normalize best to EngineDecision shape if your ExpertDecision differs
            if not isinstance(best, EngineDecision):
                # Many versions use ExpertDecision dataclass with fields:
                # (expert, score, allow, action, meta)
                allow = bool(getattr(best, "allow", False))
                score = float(getattr(best, "score", 0.0) or 0.0)
                action = str(getattr(best, "action", "hold") or "hold")
                expert = str(getattr(best, "expert", "UNKNOWN") or "UNKNOWN")
                meta = dict(getattr(best, "meta", {}) or {})
                meta.setdefault("regime", dctx.regime)
                best = EngineDecision(allow=allow, score=score, action=action, expert=expert, meta=meta)

        # Risk engine gate
        risk_cfg: Dict[str, Any] = {}
        allow = bool(best.allow)
        score = float(best.score)

        try:
            if self.risk_engine is not None and hasattr(self.risk_engine, "evaluate"):
                allow, score, risk_cfg = self.risk_engine.evaluate(trade_features, best_action=best.action, score=score)
        except Exception as e:
            risk_cfg = {"error": repr(e)}

        # attach debug for eval_reporter / shadow_run
        risk_cfg["best_expert"] = best.expert
        risk_cfg["best_action"] = best.action
        risk_cfg["best_score"] = score
        risk_cfg["regime"] = dctx.regime

        return bool(allow), float(score), risk_cfg
