# brain/decision_engine.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from brain.regime_detector import RegimeDetector
from brain.meta_controller import MetaController
from brain.experts.expert_registry import ExpertRegistry
from brain.experts.expert_gate import ExpertGate, ExpertDecision

@dataclass
class DecisionContext:
    regime: str = "unknown"
    session: Optional[str] = None
    # you can add more tags later: volatility_bucket, news_flag...

@dataclass
class DecisionResult:
    allow: bool
    score: float
    risk_cfg: Dict[str, Any]
    best: Optional[ExpertDecision]
    all_decisions: list

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
        risk_engine: Any = None,
        weight_store: Any = None,
        regime_detector: Optional[RegimeDetector] = None,
        meta_controller: Optional[MetaController] = None,
        **kwargs,
    ) -> None:
        self.risk_engine = risk_engine
        self.weight_store = weight_store
        self.regime_detector = regime_detector or RegimeDetector()
        self.meta_controller = meta_controller or MetaController()
        self.registry = self._build_registry()
        self.gate = ExpertGate(self.registry, weight_store=self.weight_store)

    def _build_registry(self) -> ExpertRegistry:
            # keep your existing ExpertRegistry behavior
        reg = ExpertRegistry()
        try:
            reg.register_defaults()
        except Exception:
            # fallback: ExpertRegistry might auto-register elsewhere
            pass
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
        ctx: Dict[str, Any] = dict(context or {})

        # 1) detect regime
        regime = "unknown"
        try:
            if hasattr(self.regime_detector, "detect"):
                regime = self.regime_detector.detect(trade_features)
            elif hasattr(self.regime_detector, "classify"):
                regime = self.regime_detector.classify(trade_features)
        except Exception:
            regime = "unknown"

        ctx["regime"] = str(regime or "unknown")

        # 2) meta policy (threshold, epsilon)
        meta_policy = {}
        try:
            meta_policy = self.meta_controller.get_policy(ctx["regime"])
        except Exception:
            meta_policy = {"score_threshold": 0.0, "epsilon": 0.05}

        ctx["meta_policy"] = meta_policy

        # 3) pick via gate
        best, all_decisions = self.gate.pick(trade_features, ctx)

        # normalize
        best_score = 0.0
        best_expert = None
        best_allow = False
        if best is not None:
            best_score = float(getattr(best, "score", 0.0) or 0.0)
            best_expert = str(getattr(best, "expert", "unknown") or "unknown")
            best_allow = bool(getattr(best, "allow", False))

        # 4) call meta_controller with decision (confidence = best_score clipped)
        try:
            conf = max(0.0, min(1.0, float(best_score)))
            self.meta_controller.on_decision(ctx["regime"], best_allow, conf)
        except Exception:
            pass

        # 5) risk cfg (used by outcome_updater + weight_store updates)
        risk_cfg = {
            "expert": best_expert,
            "regime": ctx["regime"],
            "score": best_score,
            "score_threshold": float(meta_policy.get("score_threshold", 0.0)),
            "epsilon": float(meta_policy.get("epsilon", 0.0)),
        }

        return best_allow, best_score, risk_cfg