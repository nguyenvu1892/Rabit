# brain/decision_engine.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from brain.regime_detector import RegimeDetector
from brain.meta_controller import MetaController
from brain.experts.expert_registry import ExpertRegistry
from brain.experts.expert_gate import ExpertGate
from brain.experts.expert_base import ExpertDecision

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
        risk_engine: Any,
        weight_store: Optional[WeightStore] = None,
        regime_detector: Optional[RegimeDetector] = None,
        meta_controller: Optional[MetaController] = None,
        gate: Optional[ExpertGate] = None,
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
    
    def _extract_candles(self, trade_features: Any) -> Optional[list]:
        """
        Robustly extract candles list from trade_features.
        Supports:
        - dict with keys: candles/window/recent/bars
        - already-a-list
        """
        if trade_features is None:
            return None
        if isinstance(trade_features, list):
            return trade_features
        if isinstance(trade_features, dict):
            for k in ("candles", "window", "recent", "bars"):
                v = trade_features.get(k)
                if isinstance(v, list):
                    return v
        return None

    def evaluate_trade(self, trade_features: Dict[str, Any]) -> Tuple[bool, float, Dict[str, Any]]:
            """
            Returns: (allow, score, risk_cfg)
            """

            # ---- 1) Detect regime correctly (must be candles list) ----
            candles = self._extract_candles(trade_features)
            regime_result: RegimeResult = self.regime_detector.detect(candles or [])

            # IMPORTANT: keep BOTH
            # - context["regime"] as a simple string: "trend"/"range"/"breakout"/"unknown"
            # - context["regime_result"] as the object (vol/slope/confidence)
            context: Dict[str, Any] = {
                "regime": str(regime_result.regime),
                "regime_result": regime_result,
                "regime_conf": float(getattr(regime_result, "confidence", 0.0) or 0.0),
            }

            # ---- 2) Meta controller can adjust thresholds / params by regime ----
            # Keep this non-breaking: if meta_controller doesn't implement some method, skip gracefully.
            try:
                self.meta_controller.apply(context=context, features=trade_features)
            except Exception:
                pass

            # ---- 3) Experts pick ----
            best_dec, all_decisions = self.gate.pick(trade_features, context)

            # Normalize allow/score even if best_dec is None or has different fields
            allow = False
            score = 0.0
            action = "hold"
            meta = {}

            if best_dec is not None:
                # allow
                if hasattr(best_dec, "allow"):
                    allow = bool(getattr(best_dec, "allow"))
                elif hasattr(best_dec, "allowed"):
                    allow = bool(getattr(best_dec, "allowed"))
                else:
                    # fallback (truthy default)
                    allow = bool(getattr(best_dec, "ok", False))

                # score
                try:
                    score = float(getattr(best_dec, "score", 0.0) or 0.0)
                except Exception:
                    score = 0.0

                # action
                action = str(getattr(best_dec, "action", "hold") or "hold")

                # meta
                m = getattr(best_dec, "meta", {}) or {}
                meta = m if isinstance(m, dict) else {"meta": m}

            # ---- 4) Risk engine (optional) ----
            risk_cfg: Dict[str, Any] = {
                "regime": context["regime"],
                "regime_conf": context.get("regime_conf", 0.0),
                "action": action,
                "best_meta": meta,
                "all_decisions": [d.to_dict() if hasattr(d, "to_dict") else (d.__dict__ if hasattr(d, "__dict__") else str(d)) for d in (all_decisions or [])],
            }

            try:
                # Let risk_engine override/append
                extra = self.risk_engine.compute(trade_features, context=context, decision=best_dec)
                if isinstance(extra, dict):
                    risk_cfg.update(extra)
            except Exception:
                pass

            return allow, score, risk_cfg