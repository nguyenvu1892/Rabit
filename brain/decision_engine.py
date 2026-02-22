# brain/decision_engine.py
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

# compat imports
try:
    from brain.experts.expert_gate import ExpertGate  # type: ignore
except Exception:
    ExpertGate = None  # type: ignore

try:
    from brain.experts.expert_registry import ExpertRegistry  # type: ignore
except Exception:
    ExpertRegistry = None  # type: ignore


class DecisionEngine:
    def __init__(self, weight_store=None, debug: bool = False, **kwargs) -> None:
        self.weight_store = weight_store
        self.debug = bool(debug)

        # exploration (compat)
        self.epsilon = 0.0
        self.epsilon_cooldown = 0

        # build registry + gate (compat-first)
        self.registry = self._build_registry()
        if ExpertGate is None:
            raise ImportError("ExpertGate is not available (brain.experts.expert_gate import failed).")
        self.gate = ExpertGate(self.registry, weight_store=self.weight_store, debug=self.debug)

    # -------- compat: exploration --------
    def set_exploration(self, epsilon: float = 0.0, cooldown: int = 0) -> None:
        self.epsilon = float(epsilon)
        self.epsilon_cooldown = int(cooldown)

    # -------- registry --------
    def _build_registry(self) -> Any:
        # Prefer ExpertRegistry; else fallback to DEFAULT_EXPERTS import
        if ExpertRegistry is not None:
            try:
                return ExpertRegistry()
            except Exception:
                pass

        try:
            from brain.experts.experts_basic import DEFAULT_EXPERTS  # type: ignore

            class _TmpRegistry:
                def __init__(self, xs):
                    self._xs = xs

                def get_all(self):
                    return list(self._xs)

            return _TmpRegistry(DEFAULT_EXPERTS)
        except Exception:
            class _Empty:
                def get_all(self):
                    return []

            return _Empty()

    # -------- meta ensure --------
    def _ensure_meta_dict(self, risk_cfg: Any) -> Dict[str, Any]:
        if not isinstance(risk_cfg, dict):
            risk_cfg = {}
        m = risk_cfg.get("meta")
        if not isinstance(m, dict):
            m = {}
            risk_cfg["meta"] = m
        return risk_cfg

    # -------- core API (KEEP): evaluate_trade(features)->(allow, score, risk_cfg) --------
    def evaluate_trade(self, features: Dict[str, Any]) -> Tuple[bool, float, Dict[str, Any]]:
        """
        KEEP API + schema:
        - returns (allow: bool, score: float, risk_cfg: dict)
        - risk_cfg may contain regime/regime_conf/forced/meta...
        """

        context: Dict[str, Any] = {}

        # compat: pass regime if caller already computed it
        if isinstance(features, dict):
            if "regime" in features:
                context["regime"] = features.get("regime")

        # Gate decision
        best_decision = None
        decisions = []
        try:
            best_decision, decisions = self.gate.pick(features, context)
        except Exception:
            # hard fallback (never crash)
            best_decision = None

        # risk_cfg baseline (DO NOT break old callers)
        risk_cfg: Dict[str, Any] = {}
        risk_cfg = self._ensure_meta_dict(risk_cfg)

        # Extract expert + score safely
        expert = "UNKNOWN"
        score = 0.0
        if best_decision is not None:
            try:
                expert = str(getattr(best_decision, "expert", None) or "UNKNOWN")
            except Exception:
                expert = "UNKNOWN"
            try:
                score = float(getattr(best_decision, "score", 0.0) or 0.0)
            except Exception:
                score = 0.0

            # pack meta for downstream learning/debug
            try:
                risk_cfg["meta"]["best_expert"] = expert
                risk_cfg["meta"]["best_score"] = score
                risk_cfg["meta"]["gate_src"] = "ExpertGate"
            except Exception:
                pass

        # IMPORTANT: inject expert for WeightStore/OutcomeUpdater keying
        risk_cfg["expert"] = expert
        risk_cfg["meta"]["expert"] = expert

        # allow policy (compat):
        # ExpertGate already forces best.allow=True to avoid deny=100%.
        allow = True
        try:
            allow = bool(getattr(best_decision, "allow", True))
        except Exception:
            allow = True

        # regime passthrough (if context exists)
        if "regime" in context and context["regime"] is not None:
            risk_cfg["regime"] = context["regime"]
        # else keep unknown and let RegimeDetector/RiskEngine fill later

        # forced default
        risk_cfg.setdefault("forced", False)

        return bool(allow), float(score), risk_cfg