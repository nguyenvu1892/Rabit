# brain/decision_engine.py
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

try:
    from brain.experts.expert_gate import ExpertGate
except Exception:
    ExpertGate = None  # type: ignore

try:
    from brain.experts.expert_base import ExpertDecision
except Exception:
    ExpertDecision = None  # type: ignore


class DecisionEngine:
    """
    DecisionEngine: glue layer between features -> regime -> expert_gate -> risk_cfg.

    MUST remain compatible with:
    - DecisionEngine(weight_store=..., risk_engine=..., debug=...)
    - older code paths that pass risk_engine even if unused
    """

    def __init__(
        self,
        weight_store: Optional[Any] = None,
        risk_engine: Optional[Any] = None,
        debug: bool = False,
        **kwargs: Any,
    ) -> None:
        # Keep kwargs for compat (do not crash)
        self._compat_kwargs = dict(kwargs)

        self.weight_store = weight_store
        self.risk_engine = risk_engine  # may be None, keep for compat
        self.debug = bool(debug)

        # Registry bootstrap:
        self.registry = self._build_registry()

        if ExpertGate is None:
            raise ImportError("ExpertGate is not available (brain.experts.expert_gate import failed).")

        self.gate = ExpertGate(self.registry, weight_store=self.weight_store, debug=self.debug)

        # Lazy regime detector (optional)
        self.regime_detector = None
        try:
            from brain.regime_detector import RegimeDetector

            self.regime_detector = RegimeDetector(debug=self.debug)
        except Exception:
            self.regime_detector = None

    def _build_registry(self) -> Any:
        """
        Prefer ExpertRegistry if available; else fallback to DEFAULT_EXPERTS.
        """
        # Try ExpertRegistry if exists
        try:
            from brain.experts.expert_registry import ExpertRegistry  # type: ignore

            return ExpertRegistry()
        except Exception:
            pass

        # Fallback: build a tiny registry wrapper around DEFAULT_EXPERTS
        try:
            from brain.experts.experts_basic import DEFAULT_EXPERTS

            class _TmpRegistry:
                def __init__(self, xs: Any):
                    self._xs = list(xs)

                def get_all(self):
                    return list(self._xs)

            return _TmpRegistry(DEFAULT_EXPERTS)
        except Exception:
            # Last resort empty registry (gate will baseline)
            class _Empty:
                def get_all(self):
                    return []

            return _Empty()

    def _ensure_context(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build context dict including regime/regime_conf if possible.
        """
        ctx: Dict[str, Any] = {}

        # If features already carry regime info, keep it
        for k in ("regime", "market_regime", "state"):
            if k in features and features.get(k) is not None:
                ctx["regime"] = features.get(k)
                break

        # Detect regime from features (preferred) or candles (compat)
        if ctx.get("regime") is None and self.regime_detector is not None:
                # New style: detector can read features dict
            try:
                out = self.regime_detector.detect(features)

                # ===== NEW COMPAT BLOCK =====
                # Support tuple return (regime, confidence)
                if isinstance(out, tuple) and len(out) >= 2:
                    ctx["regime"] = out[0]
                    ctx["regime_conf"] = float(out[1])
                elif isinstance(out, dict):
                    ctx["regime"] = (
                        out.get("regime")
                        or out.get("market_regime")
                        or out.get("state")
                    )
                    if "confidence" in out:
                        ctx["regime_conf"] = out["confidence"]
                    elif "regime_conf" in out:
                        ctx["regime_conf"] = out["regime_conf"]
            except Exception:
                pass

        # Hard fallback
        if ctx.get("regime") is None:
            ctx["regime"] = "unknown"

        return ctx

    def evaluate_trade(self, features: Dict[str, Any]) -> Tuple[bool, float, Dict[str, Any]]:
        """
        Returns (allow, score, risk_cfg).

        risk_cfg schema remains dict and will carry:
        - regime, regime_conf
        - expert
        - meta (optional)
        """
        if ExpertDecision is None:
            # ultra-fallback
            allow, score, risk_cfg = True, 0.0, {"regime": "unknown", "expert": "UNKNOWN"}
            return allow, score, risk_cfg

        context = self._ensure_context(features)

        try:
            best = self.gate.evaluate_trade(features, context=context)
        except Exception:
            # Compat: gate might expose evaluate() or decide()
            try:
                best = self.gate.evaluate(features, context=context)  # type: ignore
            except Exception:
                best = ExpertDecision(expert="UNKNOWN", allow=False, score=0.0, action="hold", meta={"reason": "gate_failed"})

        # Normalize
        allow = bool(getattr(best, "allow", False))
        score = float(getattr(best, "score", 0.0) or 0.0)

        # risk_cfg schema (stable)
        risk_cfg: Dict[str, Any] = {}
        risk_cfg["regime"] = context.get("regime", "unknown")
        if "regime_conf" in context:
            risk_cfg["regime_conf"] = context["regime_conf"]
        risk_cfg["expert"] = getattr(best, "expert", "UNKNOWN")

        # Keep meta, but don't explode schema
        meta = getattr(best, "meta", None)
        if isinstance(meta, dict) and meta:
            risk_cfg["meta"] = meta

        return allow, score, risk_cfg