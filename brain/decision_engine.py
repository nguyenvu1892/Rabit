# brain/decision_engine.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple


# -----------------------------
# Optional imports (compat)
# -----------------------------
try:
    from brain.experts.expert_gate import ExpertGate  # type: ignore
except Exception:
    ExpertGate = None  # type: ignore

try:
    from brain.experts.expert_base import ExpertDecision  # type: ignore
except Exception:
    ExpertDecision = None  # type: ignore

try:
    from brain.regime_detector import RegimeDetector  # type: ignore
except Exception:
    RegimeDetector = None  # type: ignore

try:
    # If you have a real registry class
    from brain.experts.expert_registry import ExpertRegistry  # type: ignore
except Exception:
    ExpertRegistry = None  # type: ignore


# -----------------------------
# DecisionEngine
# -----------------------------
class DecisionEngine:
    """
    Core evaluator that:
    - Builds expert registry + ExpertGate
    - Detects regime
    - Produces (allow, score, risk_cfg)

    IMPORTANT:
    - Keep API stable
    - Add compat blocks only
    """

    def __init__(
        self,
        weight_store: Optional[Any] = None,
        risk_engine: Optional[Any] = None,
        regime_detector: Optional[Any] = None,
        debug: bool = False,
        **kwargs: Any,
    ):
        # ---- COMPAT BLOCK: accept legacy ctor args (do not remove) ----------
        # tools/shadow_run.py or older callers may pass:
        # - risk_engine=
        # - learner=
        # - reporter=
        # - registry=
        # - gate=
        # We accept everything via **kwargs and map known fields.
        self.debug = bool(debug or kwargs.get("dbg", False) or kwargs.get("verbose", False))
        self.weight_store = weight_store or kwargs.get("weights") or kwargs.get("weightStore")
        self.risk_engine = risk_engine or kwargs.get("risk") or kwargs.get("riskEngine")
        # --------------------------------------------------------------------

        # Regime detector
        if regime_detector is not None:
            self.regime_detector = regime_detector
        else:
            # try build default
            self.regime_detector = None
            try:
                if RegimeDetector is not None:
                    self.regime_detector = RegimeDetector()
            except Exception:
                self.regime_detector = None

        # Build registry + gate
        self.registry = kwargs.get("registry") or None
        if self.registry is None:
            self.registry = self._build_registry()

        if ExpertGate is None:
            raise ImportError("ExpertGate is not available (brain.experts.expert_gate import failed).")

        # Note: ExpertGate signature: (registry, weight_store=None, debug=False)
        self.gate = ExpertGate(self.registry, weight_store=self.weight_store, debug=self.debug)

        # exploration epsilon (optional)
        self._epsilon: float = float(kwargs.get("epsilon", 0.0) or 0.0)

    # -----------------------------
    # Registry builder (compat)
    # -----------------------------
    def _build_registry(self) -> Any:
        """
        Prefer ExpertRegistry if exists, else fallback to DEFAULT_EXPERTS list wrapper.
        """
        # Prefer ExpertRegistry
        if ExpertRegistry is not None:
            try:
                return ExpertRegistry()
            except Exception:
                pass

        # Fallback: DEFAULT_EXPERTS
        try:
            from brain.experts.experts_basic import DEFAULT_EXPERTS  # type: ignore

            class _TmpRegistry:
                def __init__(self, xs: Any):
                    self._xs = xs

                def get_all(self):
                    return list(self._xs) if isinstance(self._xs, list) else []

            return _TmpRegistry(DEFAULT_EXPERTS)
        except Exception:
            # last resort empty registry
            class _Empty:
                def get_all(self):
                    return []

            return _Empty()

    # -----------------------------
    # Optional exploration hook
    # -----------------------------
    def set_exploration(self, epsilon: float) -> None:
        # COMPAT: ShadowRunner may call this
        try:
            self._epsilon = float(epsilon)
        except Exception:
            self._epsilon = 0.0

    # -----------------------------
    # Regime detection (compat)
    # -----------------------------
    def _detect_regime(self, features: Dict[str, Any]) -> Tuple[str, float]:
        """
        Returns (regime_key, regime_conf).
        Tries:
        - regime already in features
        - regime_detector.detect(features)
        - regime_detector.detect(candles) legacy
        """
        # If already provided by upstream, keep it
        r = features.get("market_regime") or features.get("regime") or features.get("state")
        if r:
            try:
                conf = float(features.get("regime_conf") or features.get("confidence") or 0.0)
            except Exception:
                conf = 0.0
            return str(r), conf

        det = getattr(self, "regime_detector", None)
        if det is None:
            return "unknown", 0.0

        # New style: detect from features
        try:
            if hasattr(det, "detect"):
                out = det.detect(features)  # could be str or tuple or dict
                if isinstance(out, tuple) and len(out) >= 2:
                    return str(out[0]), float(out[1] or 0.0)
                if isinstance(out, dict):
                    rr = out.get("regime") or out.get("market_regime") or out.get("state") or "unknown"
                    cc = out.get("regime_conf") or out.get("confidence") or 0.0
                    return str(rr), float(cc or 0.0)
                if isinstance(out, str):
                    return out, 0.0
        except Exception:
            pass

        # Legacy: detect from candles/window
        try:
            candles = features.get("candles")
            if candles is not None and hasattr(det, "detect"):
                out = det.detect(candles)
                if isinstance(out, tuple) and len(out) >= 2:
                    return str(out[0]), float(out[1] or 0.0)
                if isinstance(out, str):
                    return out, 0.0
        except Exception:
            pass

        return "unknown", 0.0

    # -----------------------------
    # Main API: evaluate_trade
    # -----------------------------
    def evaluate_trade(self, features: Dict[str, Any]) -> Tuple[bool, float, Dict[str, Any]]:
        """
        Returns:
          allow: bool
          score: float
          risk_cfg: dict

        MUST stay stable.
        """
        if not isinstance(features, dict):
            features = {}

        # Detect regime and inject to features for ExpertGate weight key
        regime, regime_conf = self._detect_regime(features)
        features.setdefault("market_regime", regime)
        features.setdefault("regime", regime)
        features.setdefault("regime_conf", regime_conf)

        # Gate call (compat)
        allow: bool = False
        score: float = 0.0
        risk_cfg: Dict[str, Any] = {}

        try:
            # Newer gate API: decide() returns GateOutput dataclass
            if hasattr(self.gate, "decide"):
                out = self.gate.decide(features)
                allow = bool(getattr(out, "allow", False))
                score = float(getattr(out, "score", 0.0) or 0.0)
                rc = getattr(out, "risk_cfg", None)
                if isinstance(rc, dict):
                    risk_cfg = dict(rc)
                else:
                    risk_cfg = {}
                # also attach expert/meta if available
                risk_cfg.setdefault("regime", regime)
                risk_cfg.setdefault("regime_conf", regime_conf)
                return allow, score, risk_cfg

            # Older gate API: evaluate() returns (allow, score, risk_cfg)
            if hasattr(self.gate, "evaluate"):
                a, s, rc = self.gate.evaluate(features)
                allow = bool(a)
                score = float(s or 0.0)
                if isinstance(rc, dict):
                    risk_cfg = dict(rc)
                risk_cfg.setdefault("regime", regime)
                risk_cfg.setdefault("regime_conf", regime_conf)
                return allow, score, risk_cfg

        except Exception:
            # fallthrough to safe fallback
            pass

        # Hard fallback: never crash the pipeline
        risk_cfg.setdefault("regime", regime)
        risk_cfg.setdefault("regime_conf", regime_conf)
        return allow, score, risk_cfg