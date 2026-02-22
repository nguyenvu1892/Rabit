# brain/decision_engine.py
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

try:
    from brain.experts.expert_gate import ExpertGate
except Exception:
    ExpertGate = None  # type: ignore

try:
    from brain.experts.registry import ExpertRegistry  # if exists
except Exception:
    ExpertRegistry = None  # type: ignore

from brain.experts.expert_base import ExpertDecision
from brain.regime_detector import RegimeDetector


class DecisionEngine:
    """
    DecisionEngine orchestrates:
    - (optional) regime detection
    - expert gate evaluation
    - return (allow, score, risk_cfg)
    """

    def __init__(
        self,
        weight_store: Optional[Any] = None,
        debug: bool = False,

        # ---- COMPAT BLOCK: accept legacy args from tools/shadow_run.py --------
        risk_engine: Optional[Any] = None,
        risk_cfg: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
        # ---------------------------------------------------------------------
    ):
        self.debug = debug
        self.weight_store = weight_store

        # keep but do not force use
        self.risk_engine = risk_engine
        self.default_risk_cfg = risk_cfg or {}

        # regime detector (lightweight)
        self.regime_detector = RegimeDetector(debug=debug)

        # build registry + gate (compat-safe)
        self.registry = self._build_registry()
        if ExpertGate is None:
            raise ImportError("ExpertGate is not available (brain.experts.expert_gate import failed).")
        self.gate = ExpertGate(self.registry, weight_store=self.weight_store, debug=debug)

    def _build_registry(self) -> Any:
        """
        Prefer ExpertRegistry if available; else fallback to DEFAULT_EXPERTS import.
        Never return a broken registry.
        """
        # Try real registry class
        if ExpertRegistry is not None:
            try:
                reg = ExpertRegistry()
                # ---- COMPAT: if registry empty, seed it with DEFAULT_EXPERTS ----
                try:
                    xs = reg.get_all() if hasattr(reg, "get_all") else None
                    if not xs:
                        from brain.experts.experts_basic import DEFAULT_EXPERTS  # type: ignore
                        # try register method if exists
                        if hasattr(reg, "register") and isinstance(DEFAULT_EXPERTS, list):
                            for e in DEFAULT_EXPERTS:
                                try:
                                    reg.register(e)
                                except Exception:
                                    pass
                except Exception:
                    pass
                # ----------------------------------------------------------------
                return reg
            except Exception:
                pass

        # Fallback registry wrapper around DEFAULT_EXPERTS
        try:
            from brain.experts.experts_basic import DEFAULT_EXPERTS  # type: ignore

            class _TmpRegistry:
                def __init__(self, xs):
                    self._xs = xs

                def get_all(self):
                    return list(self._xs) if isinstance(self._xs, list) else []

            return _TmpRegistry(DEFAULT_EXPERTS)
        except Exception:
            # Last resort empty registry object
            class _Empty:
                def get_all(self):
                    return []

            return _Empty()

    def _ensure_regime(self, features: Dict[str, Any]) -> None:
        """
        Ensure features contains market_regime + regime_conf.
        Does not overwrite if already present.
        """
        if not isinstance(features, dict):
            return

        if features.get("market_regime") is not None:
            return

        candles = features.get("candles")
        if candles is None:
            return

        try:
            regime, conf = self.regime_detector.detect(candles)
            features["market_regime"] = regime
            features["regime_conf"] = conf
        except Exception:
            # keep unknown and let upstream fill later
            pass

    def evaluate_trade(self, features: Dict[str, Any]) -> Tuple[bool, float, Dict[str, Any]]:
        """
        Returns: allow, score, risk_cfg
        """
        if not isinstance(features, dict):
            features = {}

        # ---- COMPAT: compute regime if missing -------------------------------
        self._ensure_regime(features)
        # ---------------------------------------------------------------------

        # ---- COMPAT: gate may expose decide() or evaluate() ------------------
        try:
            if hasattr(self.gate, "evaluate"):
                allow, score, risk_cfg = self.gate.evaluate(features)  # type: ignore
            else:
                out = self.gate.decide(features)  # type: ignore
                allow, score, risk_cfg = bool(out.allow), float(out.score), dict(out.risk_cfg or {})
        except Exception:
            # safe fallback
            dec = ExpertDecision(allow=False, score=0.0, expert="UNKNOWN", meta={})
            allow, score, risk_cfg = dec.allow, dec.score, {}
        # ---------------------------------------------------------------------

        # merge default risk config (do not overwrite gate output)
        try:
            if isinstance(self.default_risk_cfg, dict) and self.default_risk_cfg:
                merged = dict(self.default_risk_cfg)
                if isinstance(risk_cfg, dict):
                    merged.update(risk_cfg)
                risk_cfg = merged
        except Exception:
            pass

        # inject regime into risk_cfg for downstream reporting
        if isinstance(risk_cfg, dict):
            risk_cfg.setdefault("regime", features.get("market_regime", "unknown"))
            risk_cfg.setdefault("regime_conf", features.get("regime_conf", 0.0))

        return bool(allow), float(score), (risk_cfg if isinstance(risk_cfg, dict) else {})