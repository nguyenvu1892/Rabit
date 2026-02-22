# brain/decision_engine.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

# Keep imports lazy/soft to avoid circular import issues
try:
    from brain.regime_detector import RegimeDetector
except Exception:
    RegimeDetector = None  # type: ignore

try:
    from brain.weight_store import WeightStore
except Exception:
    WeightStore = None  # type: ignore

try:
    from brain.experts.expert_gate import ExpertGate
except Exception:
    ExpertGate = None  # type: ignore

try:
    from brain.feature_registry import ExpertRegistry
except Exception:
    ExpertRegistry = None  # type: ignore


@dataclass
class DecisionResult:
    allow: bool
    score: float
    risk_cfg: Dict[str, Any]
    meta: Dict[str, Any]


class DecisionEngine:
    """
    DecisionEngine (stable + compat).

    Design constraints:
    - Must keep existing API used by tools/shadow_run.py and sim/shadow_runner.py
    - Must provide regime + confidence in risk_cfg/meta
    - Must be resilient to missing optional modules (registry/gate/detector)
    """

    def __init__(
        self,
        weight_store: Any = None,
        *,
        debug: bool = False,
        regime_debug: bool = False,
        meta_debug: bool = False,
        **kwargs: Any,
    ) -> None:
        self.debug = bool(debug)
        self.regime_debug = bool(regime_debug)
        self.meta_debug = bool(meta_debug)

        self.weight_store = weight_store

        # detector
        self.regime_detector = None
        if RegimeDetector is not None:
            try:
                self.regime_detector = RegimeDetector(debug=regime_debug)
            except TypeError:
                try:
                    self.regime_detector = RegimeDetector()
                except Exception:
                    self.regime_detector = None
            except Exception:
                self.regime_detector = None

        # registry + gate (compat)
        self.registry = self._build_registry()
        if ExpertGate is None:
            raise ImportError("ExpertGate is not available (brain.experts.expert_gate import failed).")
        self.gate = ExpertGate(self.registry, weight_store=self.weight_store, debug=debug)

        # other optional things
        self.extra = dict(kwargs) if kwargs else {}

    # ----------------------------
    # Registry builder (compat)
    # ----------------------------
    def _build_registry(self) -> Any:
        """
        Prefer ExpertRegistry if available; else fallback to DEFAULT_EXPERTS import.

        IMPORTANT: do not break previous imports / circular dependency behavior.
        """
        if ExpertRegistry is not None:
            try:
                return ExpertRegistry()
            except Exception:
                pass

        # fallback DEFAULT_EXPERTS
        try:
            from brain.experts.experts_basic import DEFAULT_EXPERTS
        except Exception:
            DEFAULT_EXPERTS = []

        class _TmpRegistry:
            def __init__(self, xs):
                self._xs = xs

            def get_all(self):
                return list(self._xs)

        return _TmpRegistry(DEFAULT_EXPERTS)

    # ----------------------------
    # Core public API used by sim/shadow_runner.py
    # ----------------------------
    def evaluate_trade(self, features: Dict[str, Any]) -> Tuple[bool, float, Dict[str, Any]]:
        """
        Returns: (allow, score, risk_cfg)
        """
        # 1) regime detect
        regime, regime_conf = self._detect_regime(features)

        # 2) expert gate decide
        allow, score, risk_cfg, meta = self._gate_decide(features, regime, regime_conf)

        # 3) ensure regime is injected into cfg (critical for OutcomeUpdater/ShadowStats breakdown)
        risk_cfg = {
            "regime": regime,
            "regime_conf": regime_conf,
        }

        # === COMPAT: inject expert name into risk_cfg ===
        try:
            risk_cfg["expert"] = best.expert
        except Exception:
            risk_cfg["expert"] = "UNKNOWN"

        # keep meta stable
        risk_cfg.setdefault("meta", {})
        risk_cfg["meta"].setdefault("expert", risk_cfg["expert"])
        meta = meta or {}

        self._inject_regime(risk_cfg, meta, regime, regime_conf)

        # 4) ensure stable return
        return bool(allow), float(score), dict(risk_cfg)

    # ----------------------------
    # Internal helpers
    # ----------------------------
    def _detect_regime(self, features: Dict[str, Any]) -> Tuple[str, float]:
        if self.regime_detector is None:
            return "unknown", 0.0

        try:
            rr = self.regime_detector.detect(features)
            # rr might be a dict-like or dataclass-like
            if isinstance(rr, dict):
                rg = rr.get("regime", "unknown")
                cf = rr.get("confidence", rr.get("conf", 0.0))
                return str(rg), float(cf or 0.0)

            rg = getattr(rr, "regime", "unknown")
            cf = getattr(rr, "confidence", getattr(rr, "conf", 0.0))
            return str(rg), float(cf or 0.0)
        except Exception:
            return "unknown", 0.0

    def _gate_decide(
        self,
        features: Dict[str, Any],
        regime: str,
        regime_conf: float,
    ) -> Tuple[bool, float, Dict[str, Any], Dict[str, Any]]:
        """
        Gate returns: allow, score, risk_cfg, meta
        """
        try:
            res = self.gate.evaluate(features, regime=regime, regime_conf=regime_conf)
            # allow various shapes
            if isinstance(res, tuple) and len(res) == 4:
                allow, score, risk_cfg, meta = res
                return bool(allow), float(score), (risk_cfg or {}), (meta or {})
            if isinstance(res, tuple) and len(res) == 3:
                allow, score, risk_cfg = res
                return bool(allow), float(score), (risk_cfg or {}), {}
            if isinstance(res, dict):
                return bool(res.get("allow", False)), float(res.get("score", 0.0)), (res.get("risk_cfg", {}) or {}), (res.get("meta", {}) or {})
        except Exception as e:
            if self.debug:
                try:
                    print("[DecisionEngine] gate.evaluate failed:", repr(e))
                except Exception:
                    pass

        # fallback (safe allow)
        meta = {"expert": "FALLBACK", "reason": "gate_exception"}
        risk_cfg = {"expert": "FALLBACK", "forced": False}
        return True, 0.0, risk_cfg, meta

    def _inject_regime(self, risk_cfg: Dict[str, Any], meta: Dict[str, Any], regime: str, conf: float) -> None:
        """
        Inject into BOTH risk_cfg and meta, because different parts of the pipeline read different places.
        Keep schema stable.
        """
        # risk_cfg top-level
        if "regime" not in risk_cfg:
            risk_cfg["regime"] = regime
        if "regime_conf" not in risk_cfg:
            risk_cfg["regime_conf"] = float(conf)

        # meta top-level
        if "regime" not in meta:
            meta["regime"] = regime
        if "regime_conf" not in meta:
            meta["regime_conf"] = float(conf)

        # also keep nested meta inside risk_cfg (some older code expects it)
        m2 = risk_cfg.get("meta")
        if not isinstance(m2, dict):
            m2 = {}
            risk_cfg["meta"] = m2
        if "regime" not in m2:
            m2["regime"] = regime
        if "regime_conf" not in m2:
            m2["regime_conf"] = float(conf)

        # ensure expert is surfaced (OutcomeUpdater needs it)
        if "expert" not in risk_cfg and isinstance(meta.get("expert"), str):
            risk_cfg["expert"] = meta["expert"]
        if "expert" not in meta and isinstance(risk_cfg.get("expert"), str):
            meta["expert"] = risk_cfg["expert"]

    # ==========================================================
    # ✅ COMPAT BLOCKS (DO NOT REMOVE OLD API CALLS)
    # ==========================================================
    def decide_trade(self, features: Dict[str, Any]) -> Tuple[bool, float, Dict[str, Any]]:
        """
        Legacy alias (older code might call decide_trade).
        """
        return self.evaluate_trade(features)

    def evaluate(self, features: Dict[str, Any]) -> Tuple[bool, float, Dict[str, Any]]:
        """
        Legacy alias: some code calls engine.evaluate(features)
        """
        return self.evaluate_trade(features)

    def decide(self, features: Dict[str, Any]) -> DecisionResult:
        """
        Legacy style: return a structured object.
        """
        allow, score, risk_cfg = self.evaluate_trade(features)
        meta = risk_cfg.get("meta", {}) if isinstance(risk_cfg.get("meta"), dict) else {}
        return DecisionResult(bool(allow), float(score), dict(risk_cfg), dict(meta))