# brain/decision_engine.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple


# --- Optional imports (keep compat; do NOT hard crash on missing) ---
try:
    from brain.experts.expert_gate import ExpertGate  # type: ignore
except Exception:
    ExpertGate = None  # type: ignore

try:
    from brain.experts.expert_registry import ExpertRegistry  # type: ignore
except Exception:
    ExpertRegistry = None  # type: ignore

try:
    from brain.experts.experts_basic import DEFAULT_EXPERTS  # type: ignore
except Exception:
    DEFAULT_EXPERTS = None  # type: ignore

try:
    from brain.experts.expert_base import ExpertDecision  # type: ignore
except Exception:
    ExpertDecision = None  # type: ignore

try:
    from brain.regime_detector import RegimeDetector  # type: ignore
except Exception:
    RegimeDetector = None  # type: ignore


def _as_dict(x: Any) -> Dict[str, Any]:
    return x if isinstance(x, dict) else {}


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        v = float(x)
        if v != v:  # NaN
            return default
        if v == float("inf") or v == float("-inf"):
            return default
        return v
    except Exception:
        return default


def _ensure_risk_cfg(risk_cfg: Any) -> Dict[str, Any]:
    if isinstance(risk_cfg, dict):
        return risk_cfg
    return {}


def _inject_regime_into_risk_cfg(
    risk_cfg: Dict[str, Any],
    regime: str,
    regime_conf: float,
) -> Dict[str, Any]:
    # keep schema stable; only inject if absent or empty
    if not isinstance(risk_cfg, dict):
        risk_cfg = {}
    if not risk_cfg.get("regime"):
        risk_cfg["regime"] = str(regime) if regime is not None else "unknown"
    if "regime_conf" not in risk_cfg:
        risk_cfg["regime_conf"] = _safe_float(regime_conf, 0.0)
    return risk_cfg


def _regime_from_features(features: Dict[str, Any], debug: bool = False) -> Tuple[str, float]:
    """
    NEW (5.1.9): detect regime from FeaturePack features first.
    Fallback to RegimeDetector.detect if available.
    """
    # Prefer FeaturePackV1 signals if present
    slope_n = _safe_float(features.get("slope_n"), 0.0)
    ret_std = _safe_float(features.get("ret_std"), 0.0)
    atr_n = _safe_float(features.get("atr_n"), 0.0)
    ok = bool(features.get("fpv1_ok"))

    # simple, stable thresholds (tune later; keep deterministic)
    # slope_n typical magnitude is small; use conservative threshold
    thr_trend = 0.0006
    thr_range_vol = 0.0012  # if vol very low -> likely range

    if ok and (("slope_n" in features) or ("ret_std" in features) or ("atr_n" in features)):
        if slope_n > thr_trend:
            regime = "trend_up"
            conf = min(1.0, abs(slope_n) / max(1e-9, thr_trend))
        elif slope_n < -thr_trend:
            regime = "trend_down"
            conf = min(1.0, abs(slope_n) / max(1e-9, thr_trend))
        else:
            regime = "range"
            # confidence: lower vol => higher confidence for range
            vol_proxy = abs(ret_std) + abs(atr_n)
            conf = 1.0 - min(1.0, vol_proxy / max(1e-9, thr_range_vol))

        if debug:
            print(f"[DecisionEngine] regime_from_features slope_n={slope_n:.6g} ret_std={ret_std:.6g} atr_n={atr_n:.6g} -> {regime} (conf={conf:.3f})")
        return regime, conf

    # Fallback: use RegimeDetector if available (keeps old behavior)
    if RegimeDetector is not None:
        try:
            rd = RegimeDetector(debug=debug)  # type: ignore
            # Support both APIs: detect(features) or detect(candles)
            if hasattr(rd, "detect"):
                try:
                    out = rd.detect(features)  # type: ignore
                except Exception:
                    candles = features.get("candles")
                    out = rd.detect(candles)  # type: ignore
                # out can be (regime, conf) or dict-like
                if isinstance(out, tuple) and len(out) >= 2:
                    return str(out[0]), _safe_float(out[1], 0.0)
                if isinstance(out, dict):
                    return str(out.get("regime", "unknown")), _safe_float(out.get("confidence", out.get("regime_conf", 0.0)), 0.0)
        except Exception:
            pass

    return "unknown", 0.0


class DecisionEngine:
    """
    DecisionEngine (5.1.x -> 5.1.9):
    - Keeps evaluate_trade API: returns (allow: bool, score: float, risk_cfg: dict)
    - Adds compat blocks; DOES NOT delete old logic paths
    - NEW: inject regime/regime_conf into risk_cfg using features (FeaturePack)
    """

    def __init__(
        self,
        weight_store: Optional[Any] = None,
        risk_engine: Optional[Any] = None,
        debug: bool = False,
        **kwargs: Any,  # COMPAT: accept unexpected kwargs (e.g., older tools passing risk_engine)
    ):
        self.debug = bool(debug)

        # COMPAT: allow passing risk_engine by kwargs too
        if risk_engine is None and "risk_engine" in kwargs:
            risk_engine = kwargs.get("risk_engine")

        self.risk_engine = risk_engine
        self.weight_store = weight_store

        self._epsilon: float = 0.0

        # build registry + gate (compat; do not hard fail)
        self.registry = self._build_registry()
        if ExpertGate is None:
            raise ImportError("ExpertGate is not available (brain.experts.expert_gate import failed).")

        # keep same constructor style as previous iterations
        try:
            self.gate = ExpertGate(self.registry, weight_store=self.weight_store, debug=self.debug)  # type: ignore
        except TypeError:
            # COMPAT: older ExpertGate signature without weight_store
            self.gate = ExpertGate(self.registry, debug=self.debug)  # type: ignore

    # --- compat: keep method for shadow_runner ---
    def set_exploration(self, epsilon: float) -> None:
        self._epsilon = _safe_float(epsilon, 0.0)

    def _build_registry(self) -> Any:
        """
        Prefer ExpertRegistry if available; else fall back to DEFAULT_EXPERTS.
        This preserves prior behavior and avoids circular import crashes.
        """
        if ExpertRegistry is not None:
            try:
                return ExpertRegistry()  # type: ignore
            except Exception:
                pass

        # fallback to DEFAULT_EXPERTS list
        if DEFAULT_EXPERTS is not None:
            try:
                xs = list(DEFAULT_EXPERTS)
            except Exception:
                xs = []

            class _TmpRegistry:
                def __init__(self, items: Any):
                    self._items = items

                def get_all(self) -> Any:
                    return list(self._items)

            return _TmpRegistry(xs)

        # last resort empty registry
        class _Empty:
            def get_all(self) -> Any:
                return []

        return _Empty()

    def evaluate_trade(self, features: Any) -> Tuple[bool, float, Dict[str, Any]]:
        """
        Return (allow, score, risk_cfg)
        """
        feats = _as_dict(features)

        # NEW (5.1.9): detect regime from features (FeaturePack), then inject into risk_cfg
        regime, regime_conf = _regime_from_features(feats, debug=self.debug)

        # Legacy/compat: call ExpertGate in the safest way
        try:
            gate_eval = getattr(self.gate, "evaluate", None)
            gate_decide = getattr(self.gate, "decide", None)

            if callable(gate_eval):
                out = gate_eval(feats)
            elif callable(gate_decide):
                out = gate_decide(feats)
            else:
                # If gate is miswired, fail closed but keep schema stable
                return False, 0.0, _inject_regime_into_risk_cfg({}, regime, regime_conf)

            # Normalize output shapes:
            # - could be ExpertDecision
            # - could be tuple (allow, score, risk_cfg)
            # - could be tuple (allow, score, risk_cfg, expert, meta)
            allow: bool = False
            score: float = 0.0
            risk_cfg: Dict[str, Any] = {}

            if ExpertDecision is not None and isinstance(out, ExpertDecision):  # type: ignore
                allow = bool(getattr(out, "allow", False))
                score = _safe_float(getattr(out, "score", 0.0), 0.0)

                # COMPAT: some versions store risk config in .risk_cfg, some in .meta
                rc = getattr(out, "risk_cfg", None)
                if isinstance(rc, dict):
                    risk_cfg = rc
                else:
                    meta = getattr(out, "meta", None)
                    if isinstance(meta, dict) and isinstance(meta.get("risk_cfg"), dict):
                        risk_cfg = meta["risk_cfg"]
                    else:
                        risk_cfg = {}

                risk_cfg = _inject_regime_into_risk_cfg(risk_cfg, regime, regime_conf)
                return allow, score, risk_cfg

            if isinstance(out, tuple):
                if len(out) >= 3:
                    allow = bool(out[0])
                    score = _safe_float(out[1], 0.0)
                    risk_cfg = _ensure_risk_cfg(out[2])
                    risk_cfg = _inject_regime_into_risk_cfg(risk_cfg, regime, regime_conf)
                    return allow, score, risk_cfg

            # Unknown output -> safe fallback
            return False, 0.0, _inject_regime_into_risk_cfg({}, regime, regime_conf)

        except Exception as e:
            if self.debug:
                print(f"[DecisionEngine] evaluate_trade error: {e}")
            return False, 0.0, _inject_regime_into_risk_cfg({}, regime, regime_conf)