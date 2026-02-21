# brain/decision_engine.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from brain.regime_detector import RegimeDetector, RegimeResult
from brain.meta_controller import MetaController, MetaConfig

try:
    from brain.experts.expert_gate import ExpertGate
    from brain.experts.expert_registry import ExpertRegistry
except Exception:
    ExpertGate = None  # type: ignore
    ExpertRegistry = None  # type: ignore

try:
    from brain.experts.expert_base import ExpertDecision
except Exception:
    @dataclass
    class ExpertDecision:  # type: ignore
        expert: str
        score: float = 0.0
        allow: bool = True
        action: str = "hold"
        meta: Dict[str, Any] = None  # type: ignore


class DecisionEngine:
    def __init__(
        self,
        risk_engine: Any,
        weight_store: Optional[Any] = None,
        trade_memory: Optional[Any] = None,
        *,
        debug: bool = False,
        meta_cfg: Optional[MetaConfig] = None,
    ) -> None:
        self.risk_engine = risk_engine
        self.weight_store = weight_store
        self.trade_memory = trade_memory
        self.debug = debug

        self.regime_detector = RegimeDetector(debug=debug)
        self.meta = MetaController(meta_cfg or MetaConfig(debug=debug))

        # build registry + gate (compat)
        self.registry = self._build_registry()
        if ExpertGate is None:
            raise ImportError("ExpertGate is not available (brain.experts.expert_gate import failed).")
        self.gate = ExpertGate(self.registry, weight_store=self.weight_store, debug=debug)

    def _build_registry(self) -> Any:
        # Prefer ExpertRegistry if available; else fallback to DEFAULT_EXPERTS import
        if ExpertRegistry is not None:
            try:
                return ExpertRegistry()
            except Exception:
                pass

        try:
            from brain.experts.experts_basic import DEFAULT_EXPERTS

            # DEFAULT_EXPERTS might be list of ExpertBase instances/classes
            class _TmpRegistry:
                def __init__(self, xs):
                    self._xs = xs

                def get_all(self):
                    return list(self._xs)

            return _TmpRegistry(DEFAULT_EXPERTS)
        except Exception:
            # last resort empty registry
            class _Empty:
                def get_all(self):
                    return []

            return _Empty()

    def evaluate_trade(self, features: Dict[str, Any]) -> Tuple[bool, float, Dict[str, Any]]:
        """
        Returns: allow(bool), score(float), risk_cfg(dict)
        NOTE:
          - allow here means "system produced a decision event" (even if HOLD),
            not necessarily open position.
        """
        # 1) detect regime
        rr: RegimeResult = self.regime_detector.detect(features)

        # 2) build context
        context: Dict[str, Any] = {
            "regime": rr.regime,
            "regime_conf": rr.confidence,
            "vol": rr.vol,
            "slope": rr.slope,
        }
        if self.trade_memory is not None:
            context["trade_memory"] = self.trade_memory

        # 3) expert gate pick
        best, all_decs = self.gate.pick(features, context)

        # 4) meta apply (convert weak signals -> HOLD, not DENY)
        best = self.meta.apply(best, rr, context)

        # Debug traces (keep compatible with your logs)
        if self.debug:
            try:
                print("DEBUG BEST:", best)
                print("DEBUG BEST_SCORE:", getattr(best, "score", None))
                print("DEBUG BEST_EXPERT:", getattr(best, "expert", None))
                print("DEBUG BEST_META:", getattr(best, "meta", None))
            except Exception:
                pass

        # 5) risk config
        risk_cfg: Dict[str, Any] = {}
        try:
            # risk engine may accept (features, context, decision)
            if hasattr(self.risk_engine, "build"):
                risk_cfg = self.risk_engine.build(features, context, best)
            elif callable(self.risk_engine):
                risk_cfg = self.risk_engine(features, context, best)
        except Exception as e:
            risk_cfg = {"risk_error": repr(e)}

        # --- FIX TRIỆT ĐỂ unknown/conf_sum=0 ---
        # ShadowRunner đang lấy regime/conf từ risk_cfg.
        # Vì vậy luôn inject thông tin regime vào risk_cfg (KHÔNG override nếu risk_engine đã set).
        try:
            if isinstance(risk_cfg, dict):
                risk_cfg.setdefault("regime", rr.regime)
                risk_cfg.setdefault("regime_conf", rr.confidence)
                risk_cfg.setdefault("vol", rr.vol)
                risk_cfg.setdefault("slope", rr.slope)
                # thêm 1 field tiện debug
                risk_cfg.setdefault("_regime_src", "DecisionEngine")
        except Exception:
            pass
        # ---------------------------------------

        # 6) output
        allow = bool(getattr(best, "allow", True))

        try:
            score = float(getattr(best, "score", 0.0) or 0.0)
        except Exception:
            score = 0.0

        # ensure HOLD is a valid decision event (do not global-deny HOLD)
        if allow is False and getattr(best, "action", "hold") == "hold":
            allow = True

        return allow, score, risk_cfg