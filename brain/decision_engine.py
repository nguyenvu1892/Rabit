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

        # Regime detector (debug can be turned off to reduce logs)
        self.regime_detector = RegimeDetector(debug=debug)
        self.meta = MetaController(meta_cfg or MetaConfig(debug=debug))

        # build registry + gate (compat)
        self.registry = self._build_registry()

        if ExpertGate is None:
            raise ImportError("ExpertGate is not available (brain.experts.expert_gate import failed).")

        self.gate = ExpertGate(self.registry, weight_store=self.weight_store, debug=debug)

    def _build_registry(self) -> Any:
        # Prefer ExpertRegistry if available; else fallback to DEFAULT_EXPERTS import
        try:
            from brain.experts.experts_basic import DEFAULT_EXPERTS
        except Exception:
            DEFAULT_EXPERTS = []  # type: ignore

        # 1) If ExpertRegistry exists, create it AND try to register DEFAULT_EXPERTS
        if ExpertRegistry is not None:
            try:
                reg = ExpertRegistry()

                # try common registration APIs
                try:
                    if hasattr(reg, "register") and callable(getattr(reg, "register")):
                        for e in DEFAULT_EXPERTS:
                            reg.register(e)
                    elif hasattr(reg, "add") and callable(getattr(reg, "add")):
                        for e in DEFAULT_EXPERTS:
                            reg.add(e)
                    elif hasattr(reg, "_experts"):
                        xs = getattr(reg, "_experts", None)
                        if isinstance(xs, list):
                            xs.extend(list(DEFAULT_EXPERTS))
                    else:
                        # no known API -> fallback wrapper below
                        raise RuntimeError("ExpertRegistry has no known registration API")
                except Exception:
                    # if registry cannot be populated, fall back to wrapper
                    raise

                return reg
            except Exception:
                pass

        # 2) Wrapper registry with get_all() (ExpertGate can consume reliably)
        try:
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

    def _ensure_meta_dict(self, best: ExpertDecision) -> Dict[str, Any]:
        try:
            m = getattr(best, "meta", None)
            if not isinstance(m, dict):
                m = {}
                best.meta = m  # type: ignore
            return m
        except Exception:
            # last resort
            try:
                best.meta = {}  # type: ignore
                return best.meta  # type: ignore
            except Exception:
                return {}

    def _resolve_best_expert(self, best: ExpertDecision) -> str:
        """
        Prefer true expert name if available; avoid sticking to FALLBACK unless truly necessary.
        Order:
          1) best.expert if not empty
          2) best.meta["expert"]
          3) best.meta["selected_expert"]
        """
        expert = ""
        try:
            expert = str(getattr(best, "expert", "") or "").strip()
        except Exception:
            expert = ""

        meta = self._ensure_meta_dict(best)

        if not expert:
            v = meta.get("expert") or meta.get("selected_expert")
            if v:
                expert = str(v).strip()

        return expert

    def evaluate_trade(self, features: Dict[str, Any]) -> Tuple[bool, float, Dict[str, Any]]:
        """
        Returns: allow(bool), score(float), risk_cfg(dict)

        NOTE:
        - allow here means "system produced a decision event" (even if HOLD), not necessarily open position.
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

        # --- HARDEN: ensure meta dict exists ---
        meta = self._ensure_meta_dict(best)

        # Debug traces
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
                risk_cfg.setdefault("_regime_src", "DecisionEngine")
        except Exception:
            pass
        # ---------------------------------------

        # --- NEW: inject expert/action for online learning (OutcomeUpdater reads these) ---
        # Goal: weights.json -> expert_regime has real expert names, not all FALLBACK.
        try:
            if isinstance(risk_cfg, dict):
                best_expert = self._resolve_best_expert(best)
                best_action = ""
                try:
                    best_action = str(getattr(best, "action", "hold") or "hold")
                except Exception:
                    best_action = "hold"

                # Keep decision meta aligned too (useful if other parts read it)
                if best_expert:
                    meta.setdefault("expert", best_expert)
                meta.setdefault("action", best_action)
                meta.setdefault("regime", rr.regime)
                meta.setdefault("regime_conf", rr.confidence)

                # risk_cfg injection (do not override if upstream already set)
                if best_expert:
                    risk_cfg.setdefault("expert", best_expert)
                risk_cfg.setdefault("action", best_action)

                m = risk_cfg.get("meta")
                if not isinstance(m, dict):
                    m = {}
                    risk_cfg["meta"] = m
                if best_expert:
                    m.setdefault("expert", best_expert)
                m.setdefault("action", best_action)
                m.setdefault("regime", rr.regime)
                m.setdefault("regime_conf", rr.confidence)
        except Exception:
            pass
        # ------------------------------------------------------------------------------

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