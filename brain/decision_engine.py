# brain/decision_engine.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from brain.experts.expert_registry import ExpertRegistry
from brain.experts.expert_gate import ExpertGate, ExpertDecision

from brain.regime_detector import RegimeDetector, RegimeResult
from brain.weight_store import WeightStore

# Meta layer is optional (keep backward compat)
try:
    from brain.meta_controller import MetaController, MetaDecision
except Exception:
    MetaController = None  # type: ignore
    MetaDecision = Any  # type: ignore


class DecisionEngine:
    """
    Central decision engine:
    - builds expert registry
    - detects market regime (string)
    - asks ExpertGate to pick best ExpertDecision
    - optionally consults MetaController (MoE policy)
    """

    def __init__(
        self,
        risk_engine: Any,
        weight_store: Optional[WeightStore] = None,
        regime_detector: Optional[RegimeDetector] = None,
        meta_controller: Optional[Any] = None,
        score_threshold: float = 0.0,
        debug: bool = False,
    ) -> None:
        self.risk_engine = risk_engine
        self.weight_store = weight_store or WeightStore()
        self.regime_detector = regime_detector or RegimeDetector()

        # Keep attribute name stable across versions
        self.score_threshold = float(score_threshold)
        self.debug = bool(debug)

        self.registry = self._build_registry()
        self.gate = ExpertGate(registry=self.registry, weight_store=self.weight_store)

        # Meta is optional; fail-open handled inside MetaController
        self.meta = meta_controller
        if self.meta is None and MetaController is not None:
            try:
                # pass experts map if gate exposes it; otherwise create empty
                experts_map = {}
                try:
                    # If registry stores objects, expose them
                    experts_map = {e.name: e for e in self.registry.get_all()}
                except Exception:
                    experts_map = {}
                self.meta = MetaController(
                    regime_detector=self.regime_detector,
                    experts=experts_map,
                    store=self.weight_store,
                    enabled=True,
                    fail_open=True,
                )
            except Exception:
                self.meta = None

    def _build_registry(self) -> ExpertRegistry:
        reg = ExpertRegistry()

        # Default experts list is expected to exist
        try:
            from brain.experts.experts_basic import DEFAULT_EXPERTS
        except Exception:
            DEFAULT_EXPERTS = []  # type: ignore

        # Register by duck typing: each entry may be class or instance or dict
        for item in (DEFAULT_EXPERTS or []):
            try:
                if isinstance(item, dict):
                    name = str(item.get("name") or item.get("id") or "")
                    obj = item.get("expert")
                else:
                    obj = item() if isinstance(item, type) else item
                    name = str(getattr(obj, "name", obj.__class__.__name__))
                if not name or obj is None:
                    continue
                reg.register(name, obj)
            except Exception:
                continue

        # Backward compat: ensure at least 1 baseline expert exists
        if not reg.get_all():
            try:
                from brain.experts.experts_basic import BaselineExpert
                reg.register("BASELINE", BaselineExpert())
            except Exception:
                pass

        return reg

    def _extract_candles(self, trade_features: Dict[str, Any]) -> List[Dict[str, Any]]:
        # Accept many keys across versions
        return (
            trade_features.get("candles_window")
            or trade_features.get("candles")
            or trade_features.get("window")
            or trade_features.get("bars")
            or trade_features.get("rows")
            or []
        )

    def evaluate_trade(self, trade_features: Dict[str, Any]) -> Tuple[bool, float, Dict[str, Any]]:
        """
        Returns: (allow, score, risk_cfg)
        """
        candles = self._extract_candles(trade_features)

        # --- Regime detection (ALWAYS normalize to string) ---
        try:
            reg_res: RegimeResult = self.regime_detector.detect(candles) if candles else RegimeResult("unknown", 0.0, 0.0, 0.0)
        except Exception:
            reg_res = RegimeResult("unknown", 0.0, 0.0, 0.0)

        regime_name = getattr(reg_res, "regime", None) or "unknown"
        if not isinstance(regime_name, str):
            regime_name = str(regime_name)

        # --- Context for experts/gate ---
        context: Dict[str, Any] = {
            "regime": regime_name,          # IMPORTANT: string only
            "regime_result": reg_res,       # keep rich info
        }

        # --- Gate pick ---
        best, all_decisions = self.gate.pick(trade_features, context)

        # If gate returns None, do NOT crash; just deny safely
        if best is None:
            if self.debug:
                print("DEBUG BEST: None")
                print("DEBUG ALL_DECISIONS:", all_decisions)
            return False, 0.0, {"side": "neutral", "reason": "no_expert_decision", "regime": regime_name}

        # Normalize allow/score
        best_allow = bool(getattr(best, "allow", False))
        best_score = float(getattr(best, "score", 0.0) or 0.0)
        best_meta = getattr(best, "meta", {}) or {}
        best_action = getattr(best, "action", None)

        # --- Meta layer (optional, fail-open) ---
        meta_allow = True
        meta_score = 1.0
        meta_risk: Dict[str, Any] = {}
        meta_reasons: List[str] = []
        if self.meta is not None:
            try:
                md = self.meta.evaluate(trade_features)
                meta_allow = bool(getattr(md, "allow", True))
                meta_score = float(getattr(md, "score", 1.0) or 1.0)
                meta_risk = dict(getattr(md, "risk_cfg", {}) or {})
                meta_reasons = list(getattr(md, "reasons", []) or [])
                # normalize regime if meta detected
                try:
                    if getattr(md, "regime", None) and isinstance(getattr(md, "regime"), str):
                        regime_name = getattr(md, "regime")
                except Exception:
                    pass
            except Exception:
                meta_allow = True
                meta_score = 1.0

        # Blend score lightly (do not break old behavior)
        final_score = best_score
        try:
            final_score = float(0.85 * best_score + 0.15 * meta_score)
        except Exception:
            final_score = best_score

        # Final allow logic: keep old semantics but add threshold
        allow = bool(best_allow and meta_allow and (final_score >= self.score_threshold))

        # Risk cfg: risk engine can override; fallback keep meta_risk
        risk_cfg: Dict[str, Any] = {}
        try:
            # risk engine expected to provide config from decision/meta/features
            risk_cfg = self.risk_engine.build_config(trade_features, best, context)  # type: ignore
        except Exception:
            # fallback: use meta risk + best meta
            risk_cfg = {}
            risk_cfg.update(meta_risk or {})
            if isinstance(best_meta, dict):
                risk_cfg.setdefault("reason", best_meta.get("reason"))
            risk_cfg.setdefault("side", meta_risk.get("side") if isinstance(meta_risk, dict) else "neutral")
            if best_action is not None:
                risk_cfg.setdefault("action", best_action)

        # Attach debug / trace without breaking upstream
        risk_cfg.setdefault("regime", regime_name)
        risk_cfg.setdefault("expert", getattr(best, "expert", ""))
        risk_cfg.setdefault("score", final_score)
        if meta_reasons:
            risk_cfg.setdefault("meta_reasons", meta_reasons[:20])

        if self.debug:
            print("DEBUG REGIME:", regime_name, reg_res)
            print("DEBUG BEST:", best)
            print("DEBUG BEST_SCORE:", best_score, "META_SCORE:", meta_score, "FINAL:", final_score)
            print("DEBUG ALLOW:", allow, "THRESH:", self.score_threshold)

        return allow, float(final_score), risk_cfg
