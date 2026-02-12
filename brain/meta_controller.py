# brain/meta_controller.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from brain.regime_detector import RegimeDetector, RegimeResult
from brain.weight_store import WeightStore

try:
    from brain.experts.expert_base import ExpertDecision  # only for typing/compat
except Exception:
    ExpertDecision = Any  # type: ignore


@dataclass
class MetaDecision:
    allow: bool
    score: float
    reasons: List[str]
    regime: str
    primary_expert: str
    confirm_experts: List[str]
    veto_expert: Optional[str]
    side: str
    risk_cfg: Dict[str, Any]


class MetaController:
    """
    Meta layer (MoE gating).

    IMPORTANT:
    - meta must NOT be able to hard-block whole engine just because missing candles / missing expert
    - default mode is fail-open (allow=True) when inputs insufficient
    """

    def __init__(
        self,
        regime_detector: Optional[RegimeDetector] = None,
        experts: Optional[Dict[str, Any]] = None,
        store: Optional[WeightStore] = None,
        allow_threshold: float = 0.55,
        enabled: bool = True,
        fail_open: bool = True,
    ) -> None:
        self.enabled = bool(enabled)
        self.fail_open = bool(fail_open)
        self.detector = regime_detector or RegimeDetector()
        self.experts = experts or {}
        self.store = store or WeightStore()
        self.allow_threshold = float(allow_threshold)

        # regime -> (primary, confirms[], veto)
        self.policy = {
            "breakout": ("TREND_MA", ["SMC_PLACEHOLDER"], "RANGE_ZSCORE"),
            "trend": ("TREND_MA", ["SMC_PLACEHOLDER"], "RANGE_ZSCORE"),
            "range": ("RANGE_ZSCORE", ["SMC_PLACEHOLDER"], "TREND_MA"),
            "unknown": ("TREND_MA", ["RANGE_ZSCORE"], None),
        }

    def _get_expert(self, name: str):
        return self.experts.get(name)

    def _get_candles(self, trade_features: Dict[str, Any]) -> List[Dict[str, Any]]:
        # accept many keys across versions
        return (
            trade_features.get("candles_window")
            or trade_features.get("candles")
            or trade_features.get("window")
            or trade_features.get("bars")
            or trade_features.get("rows")  # added compat
            or []
        )

    def evaluate(self, trade_features: Dict[str, Any]) -> MetaDecision:
        if not self.enabled:
            return MetaDecision(
                allow=True,
                score=1.0,
                reasons=["meta_disabled"],
                regime="unknown",
                primary_expert="",
                confirm_experts=[],
                veto_expert=None,
                side="neutral",
                risk_cfg={},
            )

        candles = self._get_candles(trade_features)
        if not candles or len(candles) < 50:
            # FAIL-OPEN: do not block engine when missing candles
            if self.fail_open:
                return MetaDecision(
                    allow=True,
                    score=1.0,
                    reasons=["meta_no_candles_fail_open"],
                    regime="unknown",
                    primary_expert="",
                    confirm_experts=[],
                    veto_expert=None,
                    side="neutral",
                    risk_cfg={},
                )
            return MetaDecision(
                allow=False,
                score=0.0,
                reasons=["meta_no_candles"],
                regime="unknown",
                primary_expert="",
                confirm_experts=[],
                veto_expert=None,
                side="neutral",
                risk_cfg={},
            )

        reg: RegimeResult = self.detector.detect(candles)
        primary_name, confirms, veto_name = self.policy.get(reg.regime, self.policy["unknown"])

        primary = self._get_expert(primary_name)
        if primary is None:
            # FAIL-OPEN
            if self.fail_open:
                return MetaDecision(
                    allow=True,
                    score=1.0,
                    reasons=[f"missing_primary:{primary_name}", f"regime:{reg.regime}:{reg.confidence:.2f}", "fail_open"],
                    regime=reg.regime,
                    primary_expert=primary_name,
                    confirm_experts=[],
                    veto_expert=veto_name,
                    side="neutral",
                    risk_cfg={},
                )
            return MetaDecision(
                allow=False,
                score=0.0,
                reasons=[f"missing_primary:{primary_name}", f"regime:{reg.regime}:{reg.confidence:.2f}"],
                regime=reg.regime,
                primary_expert=primary_name,
                confirm_experts=[],
                veto_expert=veto_name,
                side="neutral",
                risk_cfg={},
            )

        # Duck-typed signal: expert may implement evaluate() or decide()
        if hasattr(primary, "evaluate"):
            sig = primary.evaluate(trade_features)
            p_side = str(getattr(sig, "side", "neutral"))
            p_conf = float(getattr(sig, "confidence", getattr(sig, "score", 0.0)) or 0.0)
            p_reasons = list(getattr(sig, "reasons", []) or [])
            p_risk = dict(getattr(sig, "risk_hint", {}) or {})
        else:
            raw = primary.decide(trade_features, {"regime": reg.regime})
            d = raw if isinstance(raw, dict) else getattr(raw, "__dict__", {})
            p_side = str(d.get("side", "neutral"))
            p_conf = float(d.get("confidence", d.get("score", 0.0)) or 0.0)
            p_reasons = list(d.get("reasons", []) or [])
            p_risk = dict(d.get("risk_hint", {}) or {})

        reasons = [f"regime:{reg.regime}:{reg.confidence:.2f}", f"primary:{primary_name}:{p_side}:{p_conf:.2f}"]
        reasons += [f"p:{r}" for r in p_reasons]

        if p_side == "neutral" or p_conf <= 0.01:
            return MetaDecision(
                allow=True if self.fail_open else False,
                score=float(p_conf),
                reasons=reasons + ["primary_neutral_fail_open" if self.fail_open else "primary_neutral"],
                regime=reg.regime,
                primary_expert=primary_name,
                confirm_experts=[],
                veto_expert=veto_name,
                side="neutral",
                risk_cfg={},
            )

        confirm_score = 0.0
        confirm_used: List[str] = []
        for cn in confirms:
            ex = self._get_expert(cn)
            if ex is None or not hasattr(ex, "evaluate"):
                continue
            s = ex.evaluate(trade_features)
            s_side = str(getattr(s, "side", "neutral"))
            s_conf = float(getattr(s, "confidence", getattr(s, "score", 0.0)) or 0.0)
            s_reasons = list(getattr(s, "reasons", []) or [])
            confirm_used.append(cn)
            reasons.append(f"confirm:{cn}:{s_side}:{s_conf:.2f}")
            reasons += [f"c:{cn}:{r}" for r in s_reasons]
            if s_side == "neutral":
                continue
            confirm_score += (0.25 * s_conf) if (s_side == p_side) else (-0.35 * s_conf)

        veto_hit = False
        if veto_name:
            vx = self._get_expert(veto_name)
            if vx is not None and hasattr(vx, "evaluate"):
                v = vx.evaluate(trade_features)
                v_side = str(getattr(v, "side", "neutral"))
                v_conf = float(getattr(v, "confidence", getattr(v, "score", 0.0)) or 0.0)
                reasons.append(f"veto:{veto_name}:{v_side}:{v_conf:.2f}")
                if v_side != "neutral" and v_side != p_side and v_conf > 0.35:
                    veto_hit = True
                    reasons.append("veto_hit")

        base = float(p_conf) * (0.6 + 0.4 * float(reg.confidence))
        score = max(0.0, min(1.0, base + confirm_score))

        allow = (score >= self.allow_threshold) and (not veto_hit)
        if self.fail_open and (not allow):
            reasons.append("meta_recommend_deny_but_fail_open")
            allow = True

        risk_cfg = dict(p_risk or {})
        risk_cfg.setdefault("side", p_side)

        return MetaDecision(
            allow=bool(allow),
            score=float(score),
            reasons=reasons,
            regime=reg.regime,
            primary_expert=primary_name,
            confirm_experts=confirm_used,
            veto_expert=veto_name,
            side=p_side if allow else "neutral",
            risk_cfg=risk_cfg,
        )
