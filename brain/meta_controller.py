# brain/meta_controller.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from brain.regime_detector import RegimeDetector, RegimeResult
from brain.experts.expert_base import ExpertSignal
from brain.weight_store import WeightStore

@dataclass
class MetaDecision:
    allow: bool
    score: float
    reasons: List[str]
    regime: str
    primary_expert: str
    confirm_experts: List[str]
    veto_expert: Optional[str]
    side: str  # long/short/neutral
    risk_cfg: Dict[str, Any]

class MetaController:
    """
    MoE gating:
      - Detect regime
      - Pick primary expert
      - Optionally confirm with another expert
      - Optionally veto if conflicts
      - Produce final allow/deny + score + reasons
    """

    def __init__(
        self,
        regime_detector: Optional[RegimeDetector] = None,
        experts: Optional[Dict[str, Any]] = None,
        store: Optional[WeightStore] = None,
        allow_threshold: float = 0.55,
    ):
        self.detector = regime_detector or RegimeDetector()
        self.experts = experts or {}
        self.store = store or WeightStore()
        self.allow_threshold = float(allow_threshold)

        # default mapping regime -> primary/confirm/veto
        self.policy = {
            "TREND_STRONG": ("TREND_MA", ["SMC_PLACEHOLDER"], "RANGE_ZSCORE"),
            "TREND_WEAK": ("TREND_MA", ["SMC_PLACEHOLDER"], "RANGE_ZSCORE"),
            "RANGE": ("RANGE_ZSCORE", ["SMC_PLACEHOLDER"], "TREND_MA"),
            "VOLATILITY_SPIKE": ("SMC_PLACEHOLDER", ["TREND_MA"], None),
            "MIXED": ("TREND_MA", ["RANGE_ZSCORE"], "SMC_PLACEHOLDER"),
            "UNKNOWN": ("TREND_MA", ["RANGE_ZSCORE"], None),
        }

    def _get_expert(self, name: str):
        return self.experts.get(name)

    def evaluate(self, trade_features: Dict[str, Any]) -> MetaDecision:
        w = trade_features.get("candles_window", [])
        reg: RegimeResult = self.detector.detect(w)

        primary_name, confirms, veto_name = self.policy.get(reg.regime, self.policy["MIXED"])
        primary = self._get_expert(primary_name)
        if primary is None:
            # fail-safe: deny
            return MetaDecision(
                allow=False,
                score=0.0,
                reasons=[f"missing_primary:{primary_name}"],
                regime=reg.regime,
                primary_expert=primary_name,
                confirm_experts=confirms,
                veto_expert=veto_name,
                side="neutral",
                risk_cfg={},
            )

        p_sig: ExpertSignal = primary.evaluate(trade_features)
        reasons = [f"regime:{reg.regime}:{reg.confidence:.2f}", f"primary:{primary_name}:{p_sig.side}:{p_sig.confidence:.2f}"]
        reasons += [f"p:{r}" for r in p_sig.reasons]

        # If primary neutral => deny
        if p_sig.side == "neutral" or p_sig.confidence <= 0.01:
            return MetaDecision(
                allow=False,
                score=p_sig.confidence,
                reasons=reasons + ["primary_neutral"],
                regime=reg.regime,
                primary_expert=primary_name,
                confirm_experts=confirms,
                veto_expert=veto_name,
                side="neutral",
                risk_cfg={},
            )

        # confirm stage
        confirm_score = 0.0
        confirm_used: List[str] = []
        for cn in confirms:
            ex = self._get_expert(cn)
            if ex is None:
                continue
            s = ex.evaluate(trade_features)
            confirm_used.append(cn)
            reasons.append(f"confirm:{cn}:{s.side}:{s.confidence:.2f}")
            reasons += [f"c:{cn}:{r}" for r in s.reasons]
            if s.side == "neutral":
                continue
            # add if aligns, subtract if conflicts
            if s.side == p_sig.side:
                confirm_score += 0.25 * s.confidence
            else:
                confirm_score -= 0.35 * s.confidence

        # veto stage
        veto_hit = False
        if veto_name:
            vx = self._get_expert(veto_name)
            if vx is not None:
                v = vx.evaluate(trade_features)
                reasons.append(f"veto:{veto_name}:{v.side}:{v.confidence:.2f}")
                reasons += [f"v:{veto_name}:{r}" for r in v.reasons]
                if v.side != "neutral" and v.side != p_sig.side and v.confidence > 0.35:
                    veto_hit = True
                    reasons.append("veto_hit")

        # base score: primary confidence * regime confidence (soft)
        base = p_sig.confidence * (0.6 + 0.4 * reg.confidence)
        score = max(0.0, min(1.0, base + confirm_score))

        allow = (score >= self.allow_threshold) and (not veto_hit)

        risk_cfg = p_sig.risk_hint or {}
        risk_cfg.setdefault("side", p_sig.side)

        return MetaDecision(
            allow=bool(allow),
            score=float(score),
            reasons=reasons,
            regime=reg.regime,
            primary_expert=primary_name,
            confirm_experts=confirm_used,
            veto_expert=veto_name,
            side=p_sig.side if allow else "neutral",
            risk_cfg=risk_cfg,
        )
