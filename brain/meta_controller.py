# brain/meta_controller.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from brain.regime_detector import RegimeDetector, RegimeResult
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
    Meta-layer MoE gating:
      - Detect regime
      - Pick primary expert
      - Optional confirm experts adjust score
      - Optional veto expert can block
    """

    def __init__(
        self,
        enabled: bool = True,                      # ✅ ADD
        regime_detector: Optional[RegimeDetector] = None,
        experts: Optional[Dict[str, Any]] = None,
        store: Optional[WeightStore] = None,
        allow_threshold: float = 0.55,
    ) -> None:
        self.enabled = bool(enabled)               # ✅ ADD
        self.detector = regime_detector or RegimeDetector()
        self.experts: Dict[str, Any] = experts or {}
        self.store = store or WeightStore()
        self.allow_threshold = float(allow_threshold)

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

    def _get_candles(self, trade_features: Dict[str, Any]) -> List[Dict[str, Any]]:
        # support multiple legacy keys
        return (
            trade_features.get("candles")
            or trade_features.get("candles_window")
            or trade_features.get("window")
            or []
        )

    def evaluate(self, trade_features: Dict[str, Any]) -> MetaDecision:
        if not getattr(self, "enabled", True):
            return MetaDecision(
                allow=True,
                score=1.0,
                reasons=["meta_disabled"],
                regime="UNKNOWN",
                primary_expert="",
                confirm_experts=[],
                veto_expert=None,
                side="neutral",
                risk_cfg={},
            )
        candles = self._get_candles(trade_features)
        reg: RegimeResult = self.detector.detect(candles)

        primary_name, confirms, veto_name = self.policy.get(reg.regime, self.policy["MIXED"])
        primary = self._get_expert(primary_name)

        if primary is None:
            return MetaDecision(
                allow=False,
                score=0.0,
                reasons=[f"missing_primary:{primary_name}"],
                regime=reg.regime,
                primary_expert=primary_name,
                confirm_experts=[],
                veto_expert=veto_name,
                side="neutral",
                risk_cfg={},
            )

        # ExpertSignal-like object (we treat as duck-typed)
        p_sig = primary.evaluate(trade_features)
        p_side = str(getattr(p_sig, "side", "neutral"))
        p_conf = float(getattr(p_sig, "confidence", getattr(p_sig, "score", 0.0)) or 0.0)
        p_reasons = list(getattr(p_sig, "reasons", []) or [])
        p_risk = dict(getattr(p_sig, "risk_hint", {}) or {})

        reasons = [
            f"regime:{reg.regime}:{reg.confidence:.2f}",
            f"primary:{primary_name}:{p_side}:{p_conf:.2f}",
            *[f"p:{r}" for r in p_reasons],
        ]

        if p_side == "neutral" or p_conf <= 0.01:
            return MetaDecision(
                allow=False,
                score=float(p_conf),
                reasons=reasons + ["primary_neutral"],
                regime=reg.regime,
                primary_expert=primary_name,
                confirm_experts=[],
                veto_expert=veto_name,
                side="neutral",
                risk_cfg={},
            )

        # Confirm stage
        confirm_score = 0.0
        confirm_used: List[str] = []

        for cn in confirms:
            ex = self._get_expert(cn)
            if ex is None:
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

            if s_side == p_side:
                confirm_score += 0.25 * s_conf
            else:
                confirm_score -= 0.35 * s_conf

        # Veto stage
        veto_hit = False
        if veto_name:
            vx = self._get_expert(veto_name)
            if vx is not None:
                v = vx.evaluate(trade_features)
                v_side = str(getattr(v, "side", "neutral"))
                v_conf = float(getattr(v, "confidence", getattr(v, "score", 0.0)) or 0.0)
                v_reasons = list(getattr(v, "reasons", []) or [])

                reasons.append(f"veto:{veto_name}:{v_side}:{v_conf:.2f}")
                reasons += [f"v:{veto_name}:{r}" for r in v_reasons]

                if v_side != "neutral" and v_side != p_side and v_conf > 0.35:
                    veto_hit = True
                    reasons.append("veto_hit")

        # Score
        base = p_conf * (0.6 + 0.4 * float(reg.confidence))
        score = max(0.0, min(1.0, base + confirm_score))
        allow = (score >= self.allow_threshold) and (not veto_hit)

        risk_cfg = p_risk or {}
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
