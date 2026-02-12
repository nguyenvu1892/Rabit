from __future__ import annotations

from typing import Any, Dict, List

from brain.experts.expert_base import BaseExpert, ExpertDecision


class BaselineExpert(BaseExpert):
    name = "BASELINE"

    def decide(self, features: Dict[str, Any], context: Dict[str, Any]) -> ExpertDecision:
        # Always produce a decision object; allow can be False depending on your gate threshold.
        # Score = 0 means neutral.
        return ExpertDecision(
            allow=False,
            score=0.0,
            expert=self.name,
            meta={"reason": "baseline_neutral"},
            action="hold",
        )


class TrendMAExpert(BaseExpert):
    name = "TREND_MA"

    def decide(self, features: Dict[str, Any], context: Dict[str, Any]) -> ExpertDecision:
        # Keep it simple but stable: if no signal -> deny with 0 score
        s = float(features.get("trend_score", 0.0) or 0.0)
        # Example: allow if abs signal > small epsilon
        allow = abs(s) > 1e-9
        action = "buy" if s > 0 else ("sell" if s < 0 else "hold")
        return ExpertDecision(
            allow=allow,
            score=s,
            expert=self.name,
            meta={"raw": s},
            action=action,
        )


class RangeMRExpert(BaseExpert):
    name = "RANGE_MR"

    def decide(self, features: Dict[str, Any], context: Dict[str, Any]) -> ExpertDecision:
        s = float(features.get("range_score", 0.0) or 0.0)
        allow = abs(s) > 1e-9
        action = "buy" if s > 0 else ("sell" if s < 0 else "hold")
        return ExpertDecision(
            allow=allow,
            score=s,
            expert=self.name,
            meta={"raw": s},
            action=action,
        )


class BreakoutExpert(BaseExpert):
    name = "BREAKOUT"

    def decide(self, features: Dict[str, Any], context: Dict[str, Any]) -> ExpertDecision:
        s = float(features.get("breakout_score", 0.0) or 0.0)
        allow = abs(s) > 1e-9
        action = "buy" if s > 0 else ("sell" if s < 0 else "hold")
        return ExpertDecision(
            allow=allow,
            score=s,
            expert=self.name,
            meta={"raw": s},
            action=action,
        )


DEFAULT_EXPERTS: List[BaseExpert] = [
    BaselineExpert(),
    TrendMAExpert(),
    RangeMRExpert(),
    BreakoutExpert(),
]


def register_basic_experts(registry: Any) -> None:
    """
    DecisionEngine expects this function.
    Registry implementations differ, so we support common method names.
    """
    experts = [TrendMAExpert(), MeanRevertExpert(), BreakoutExpert()]

    if hasattr(registry, "register"):
        for e in experts:
            registry.register(e)
        return

    # fallback: some registries use add()
    if hasattr(registry, "add"):
        for e in experts:
            registry.add(e)
        return

    # last resort: store on a known attribute
    if not hasattr(registry, "_experts"):
        setattr(registry, "_experts", [])
    registry._experts.extend(experts)
