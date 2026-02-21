# brain/meta_controller.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

try:
    from brain.experts.expert_base import ExpertDecision
except Exception:
    @dataclass
    class ExpertDecision:  # type: ignore
        expert: str
        score: float = 0.0
        allow: bool = True
        action: str = "hold"
        meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MetaConfig:
    # Minimum score to "act" (buy/sell); below => hold
    act_threshold: float = 0.55

    # If confidence is low, force hold
    min_regime_conf: float = 0.15

    # Never global-deny; use hold instead
    deny_when_no_signal: bool = False

    # Debug print
    debug: bool = False


class MetaController:
    """
    Meta intelligence layer:
    - uses regime confidence & decision score
    - converts weak signals into HOLD rather than DENY
    """

    def __init__(self, cfg: Optional[MetaConfig] = None) -> None:
        self.cfg = cfg or MetaConfig()

    def apply(
        self,
        best: Optional[ExpertDecision],
        regime_result: Optional[Any],
        context: Dict[str, Any],
    ) -> ExpertDecision:
        # If no decision, create fallback HOLD
        if best is None:
            if self.cfg.debug:
                print("META: best=None -> fallback HOLD")
            return ExpertDecision(
                expert="META_FALLBACK",
                score=0.0,
                allow=(not self.cfg.deny_when_no_signal),
                action="hold",
                meta={"reason": "best_none"},
            )

        # Extract regime + confidence safely
        regime = None
        conf = 0.0
        try:
            regime = getattr(regime_result, "regime", None) or (regime_result.get("regime") if isinstance(regime_result, dict) else None)
        except Exception:
            regime = None
        try:
            conf = float(getattr(regime_result, "confidence", 0.0)) if regime_result is not None else 0.0
        except Exception:
            conf = 0.0

        # attach trace
        best.meta = best.meta or {}
        best.meta.setdefault("meta_regime", regime)
        best.meta.setdefault("meta_regime_conf", conf)
        best.meta.setdefault("meta_act_threshold", self.cfg.act_threshold)

        # If regime confidence too low -> HOLD (not deny)
        if conf < self.cfg.min_regime_conf:
            if self.cfg.debug:
                print(f"META: low regime conf={conf:.3f} -> HOLD")
            best.allow = True
            best.action = "hold"
            best.meta["meta_reason"] = "low_regime_conf_hold"
            return best

        # If score below act threshold -> HOLD (not deny)
        s = 0.0
        try:
            s = float(best.score)
        except Exception:
            s = 0.0

        if s < self.cfg.act_threshold:
            if self.cfg.debug:
                print(f"META: score={s:.3f} < thr={self.cfg.act_threshold:.3f} -> HOLD")
            best.allow = True
            best.action = "hold"
            best.meta["meta_reason"] = "below_threshold_hold"
            return best

        # Otherwise keep decision (buy/sell/hold as expert said)
        if self.cfg.debug:
            print(f"META: pass score={s:.3f} conf={conf:.3f}")
        best.allow = True  # always allow decision object to flow
        best.meta["meta_reason"] = "pass"
        return best
