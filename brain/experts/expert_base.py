# brain/experts/expert_base.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class ExpertDecision:
    """
    Canonical decision object used across ExpertGate/DecisionEngine.

    NOTE:
    - Keep fields stable (API/schema).
    - COMPAT: accept legacy callers that pass risk_cfg=... into ExpertDecision().
    """
    allow: bool = False
    score: float = 0.0
    expert: str = "UNKNOWN"
    meta: Dict[str, Any] = field(default_factory=dict)

    # ---- COMPAT BLOCK (do not remove old fields) -----------------------------
    # Some older code paths constructed ExpertDecision(..., risk_cfg={...}).
    # We keep schema backward-compatible by adding an optional field.
    risk_cfg: Dict[str, Any] = field(default_factory=dict)
    # -------------------------------------------------------------------------

    @staticmethod
    def coerce(obj: Any) -> "ExpertDecision":
        """
        Convert a dict/ExpertDecision/None into ExpertDecision safely.
        """
        if isinstance(obj, ExpertDecision):
            return obj
        if isinstance(obj, dict):
            d = obj
            return ExpertDecision(
                allow=bool(d.get("allow", False)),
                score=float(d.get("score", 0.0) or 0.0),
                expert=str(d.get("expert", "UNKNOWN") or "UNKNOWN"),
                meta=d.get("meta") if isinstance(d.get("meta"), dict) else {},
                risk_cfg=d.get("risk_cfg") if isinstance(d.get("risk_cfg"), dict) else {},
            )
        return ExpertDecision()


class BaseExpert:
    """
    Base class for experts.
    """

    name: str = "BASE"

    def decide(self, features: Dict[str, Any]) -> ExpertDecision:
        """
        Return ExpertDecision. Override in child classes.
        """
        return ExpertDecision(allow=False, score=0.0, expert=getattr(self, "name", "BASE"), meta={})

    # ---- COMPAT BLOCK --------------------------------------------------------
    # Some code used evaluate() naming.
    def evaluate(self, features: Dict[str, Any]) -> ExpertDecision:
        return self.decide(features)
    # -------------------------------------------------------------------------


# Alias for older imports
ExpertBase = BaseExpert