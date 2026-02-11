from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Protocol


@dataclass
class ExpertDecision:
    def __init__(self, allow: bool, score: float, expert: str, meta=None):
        self.allow = allow
        self.score = score
        self.expert = expert
        self.meta = meta or {}
        
class ExpertBase(Protocol):
    """
    Backward-compatible base interface for experts.

    Some modules expect ExpertBase to exist (type hints / registry).
    Experts can implement any callable method signature as long as
    ExpertGate/DecisionEngine can call them consistently.
    """

    name: str

    def decide(self, trade_features: Dict[str, Any], context: Dict[str, Any]) -> ExpertDecision:
        raise NotImplementedError

class BaseExpert:
    name: str = "BASE"

    def evaluate(self, trade_features: Dict[str, Any], context: Dict[str, Any]) -> ExpertDecision:
        raise NotImplementedError
