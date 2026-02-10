from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class ExpertDecision:
    allow: bool
    score: float
    expert: str
    meta: Dict[str, Any]


class BaseExpert:
    name: str = "BASE"

    def evaluate(self, trade_features: Dict[str, Any], context: Dict[str, Any]) -> ExpertDecision:
        raise NotImplementedError
