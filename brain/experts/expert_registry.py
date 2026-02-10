from __future__ import annotations
from typing import Dict, List
from brain.experts.expert_base import BaseExpert


class ExpertRegistry:
    def __init__(self) -> None:
        self._experts: List[BaseExpert] = []

    def register(self, expert: BaseExpert) -> None:
        self._experts.append(expert)

    def all(self) -> List[BaseExpert]:
        return list(self._experts)

    def names(self) -> List[str]:
        return [e.name for e in self._experts]
