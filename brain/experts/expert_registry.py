# brain/experts/expert_registry.py
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

# IMPORTANT: do NOT import ExpertGate or DecisionEngine here (avoid circular)
from .expert_base import ExpertBase


class ExpertRegistry:
    """
    Holds expert instances.

    API intentionally small/stable:
      - register(expert)
      - get_all() -> List[ExpertBase]
      - get(name)
    """

    def __init__(self, experts: Optional[Iterable[ExpertBase]] = None) -> None:
        self._experts: Dict[str, ExpertBase] = {}
        if experts:
            for e in experts:
                self.register(e)

    def register(self, expert: ExpertBase) -> None:
        name = getattr(expert, "name", None)
        if not name:
            raise ValueError("Expert must have .name")
        self._experts[str(name)] = expert

    # alias for convenience
    add = register

    def get(self, name: str) -> Optional[ExpertBase]:
        return self._experts.get(name)

    def get_all(self) -> List[ExpertBase]:
        return list(self._experts.values())

    def __len__(self) -> int:
        return len(self._experts)

    def __iter__(self):
        return iter(self._experts.values())
