# brain/experts/expert_registry.py
from __future__ import annotations

from typing import Dict, Iterable, List, Optional

from .expert_base import ExpertBase


class ExpertRegistry:
    """
    Registry of experts.

    Backward/Forward compatibility:
    - existing: all()
    - added: get_all() alias (some modules call get_all)
    - added: get(name), __iter__, __len__ for convenience
    """

    def __init__(self) -> None:
        self._experts: Dict[str, ExpertBase] = {}

    def register(self, expert: ExpertBase) -> None:
        name = getattr(expert, "name", None) or expert.__class__.__name__
        self._experts[str(name)] = expert

    def get(self, name: str) -> Optional[ExpertBase]:
        return self._experts.get(name)

    def all(self) -> List[ExpertBase]:
        return list(self._experts.values())

    # ✅ compatibility: ExpertGate expects get_all()
    def get_all(self):
        """
        Backward/forward compatible: return list of all expert instances.
        """
        if hasattr(self, "experts") and isinstance(self.experts, dict):
            return list(self.experts.values())
        if hasattr(self, "_experts") and isinstance(self._experts, dict):
            return list(self._experts.values())
        if hasattr(self, "items") and callable(getattr(self, "items")):
            # in case registry behaves like dict
            return [v for _, v in self.items()]
        return []

    def names(self) -> List[str]:
        return list(self._experts.keys())

    def __iter__(self) -> Iterable[ExpertBase]:
        return iter(self._experts.values())

    def __len__(self) -> int:
        return len(self._experts)

    def get_all(self):
        return list(self._experts.values())

    def build_default_registry() -> "ExpertRegistry":
        from brain.experts.experts_basic import TrendMAExpert, MeanRevertExpert, BreakoutExpert
        reg = ExpertRegistry()
        reg.register(TrendMAExpert())
        reg.register(MeanRevertExpert())
        reg.register(BreakoutExpert())
        return reg
