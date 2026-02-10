from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple


def _key(expert: str, regime: Optional[str] = None) -> str:
    # Backward compatible: old key is just expert name.
    # New key supports regime-specific weights.
    if regime is None:
        return str(expert)
    return f"{regime}::{expert}"


@dataclass
class WeightStore:
    """
    Safe weight storage for experts (optionally per regime).
    - Backward compatible: get(expert) still works.
    - New: get(expert, regime) for regime-specific learning.
    """

    default_weight: float = 1.0
    min_weight: float = 0.10
    max_weight: float = 3.00
    _w: Dict[str, float] = field(default_factory=dict)

    # ---------- basic ops ----------
    def get(self, expert: str, regime: Optional[str] = None) -> float:
        return float(self._w.get(_key(expert, regime), self.default_weight))

    def set(self, expert: str, weight: float, regime: Optional[str] = None) -> None:
        w = float(weight)
        w = max(self.min_weight, min(self.max_weight, w))
        self._w[_key(expert, regime)] = w

    def update(self, expert: str, delta: float, regime: Optional[str] = None) -> float:
        cur = self.get(expert, regime)
        nxt = cur + float(delta)
        self.set(expert, nxt, regime)
        return self.get(expert, regime)

    def multiply(self, expert: str, factor: float, regime: Optional[str] = None) -> float:
        cur = self.get(expert, regime)
        nxt = cur * float(factor)
        self.set(expert, nxt, regime)
        return self.get(expert, regime)

    def items(self) -> Dict[str, float]:
        return dict(self._w)

    # ---------- persistence ----------
    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "default_weight": self.default_weight,
            "min_weight": self.min_weight,
            "max_weight": self.max_weight,
            "weights": self._w,
        }
        p.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "WeightStore":
        p = Path(path)
        if not p.exists():
            return cls()
        payload = json.loads(p.read_text(encoding="utf-8"))
        ws = cls(
            default_weight=float(payload.get("default_weight", 1.0)),
            min_weight=float(payload.get("min_weight", 0.10)),
            max_weight=float(payload.get("max_weight", 3.00)),
        )
        ws._w = {str(k): float(v) for k, v in dict(payload.get("weights", {})).items()}
        return ws
