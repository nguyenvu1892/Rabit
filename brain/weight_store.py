# brain/weight_store.py
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List


def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


@dataclass
class WeightUpdate:
    regime: str
    expert: str
    old: float
    new: float
    delta: float
    reason: str


class WeightStore:
    """
    Stable, standalone weight store (stdlib-only) to avoid circular imports.

    Layout:
        self._w[regime][expert] = weight

    Backward compatible behaviors:
      - get(expert) works (uses regime="global")
      - set(expert, value) works (regime="global")
      - update(expert, delta, ...) works (regime="global")
      - get(expert, regime) also works
    """

    def __init__(
        self,
        init_weight: float = 1.0,
        min_w: float = 0.2,
        max_w: float = 5.0,
        default_regime: str = "global",
    ) -> None:
        self.init_weight = float(init_weight)
        self.min_w = float(min_w)
        self.max_w = float(max_w)
        self.default_regime = str(default_regime)
        self._w: Dict[str, Dict[str, float]] = {}

    # -----------------------------
    # Core access
    # -----------------------------
    def get(self, expert: str, regime: Optional[str] = None, default: Optional[float] = None) -> float:
        r = self.default_regime if regime is None else str(regime)
        d = self.init_weight if default is None else float(default)
        return float(self._w.get(r, {}).get(str(expert), d))

    def set(self, expert: str, value: float, regime: Optional[str] = None) -> None:
        r = self.default_regime if regime is None else str(regime)
        e = str(expert)
        self._w.setdefault(r, {})[e] = _clamp(float(value), self.min_w, self.max_w)

    def bump(
        self,
        expert: str,
        delta: float,
        regime: Optional[str] = None,
        reason: str = "bump",
    ) -> WeightUpdate:
        r = self.default_regime if regime is None else str(regime)
        e = str(expert)
        old = self.get(e, r)
        new = _clamp(old + float(delta), self.min_w, self.max_w)
        self.set(e, new, r)
        return WeightUpdate(regime=r, expert=e, old=old, new=new, delta=float(delta), reason=reason)

    def mul(
        self,
        expert: str,
        factor: float,
        regime: Optional[str] = None,
        reason: str = "mul",
    ) -> WeightUpdate:
        r = self.default_regime if regime is None else str(regime)
        e = str(expert)
        old = self.get(e, r)
        new = _clamp(old * float(factor), self.min_w, self.max_w)
        self.set(e, new, r)
        return WeightUpdate(regime=r, expert=e, old=old, new=new, delta=new - old, reason=reason)

    # -----------------------------
    # Stabilizers (5.0.8.3+)
    # -----------------------------
    def decay_toward(
        self,
        target: float = 1.0,
        rate: float = 0.01,
        regime: Optional[str] = None,
    ) -> None:
        """
        Softly decay weights toward target to avoid runaway overfit.
        w := w + rate * (target - w)
        """
        r = self.default_regime if regime is None else str(regime)
        if r not in self._w:
            return
        t = float(target)
        a = float(rate)
        for e, w in list(self._w[r].items()):
            nw = _clamp(w + a * (t - w), self.min_w, self.max_w)
            self._w[r][e] = nw

    def normalize_mean(
        self,
        regime: Optional[str] = None,
        target_mean: float = 1.0,
    ) -> None:
        """
        Scale all weights so that mean becomes target_mean.
        Useful to keep the bucket stable.
        """
        r = self.default_regime if regime is None else str(regime)
        bucket = self._w.get(r)
        if not bucket:
            return
        vals = list(bucket.values())
        if not vals:
            return
        mean = sum(vals) / max(1, len(vals))
        if mean <= 0:
            return
        scale = float(target_mean) / mean
        for e, w in list(bucket.items()):
            bucket[e] = _clamp(w * scale, self.min_w, self.max_w)

    def topk(self, k: int = 5, regime: Optional[str] = None) -> List[Tuple[str, float]]:
        r = self.default_regime if regime is None else str(regime)
        bucket = self._w.get(r, {})
        return sorted(bucket.items(), key=lambda x: x[1], reverse=True)[: max(0, int(k))]

    def bottomk(self, k: int = 5, regime: Optional[str] = None) -> List[Tuple[str, float]]:
        r = self.default_regime if regime is None else str(regime)
        bucket = self._w.get(r, {})
        return sorted(bucket.items(), key=lambda x: x[1])[: max(0, int(k))]

    # -----------------------------
    # Persistence
    # -----------------------------
    def to_dict(self) -> Dict[str, Dict[str, float]]:
        # deep copy-ish
        return {r: dict(b) for r, b in self._w.items()}

    def load_dict(self, data: Dict[str, Dict[str, float]]) -> None:
        self._w = {}
        for r, bucket in (data or {}).items():
            rr = str(r)
            self._w[rr] = {}
            for e, w in (bucket or {}).items():
                self._w[rr][str(e)] = _clamp(float(w), self.min_w, self.max_w)

    def load_json(self, path: str) -> None:
        if not path:
            return
        if not os.path.exists(path):
            return
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)

        # allow both formats:
        # (A) {"regime": {"expert": w}}
        # (B) {"session": {...}, "pattern": {...}}  (your earlier structure)
        if isinstance(obj, dict) and all(isinstance(v, dict) for v in obj.values()):
            self.load_dict(obj)  # type: ignore[arg-type]
            return

        # fallback: try interpret as legacy structure
        data: Dict[str, Dict[str, float]] = {}
        if isinstance(obj, dict):
            for r, bucket in obj.items():
                if isinstance(bucket, dict):
                    data[str(r)] = {str(e): float(w) for e, w in bucket.items()}
        self.load_dict(data)

    def save_json(self, path: str) -> None:
        if not path:
            return
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
