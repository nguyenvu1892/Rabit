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
        path: str | None = None,          # ✅ add
    ) -> None:
        self.init_weight = float(init_weight)
        self.min_w = float(min_w)
        self.max_w = float(max_w)
        self.default_regime = str(default_regime)
        self._w: Dict[str, Dict[str, float]] = {}

        # ✅ backward-compatible: allow WeightStore(path=...)
        if path:
            try:
                self.load_json(path)
            except Exception:
                pass


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
    def decay_toward(self, target: float = 1.0, rate: float = 0.002, regime: Optional[str] = None) -> None:
        """
        Pull weights slowly back toward target (default 1.0).
        rate small => stable. Suggested 0.001~0.01.
        """
        if rate <= 0:
            return

        def _decay_map(mp: Dict[str, float]) -> None:
            for k, v in list(mp.items()):
                v = float(v)
                v = v + (target - v) * float(rate)
                mp[k] = float(self._clamp(v))

        if regime is not None:
            mp = self.weights.get(regime)
            if mp:
                _decay_map(mp)
        else:
            for _, mp in self.weights.items():
                _decay_map(mp)

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

    def normalize_bucket(self, regime: str, target_mean: float = 1.0) -> None:
        """Normalize a regime bucket so mean weight ~= target_mean."""
        mp = self.weights.get(regime)
        if not mp:
            return
        vals = [float(v) for v in mp.values()]
        if not vals:
            return
        mean = sum(vals) / max(1, len(vals))
        if mean <= 0:
            return
        scale = float(target_mean) / float(mean)
        for k, v in list(mp.items()):
            mp[k] = float(self._clamp(float(v) * scale))

    def normalize_all_buckets(self, target_mean: float = 1.0) -> None:
        for regime in list(self.weights.keys()):
            self.normalize_bucket(regime, target_mean=target_mean)

    def topk(self, regime: str, k: int = 3) -> List[Tuple[str, float]]:
        mp = self.weights.get(regime) or {}
        items = sorted(((str(a), float(b)) for a, b in mp.items()), key=lambda x: x[1], reverse=True)
        return items[: max(1, k)]

    def bottomk(self, regime: str, k: int = 3) -> List[Tuple[str, float]]:
        mp = self.weights.get(regime) or {}
        items = sorted(((str(a), float(b)) for a, b in mp.items()), key=lambda x: x[1])
        return items[: max(1, k)]

    def normalize_bucket(
        self,
        bucket: str,
        target_mean: float = 1.0,
        min_w: float | None = None,
        max_w: float | None = None,
        eps: float = 1e-12,
    ) -> dict:
        """
        Normalize weights in a bucket so that mean ~= target_mean (default 1.0).
        Returns stats: {bucket, n, mean_before, mean_after, scale}
        """
        b = self.weights.get(bucket, {})
        if not isinstance(b, dict) or not b:
            return {"bucket": bucket, "n": 0, "mean_before": 0.0, "mean_after": 0.0, "scale": 1.0}

        vals = []
        for _, v in b.items():
            try:
                vals.append(float(v))
            except Exception:
                pass

        if not vals:
            return {"bucket": bucket, "n": 0, "mean_before": 0.0, "mean_after": 0.0, "scale": 1.0}

        mean_before = sum(vals) / max(1, len(vals))
        if abs(mean_before) < eps:
            return {"bucket": bucket, "n": len(vals), "mean_before": mean_before, "mean_after": mean_before, "scale": 1.0}

        scale = float(target_mean) / float(mean_before)

        # apply scaling
        for k, v in list(b.items()):
            try:
                b[k] = float(v) * scale
            except Exception:
                # keep original if can't cast
                pass

        # optional clamp after normalize (safe)
        if (min_w is not None) or (max_w is not None):
            self.clamp_bucket(bucket, min_w=min_w, max_w=max_w)

        # compute mean after
        vals2 = []
        for _, v in b.items():
            try:
                vals2.append(float(v))
            except Exception:
                pass
        mean_after = (sum(vals2) / max(1, len(vals2))) if vals2 else 0.0

        return {
            "bucket": bucket,
            "n": len(vals2),
            "mean_before": mean_before,
            "mean_after": mean_after,
            "scale": scale,
        }


    def clamp_bucket(self, bucket: str, min_w: float | None = None, max_w: float | None = None) -> None:
        b = self.weights.get(bucket, {})
        if not isinstance(b, dict) or not b:
            return
        for k, v in list(b.items()):
            try:
                x = float(v)
            except Exception:
                continue
            if min_w is not None and x < float(min_w):
                x = float(min_w)
            if max_w is not None and x > float(max_w):
                x = float(max_w)
            b[k] = x


    def stabilize_bucket(
        self,
        bucket: str,
        min_w: float | None = None,
        max_w: float | None = None,
        decay_rate: float = 0.0,
        target_mean: float = 1.0,
    ) -> dict:
        """
        5.0.8.7: clamp -> decay_toward_one -> normalize(mean->1.0) -> clamp again
        """
        # 1) clamp
        if (min_w is not None) or (max_w is not None):
            self.clamp_bucket(bucket, min_w=min_w, max_w=max_w)

        # 2) decay
        if decay_rate and decay_rate > 0:
            self.decay_toward_one(bucket, rate=float(decay_rate))

        # 3) normalize
        stats = self.normalize_bucket(bucket, target_mean=float(target_mean), min_w=min_w, max_w=max_w)

        return stats

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
