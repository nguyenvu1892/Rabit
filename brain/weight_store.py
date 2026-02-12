# brain/weight_store.py
from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return default
        return v
    except Exception:
        return default


@dataclass
class WeightUpdate:
    key: str
    prev: float
    new: float
    delta: float
    meta: Dict[str, Any]


class WeightStore:
    """
    Stores adaptive weights for:
      - session bucket weights (e.g. London, NY...)
      - pattern weights (e.g. Engulf...)
      - structure weights (e.g. BOS, FVG...)
      - trend/regime weights (e.g. range/trend/breakout...)
      - expert-regime pair weights: key = f"{expert}|{regime}"

    Design goals:
      - Backward compatible with previous versions
      - Works even if file missing/corrupted
      - Supports update, decay, normalization, snapshots
    """

    def __init__(
        self,
        path: Optional[str] = None,
        data: Optional[Dict[str, Any]] = None,
        clamp_min: float = 0.1,
        clamp_max: float = 10.0,
        decay: float = 0.9995,
        normalize: bool = True,
    ) -> None:
        self.path = str(path) if path else None
        self.clamp_min = float(clamp_min)
        self.clamp_max = float(clamp_max)
        self.decay = float(decay)
        self.normalize = bool(normalize)

        # main container
        self.weights: Dict[str, Any] = data if isinstance(data, dict) else {
            "session": {},
            "pattern": {},
            "structure": {},
            "trend": {},
            "expert_regime": {},  # expert|regime -> float
            "meta": {
                "created_at": time.time(),
                "updated_at": time.time(),
                "version": "5.1.7",
            },
        }

        # ensure required keys exist
        for k in ["session", "pattern", "structure", "trend", "expert_regime", "meta"]:
            if k not in self.weights or not isinstance(self.weights[k], dict):
                self.weights[k] = {} if k != "meta" else {"updated_at": time.time()}

        self._last_snapshot_ts: float = 0.0

    # -----------------------------
    # IO
    # -----------------------------
    @classmethod
    def load(cls, path: str, **kwargs) -> "WeightStore":
        """
        Preferred loader. Safe if file missing.
        """
        if not path:
            return cls(path=None, **kwargs)

        p = Path(path)
        if not p.exists():
            return cls(path=str(p), **kwargs)

        try:
            raw = p.read_text(encoding="utf-8")
            data = json.loads(raw) if raw.strip() else {}
            return cls(path=str(p), data=data, **kwargs)
        except Exception:
            # fallback: corrupted file -> start fresh but keep path
            return cls(path=str(p), **kwargs)

    def save(self, path: Optional[str] = None) -> None:
        out = str(path) if path else self.path
        if not out:
            return
        p = Path(out)
        p.parent.mkdir(parents=True, exist_ok=True)
        self.weights.setdefault("meta", {})
        self.weights["meta"]["updated_at"] = time.time()
        p.write_text(json.dumps(self.weights, indent=2, ensure_ascii=False), encoding="utf-8")

    # backward compat
    def load_json(self, path: Optional[str] = None) -> "WeightStore":
        """
        Backward compatible alias used by older scripts.
        Mutates self to load file.
        """
        src = path or self.path
        if not src:
            return self
        loaded = WeightStore.load(src, clamp_min=self.clamp_min, clamp_max=self.clamp_max, decay=self.decay, normalize=self.normalize)
        self.path = loaded.path
        self.weights = loaded.weights
        return self

    def save_json(self, path: Optional[str] = None) -> None:
        """
        Backward compatible alias used by older scripts.
        """
        self.save(path)

    def to_dict(self) -> Dict[str, Any]:
        """
        Used by shadow_run debug / eval reporter snapshot.
        """
        return self.weights

    # -----------------------------
    # Get/Set helpers
    # -----------------------------
    def _bucket(self, bucket: str) -> Dict[str, float]:
        b = self.weights.get(bucket)
        if not isinstance(b, dict):
            b = {}
            self.weights[bucket] = b
        return b  # type: ignore[return-value]

    def get(self, key: str, regime: Optional[str] = None, default: float = 1.0) -> float:
        """
        Unified getter:
          - if regime provided -> expert_regime weight with key = f"{key}|{regime}"
          - else try to find key in known buckets, else default
        """
        if regime:
            k = f"{key}|{regime}"
            v = self._bucket("expert_regime").get(k, default)
            return _safe_float(v, default)

        # search across buckets
        for bucket in ("session", "pattern", "structure", "trend"):
            b = self._bucket(bucket)
            if key in b:
                return _safe_float(b.get(key), default)

        # expert_regime by raw key if exists
        if key in self._bucket("expert_regime"):
            return _safe_float(self._bucket("expert_regime").get(key), default)

        return float(default)

    def set(self, bucket: str, key: str, value: float) -> None:
        b = self._bucket(bucket)
        b[key] = float(value)
        self.weights.setdefault("meta", {})
        self.weights["meta"]["updated_at"] = time.time()

    # -----------------------------
    # Learning / Update logic
    # -----------------------------
    def apply_decay(self) -> None:
        """
        Decay weights gently toward 1.0 to prevent runaway.
        """
        d = float(self.decay)
        if d <= 0.0 or d >= 1.0:
            return

        for bucket in ("session", "pattern", "structure", "trend", "expert_regime"):
            b = self._bucket(bucket)
            for k, v in list(b.items()):
                fv = _safe_float(v, 1.0)
                # move toward 1.0
                b[k] = 1.0 + (fv - 1.0) * d

    def clamp_all(self) -> None:
        mn, mx = self.clamp_min, self.clamp_max
        for bucket in ("session", "pattern", "structure", "trend", "expert_regime"):
            b = self._bucket(bucket)
            for k, v in list(b.items()):
                fv = _safe_float(v, 1.0)
                if fv < mn:
                    fv = mn
                if fv > mx:
                    fv = mx
                b[k] = fv

    def normalize_bucket(self, bucket: str) -> None:
        """
        Optional: normalize weights to have mean ~= 1.0
        """
        b = self._bucket(bucket)
        if not b:
            return
        vals = [_safe_float(v, 1.0) for v in b.values()]
        if not vals:
            return
        mean = sum(vals) / max(1, len(vals))
        if mean <= 0:
            return
        for k, v in list(b.items()):
            b[k] = _safe_float(v, 1.0) / mean

    def normalize_all(self) -> None:
        if not self.normalize:
            return
        for bucket in ("session", "pattern", "structure", "trend", "expert_regime"):
            self.normalize_bucket(bucket)

    def update_weight(
        self,
        bucket: str,
        key: str,
        delta: float,
        lr: float = 0.05,
        meta: Optional[Dict[str, Any]] = None,
    ) -> WeightUpdate:
        """
        Core update: w <- w + lr * delta
        delta should be positive for good outcomes, negative for bad outcomes.
        """
        b = self._bucket(bucket)
        prev = _safe_float(b.get(key, 1.0), 1.0)
        new = prev + float(lr) * float(delta)
        b[key] = new
        upd = WeightUpdate(key=key, prev=prev, new=new, delta=float(delta), meta=meta or {})
        self.weights.setdefault("meta", {})
        self.weights["meta"]["updated_at"] = time.time()
        return upd

    def update_expert_regime(
        self,
        expert: str,
        regime: str,
        delta: float,
        lr: float = 0.05,
        meta: Optional[Dict[str, Any]] = None,
    ) -> WeightUpdate:
        k = f"{expert}|{regime}"
        return self.update_weight("expert_regime", k, delta=delta, lr=lr, meta=meta)

    def maybe_snapshot(self, out_dir: str = "data", every_sec: int = 60) -> Optional[str]:
        """
        Periodic snapshot logging used earlier (5.0.8.6+).
        Writes: data/weights_snapshot_<ts>.json
        """
        now = time.time()
        if (now - self._last_snapshot_ts) < float(every_sec):
            return None
        self._last_snapshot_ts = now

        p = Path(out_dir) / f"weights_snapshot_{int(now)}.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.weights, indent=2, ensure_ascii=False), encoding="utf-8")
        return str(p)

    def size(self) -> int:
        """
        Count total number of scalar weights.
        """
        total = 0
        for bucket in ("session", "pattern", "structure", "trend", "expert_regime"):
            b = self._bucket(bucket)
            total += len(b)
        return total
