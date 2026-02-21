# brain/weight_store.py
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


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

        self.weights: Dict[str, Any] = data if isinstance(data, dict) else {
            "session": {},
            "pattern": {},
            "structure": {},
            "trend": {},
            "expert_regime": {},  # expert|regime -> float
            "meta": {
                "created_at": time.time(),
                "updated_at": time.time(),
                "version": "5.1.8",
            },
        }

        for k in ["session", "pattern", "structure", "trend", "expert_regime", "meta"]:
            if k not in self.weights or not isinstance(self.weights[k], dict):
                self.weights[k] = {} if k != "meta" else {"updated_at": time.time()}

        self._last_snapshot_ts: float = 0.0

    # -----------------------------
    # IO
    # -----------------------------
    @classmethod
    def load(cls, path: str, **kwargs) -> "WeightStore":
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
        src = path or self.path
        if not src:
            return self
        loaded = WeightStore.load(
            src,
            clamp_min=self.clamp_min,
            clamp_max=self.clamp_max,
            decay=self.decay,
            normalize=self.normalize,
        )
        self.path = loaded.path
        self.weights = loaded.weights
        return self

    def save_json(self, path: Optional[str] = None) -> None:
        self.save(path)

    def to_dict(self) -> Dict[str, Any]:
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
        if regime:
            k = f"{key}|{regime}"
            v = self._bucket("expert_regime").get(k, default)
            return _safe_float(v, default)

        for bucket in ("session", "pattern", "structure", "trend"):
            b = self._bucket(bucket)
            if key in b:
                return _safe_float(b.get(key), default)

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
        d = float(self.decay)
        if d <= 0.0 or d >= 1.0:
            return
        for bucket in ("session", "pattern", "structure", "trend", "expert_regime"):
            b = self._bucket(bucket)
            for k, v in list(b.items()):
                fv = _safe_float(v, 1.0)
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
        b = self._bucket(bucket)
        prev = _safe_float(b.get(key, 1.0), 1.0)
        new = prev + float(lr) * float(delta)
        b[key] = new

        upd = WeightUpdate(
            key=key,
            prev=prev,
            new=new,
            delta=float(delta),
            meta=meta or {},
        )
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

    # ---------------------------------------------------------
    # ✅ NEW: Backward-compatible alias expected by OutcomeUpdater
    # OutcomeUpdater calls: weight_store.update(expert, regime, reward, autosave=..., log=False)
    # ---------------------------------------------------------
    def update(
        self,
        expert: str,
        regime: str,
        reward: float,
        *,
        lr: float = 0.05,
        autosave: bool = True,
        log: bool = False,
        meta: Optional[Dict[str, Any]] = None,
        apply_decay: bool = True,
        clamp: bool = True,
        normalize: bool = True,
        save_path: Optional[str] = None,
    ) -> WeightUpdate:
        """
        Safe online update entrypoint.
        - reward: already shaped in OutcomeUpdater (±)
        - lr: small to ensure stability
        """
        expert_s = str(expert) if expert is not None else ""
        regime_s = str(regime) if regime is not None else ""
        if not expert_s or not regime_s:
            # no-op but return a neutral update record
            return WeightUpdate(key=f"{expert_s}|{regime_s}", prev=1.0, new=1.0, delta=0.0, meta=meta or {})

        upd = self.update_expert_regime(expert_s, regime_s, delta=float(reward), lr=float(lr), meta=meta)

        # stability stack (lightweight)
        if apply_decay:
            self.apply_decay()
        if clamp:
            self.clamp_all()
        if normalize and self.normalize:
            self.normalize_all()

        if log:
            try:
                print(f"[WeightStore] update {upd.key}: {upd.prev:.4f} -> {upd.new:.4f} (delta={upd.delta:.4f})")
            except Exception:
                pass

        if autosave:
            try:
                self.save(save_path or self.path)
            except Exception:
                pass

        return upd

    def maybe_snapshot(self, out_dir: str = "data", every_sec: int = 60) -> Optional[str]:
        now = time.time()
        if (now - self._last_snapshot_ts) < float(every_sec):
            return None
        self._last_snapshot_ts = now
        p = Path(out_dir) / f"weights_snapshot_{int(now)}.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.weights, indent=2, ensure_ascii=False), encoding="utf-8")
        return str(p)

    def size(self) -> int:
        total = 0
        for bucket in ("session", "pattern", "structure", "trend", "expert_regime"):
            total += len(self._bucket(bucket))
        return total