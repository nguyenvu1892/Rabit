# brain/weight_store.py
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

# add near top-level (after imports / class header ok)
REGIME_BUCKET = "__REGIME__"      # bucket name for regime-only prior
GLOBAL_BUCKET = "__GLOBAL__"      # optional future use

def _norm_key(x: str) -> str:
    return (x or "").strip()

def _now_ts() -> float:
    return time.time()


def _safe_float(x: Any, default: float = 1.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


@dataclass
class WeightStoreConfig:
    # hard clamp
    min_w: float = 0.20
    max_w: float = 5.00

    # learning rate (base)
    lr: float = 0.05

    # mild decay back toward 1.0 each update
    decay: float = 0.0005

    # periodic persistence
    autosave_every_updates: int = 50

    # logging
    log_every_updates: int = 200
    top_k_log: int = 5


class WeightStore:
    """
    WeightStore:
    - stores weights by bucket/key (nested dict)
    - supports pair-intelligence:
        mode="expert_regime": bucket=expert, key=regime   (legacy)
        mode="regime_expert": bucket=regime, key=expert   (regime-first)
    JSON format:
      {
        "bucketA": {"key1": 1.05, "key2": 0.97},
        "bucketB": {"keyX": 1.20}
      }
    """

    def __init__(
        self,
        path: Optional[str] = None,
        *,
        cfg: Optional[WeightStoreConfig] = None,
        min_w: Optional[float] = None,
        max_w: Optional[float] = None,
        lr: Optional[float] = None,
        decay: Optional[float] = None,
        autosave_every_updates: Optional[int] = None,
        log_every_updates: Optional[int] = None,
        top_k_log: Optional[int] = None,
        **_kwargs: Any,  # accept unknown args to avoid breaking callers
    ) -> None:
        self.path = path
        self.cfg = cfg or WeightStoreConfig()

        # allow overrides
        if min_w is not None:
            self.cfg.min_w = float(min_w)
        if max_w is not None:
            self.cfg.max_w = float(max_w)
        if lr is not None:
            self.cfg.lr = float(lr)
        if decay is not None:
            self.cfg.decay = float(decay)
        if autosave_every_updates is not None:
            self.cfg.autosave_every_updates = int(autosave_every_updates)
        if log_every_updates is not None:
            self.cfg.log_every_updates = int(log_every_updates)
        if top_k_log is not None:
            self.cfg.top_k_log = int(top_k_log)

        self._w: Dict[str, Dict[str, float]] = {}
        self._updates_since_save = 0
        self._updates_since_log = 0
        self._last_loaded_ts: Optional[float] = None

        if self.path and os.path.exists(self.path):
            self.load(self.path)

    # ------------------------
    # Compatibility / helpers
    # ------------------------
    @property
    def weights(self) -> Dict[str, Dict[str, float]]:
        # backward compat
        return self._w

    def __len__(self) -> int:
        return sum(len(sub) for sub in self._w.values())

    def to_dict(self) -> Dict[str, Dict[str, float]]:
        out: Dict[str, Dict[str, float]] = {}
        for bucket, mp in self._w.items():
            out[str(bucket)] = {str(k): float(v) for k, v in mp.items()}
        return out

    as_dict = to_dict  # alias

    def bucket_mean(self, bucket: str, default: float = 1.0) -> float:
        b = self._w.get(str(bucket))
        if not b:
            return float(default)
        vals = [float(_safe_float(v, 1.0)) for v in b.values()]
        if not vals:
            return float(default)
        return float(sum(vals) / max(1, len(vals)))

    # ------------------------
    # Persistence
    # ------------------------
    def load(self, path: Optional[str] = None) -> None:
        p = path or self.path
        if not p or not os.path.exists(p):
            return

        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f) or {}

        out: Dict[str, Dict[str, float]] = {}
        if isinstance(data, dict):
            for bucket, sub in data.items():
                if isinstance(sub, dict):
                    out[str(bucket)] = {str(k): _safe_float(v, 1.0) for k, v in sub.items()}

        self._w = out
        self._last_loaded_ts = _now_ts()

    def save(self, path: Optional[str] = None) -> None:
        p = path or self.path
        if not p:
            return
        os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(self._w, f, indent=2, sort_keys=True)
        self._updates_since_save = 0

    def load_json(self, path: str) -> bool:
        try:
            self.load(path)
            return True
        except Exception:
            return False

    def save_json(self, path: str) -> bool:
        try:
            self.save(path)
            return True
        except Exception:
            return False

    # ------------------------
    # Core accessors
    # ------------------------
    def get(self, bucket: str, key: str, default: float = 1.0) -> float:
        b = self._w.get(str(bucket))
        if not b:
            return float(default)
        return _safe_float(b.get(str(key), default), default)

    def set(self, bucket: str, key: str, value: float) -> None:
        b = self._w.setdefault(str(bucket), {})
        b[str(key)] = self._clamp(float(value))

    def ensure(self, bucket: str, key: str, default: float = 1.0) -> float:
        b = self._w.setdefault(str(bucket), {})
        k = str(key)
        if k not in b:
            b[k] = self._clamp(float(default))
        return b[k]

    # ------------------------
    # Pair helpers (5.1.4)
    # ------------------------
    @staticmethod
    def _pair_bucket_key(expert: str, regime: str, mode: str) -> Tuple[str, str]:
        m = (mode or "expert_regime").lower().strip()
        if m in ("regime_expert", "regime-first", "regime_first"):
            return str(regime), str(expert)
        # default / legacy
        return str(expert), str(regime)

    def get_pair(self, expert: str, regime: str, default: float = 1.0, *, mode: str = "expert_regime") -> float:
        b, k = self._pair_bucket_key(expert, regime, mode)
        return self.get(b, k, default=default)

    def update_pair(
        self,
        expert: str,
        regime: str,
        reward: float,
        *,
        mode: str = "expert_regime",
        mirror: bool = True,
        normalize_bucket: bool = False,
        target_mean: float = 1.0,
        autosave: Optional[bool] = None,
        log: bool = False,
    ) -> float:
        # update primary
        b, k = self._pair_bucket_key(expert, regime, mode)
        w1 = self.update(
            b,
            k,
            reward,
            normalize_bucket=normalize_bucket,
            target_mean=target_mean,
            autosave=autosave,
            log=log,
        )

        # update mirror direction so both expert-first and regime-first can read
        if mirror:
            b2, k2 = (str(regime), str(expert)) if (b, k) == (str(expert), str(regime)) else (str(expert), str(regime))
            # mirror update WITHOUT extra logging spam
            self.update(
                b2,
                k2,
                reward,
                normalize_bucket=normalize_bucket,
                target_mean=target_mean,
                autosave=autosave,
                log=False,
            )
        return w1

    # ------------------------
    # Learning update
    # ------------------------
    def update(
        self,
        bucket: str,
        key: str,
        reward: float,
        *,
        lr: Optional[float] = None,
        decay: Optional[float] = None,
        min_w: Optional[float] = None,
        max_w: Optional[float] = None,
        normalize_bucket: bool = False,
        target_mean: float = 1.0,
        autosave: Optional[bool] = None,
        log: bool = False,
    ) -> float:
        _lr = float(lr) if lr is not None else float(self.cfg.lr)
        _decay = float(decay) if decay is not None else float(self.cfg.decay)
        _min = float(min_w) if min_w is not None else float(self.cfg.min_w)
        _max = float(max_w) if max_w is not None else float(self.cfg.max_w)

        b = str(bucket)
        k = str(key)

        cur = self.ensure(b, k, default=1.0)
        r = _safe_float(reward, 0.0)

        nxt = cur + (_lr * r)
        nxt = nxt + (_decay * (1.0 - nxt))
        nxt = max(_min, min(_max, float(nxt)))
        self._w[b][k] = nxt

        if normalize_bucket:
            self._normalize_bucket(b, target_mean=float(target_mean), min_w=_min, max_w=_max)

        self._updates_since_save += 1
        self._updates_since_log += 1

        do_autosave = (
            self.cfg.autosave_every_updates > 0
            and (self._updates_since_save >= self.cfg.autosave_every_updates)
        )
        if autosave is not None:
            do_autosave = bool(autosave)
        if do_autosave:
            self.save()

        do_log = log or (
            self.cfg.log_every_updates > 0
            and (self._updates_since_log >= self.cfg.log_every_updates)
        )
        if do_log:
            self._updates_since_log = 0
            self.log_summary(title=f"WeightStore summary (last update: {b}/{k})")

        return nxt

    # ------------------------
    # Stabilization helpers
    # ------------------------
    def clamp_all(self, *, min_w: Optional[float] = None, max_w: Optional[float] = None) -> None:
        _min = float(min_w) if min_w is not None else float(self.cfg.min_w)
        _max = float(max_w) if max_w is not None else float(self.cfg.max_w)
        for _, sub in self._w.items():
            for k, v in list(sub.items()):
                sub[k] = max(_min, min(_max, _safe_float(v, 1.0)))

    def decay_all(self, rate: Optional[float] = None) -> None:
        r = float(rate) if rate is not None else float(self.cfg.decay)
        if r <= 0:
            return
        for _, sub in self._w.items():
            for k, v in list(sub.items()):
                vv = _safe_float(v, 1.0)
                vv = vv + r * (1.0 - vv)
                sub[k] = self._clamp(vv)

    def normalize_all(self, target_mean: float = 1.0) -> None:
        for b in list(self._w.keys()):
            self._normalize_bucket(b, target_mean=target_mean)

    def _normalize_bucket(
        self,
        bucket: str,
        *,
        target_mean: float = 1.0,
        min_w: Optional[float] = None,
        max_w: Optional[float] = None,
    ) -> None:
        b = self._w.get(bucket)
        if not b:
            return
        vals = [float(_safe_float(v, 1.0)) for v in b.values()]
        if not vals:
            return
        cur_mean = sum(vals) / max(1, len(vals))
        if cur_mean <= 1e-9:
            return
        ratio = float(target_mean) / cur_mean
        _min = float(min_w) if min_w is not None else float(self.cfg.min_w)
        _max = float(max_w) if max_w is not None else float(self.cfg.max_w)
        for k, v in list(b.items()):
            nv = float(_safe_float(v, 1.0)) * ratio
            b[k] = max(_min, min(_max, nv))

    def _clamp(self, v: float) -> float:
        return max(float(self.cfg.min_w), min(float(self.cfg.max_w), float(v)))

    # ------------------------
    # Logging
    # ------------------------
    def log_summary(self, title: str = "WeightStore summary") -> None:
        flat: list[Tuple[str, str, float]] = []
        for b, sub in self._w.items():
            for k, v in sub.items():
                flat.append((b, k, _safe_float(v, 1.0)))

        if not flat:
            print(f"[WeightStore] {title}: (empty)")
            return

        flat.sort(key=lambda x: x[2], reverse=True)
        kk = max(1, int(self.cfg.top_k_log))
        top = flat[:kk]
        bottom = list(reversed(flat[-kk:]))

        print(f"\n[WeightStore] {title}")
        print(" Top:")
        for b, k, v in top:
            print(f" {b} / {k} = {v:.4f}")
        print(" Bottom:")
        for b, k, v in bottom:
            print(f" {b} / {k} = {v:.4f}")
        print("")

    def get_regime_prior(self, regime: str, default: float = 1.0) -> float:
        """Regime-only prior weight."""
        return float(self.get(REGIME_BUCKET, _norm_key(regime), default=default))

    def update_regime_prior(self, regime: str, delta: float, **kwargs) -> float:
        """Update regime-only prior weight using same update() pipeline."""
        return float(self.update(REGIME_BUCKET, _norm_key(regime), delta, **kwargs))

    def combined_weight(self, expert: str, regime: str, *, beta: float = 0.35, default: float = 1.0) -> float:
        """
        Combine pair weight and regime prior:
        w = w_pair * (w_regime ** beta)
        beta nhỏ để regime prior chỉ là bias nhẹ (ổn định).
        """
        w_pair = float(self.get(_norm_key(expert), _norm_key(regime), default=default))
        w_reg  = float(self.get_regime_prior(_norm_key(regime), default=default))
        if w_reg <= 0:
            w_reg = default
        return w_pair * (w_reg ** float(beta))
