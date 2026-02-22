# brain/weight_store.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional
import json
import os
import time


@dataclass
class WeightMeta:
    created_at: float = 0.0
    updated_at: float = 0.0
    version: str = "5.1.9"


class WeightStore:
    """
    Schema (KEEP):
    {
      "session": {...},
      "pattern": {...},
      "structure": {...},
      "trend": {...},
      "expert_regime": {...},
      "meta": {...}
    }
    """

    def __init__(self, data: Optional[Dict[str, Any]] = None) -> None:
        self.data: Dict[str, Any] = data or {}
        self._ensure_schema()

    # ----------------------------
    # Legacy behavior (KEEP)
    # ----------------------------
    def _ensure_schema(self) -> None:
        if not isinstance(self.data, dict):
            self.data = {}

        for k in ["session", "pattern", "structure", "trend", "expert_regime"]:
            v = self.data.get(k)
            if not isinstance(v, dict):
                self.data[k] = {}

        meta = self.data.get("meta")
        if not isinstance(meta, dict):
            meta = {}
            self.data["meta"] = meta

        # compat meta defaults (ADD but schema-safe)
        meta.setdefault("created_at", time.time())
        meta.setdefault("updated_at", time.time())
        meta.setdefault("version", "5.1.9")

    def get_bucket(self, bucket: str) -> Dict[str, float]:
        self._ensure_schema()
        b = self.data.get(bucket)
        if not isinstance(b, dict):
            b = {}
            self.data[bucket] = b
        return b  # type: ignore

    def get(self, bucket: str, key: str, default: float = 1.0) -> float:
        b = self.get_bucket(bucket)
        v = b.get(key, default)
        try:
            return float(v)
        except Exception:
            return float(default)

    def set(self, bucket: str, key: str, value: float) -> None:
        b = self.get_bucket(bucket)
        b[key] = float(value)
        self._touch()

    def update_add(self, bucket: str, key: str, delta: float) -> float:
        cur = self.get(bucket, key, default=1.0)
        nxt = float(cur) + float(delta)
        self.set(bucket, key, nxt)
        return nxt

    def keys(self, bucket: str) -> list[str]:
        b = self.get_bucket(bucket)
        return list(b.keys())

    def _touch(self) -> None:
        self._ensure_schema()
        meta = self.data["meta"]
        if isinstance(meta, dict):
            meta["updated_at"] = time.time()

    # ----------------------------
    # I/O (KEEP) + compat safety (ADD)
    # ----------------------------
    def save(self, path: str) -> None:
        self._ensure_schema()
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "WeightStore":
        if not path or not os.path.exists(path):
            return cls({})
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return cls(data if isinstance(data, dict) else {})
        except Exception:
            return cls({})

    # Convenience for existing code (KEEP API)
    def to_dict(self) -> Dict[str, Any]:
        self._ensure_schema()
        return dict(self.data)