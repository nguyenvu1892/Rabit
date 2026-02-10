# brain/strategy_store.py
from __future__ import annotations

import json
import os
import time
from typing import Dict, Any, Optional


class StrategyStore:
    def __init__(self, path: str = "data/best_strategy.json"):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)

    def save(self, genome: Dict[str, Any], meta: Optional[Dict[str, Any]] = None) -> None:
        payload = {
            "version": 1,
            "timestamp": int(time.time()),
            "genome": genome,
            "meta": meta or {},
        }
        tmp_path = self.path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, self.path)  # atomic on most OS

    def load(self) -> Optional[Dict[str, Any]]:
        if not os.path.exists(self.path):
            return None
        with open(self.path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        return payload.get("genome")

        def save_as(self, genome: Dict[str, Any], path: str, meta: Optional[Dict[str, Any]] = None) -> None:
            old_path = self.path
        try:
            self.path = path
            self.save(genome, meta=meta)
        finally:
            self.path = old_path

    def load_from(self, path: str) -> Optional[Dict[str, Any]]:
        import json, os
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        return payload.get("genome")
