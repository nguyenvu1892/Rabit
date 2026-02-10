# persistence/state_store.py
from __future__ import annotations

import json
import os
from typing import Optional

from persistence.state_bundle import CoreStateBundle


class CoreStateStore:
    def __init__(self, path: str):
        self.path = path

    def save(self, bundle: CoreStateBundle) -> None:
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(bundle.to_dict(), f, indent=2)

    def load(self) -> Optional[CoreStateBundle]:
        if not os.path.exists(self.path):
            return None
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                d = json.load(f)
            if not isinstance(d, dict):
                return None
            return CoreStateBundle.from_dict(d)
        except Exception:
            return None
