# brain/journal.py
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class JournalEvent:
    type: str
    payload: Dict[str, Any]


class Journal:
    def __init__(self, path: Optional[str] = None) -> None:
        self.path = path

    def append(self, event: JournalEvent) -> None:
        if not self.path:
            return
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"type": event.type, **event.payload}, ensure_ascii=False) + "\n")

    # backward compatible helpers
    def log_decision(self, payload: Dict[str, Any]) -> None:
        self.append(JournalEvent(type="decision", payload=dict(payload)))

    def log_outcome(self, payload: Dict[str, Any]) -> None:
        self.append(JournalEvent(type="outcome", payload=dict(payload)))
