# brain/journal.py
from __future__ import annotations

import json
import os
from datetime import datetime


class Journal:
    def __init__(self, path: str):
        self.path = path
        folder = os.path.dirname(path)
        if folder:
            os.makedirs(folder, exist_ok=True)

    def _write(self, obj: dict):
        obj["ts_iso"] = datetime.utcnow().isoformat()
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(obj) + "\n")

    def log_decision(self, data: dict):
        data["type"] = "decision"
        self._write(data)

    def log_outcome(self, data: dict):
        data["type"] = "outcome"
        self._write(data)

    def log_heartbeat(self, data: dict):
        data["type"] = "heartbeat"
        self._write(data)
