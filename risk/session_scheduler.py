# risk/session_scheduler.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class SessionScheduler:
    every_n_steps: int = 1000

    def should_reset(self, step: int) -> bool:
        n = int(self.every_n_steps)
        if n <= 0:
            return False
        return (int(step) % n) == 0 and int(step) > 0
