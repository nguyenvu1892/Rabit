# risk/session_scheduler_day.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from risk.time_utils import parse_ts_to_local_date


@dataclass
class DaySessionScheduler:
    """
    Resets when candle timestamp crosses a local-day boundary (Asia/Ho_Chi_Minh).
    """
    last_day: Optional[str] = None

    def should_reset(self, candle: dict) -> bool:
        ts = candle.get("ts") if isinstance(candle, dict) else None
        day = parse_ts_to_local_date(ts)
        if day is None:
            return False

        if self.last_day is None:
            self.last_day = day
            return False

        if day != self.last_day:
            self.last_day = day
            return True

        return False

    def get_state(self) -> dict:
        return {"last_day": self.last_day}

    def set_state(self, state: dict) -> None:
        s = state or {}
        self.last_day = s.get("last_day", None)
