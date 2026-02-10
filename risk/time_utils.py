# risk/time_utils.py
from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo
from typing import Any, Optional


VN_TZ = ZoneInfo("Asia/Ho_Chi_Minh")


def parse_ts_to_local_date(ts: Any) -> Optional[str]:
    """
    Accepts:
      - int/float unix seconds
      - ISO string (e.g. "2026-02-09T10:00:00Z")
      - datetime
    Returns local date string "YYYY-MM-DD" in Asia/Ho_Chi_Minh or None if cannot parse.
    """
    try:
        if ts is None:
            return None

        if isinstance(ts, datetime):
            dt = ts
        elif isinstance(ts, (int, float)):
            dt = datetime.fromtimestamp(float(ts), tz=VN_TZ)
            return dt.date().isoformat()
        elif isinstance(ts, str):
            s = ts.strip()
            # support trailing Z
            if s.endswith("Z"):
                s = s[:-1] + "+00:00"
            dt = datetime.fromisoformat(s)
        else:
            return None

        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=VN_TZ)
        dt_local = dt.astimezone(VN_TZ)
        return dt_local.date().isoformat()
    except Exception:
        return None
