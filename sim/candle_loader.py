# sim/candle_loader.py
from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class Candle:
    ts: int  # unix seconds
    o: float
    h: float
    l: float
    c: float
    v: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {"ts": self.ts, "o": self.o, "h": self.h, "l": self.l, "c": self.c, "v": self.v}


# Common column aliases (normalized by _norm: lowercase + strip <>)
_COL_ALIASES = {
    "time": {"time", "timestamp", "datetime", "dt"},
    "open": {"open", "o"},
    "high": {"high", "h"},
    "low": {"low", "l"},
    "close": {"close", "c"},
    "volume": {"volume", "vol", "v", "tick_volume", "tickvol"},
}


def _norm(s: str) -> str:
    """
    Normalize header/value tokens:
    - trim spaces
    - lowercase
    - strip MT5 angle brackets: <OPEN> -> open
    """
    x = (s or "").strip().lower()
    x = x.strip("<>").strip()
    return x


def _detect_delimiter(sample: str) -> str:
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=[",", ";", "\t", "|"])
        return dialect.delimiter
    except Exception:
        return ","


def _parse_ts(value: str) -> int:
    """
    Accept:
      - unix seconds / milliseconds
      - ISO strings: '2025-01-01 12:30:00' or '2025-01-01T12:30:00Z'
      - MT5 date-time joined: '2025.01.01 00:05'
    Output unix seconds (int, UTC).
    """
    v = (value or "").strip()
    if v == "":
        raise ValueError("empty timestamp")

    # numeric unix?
    try:
        fv = float(v)
        iv = int(fv)
        if iv > 10_000_000_000:  # treat as ms
            return int(iv / 1000)
        return iv
    except Exception:
        pass

    # MT5 date format 2025.01.01 -> convert to 2025-01-01 for parsing
    v = v.replace(".", "-")

    # ISO-like parse
    v2 = v.replace("Z", "+00:00")
    dt: Optional[datetime] = None

    try:
        dt = datetime.fromisoformat(v2)
    except Exception:
        fmts = [
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d %H:%M",
            "%Y-%m-%d",
        ]
        for f in fmts:
            try:
                dt = datetime.strptime(v, f)
                break
            except Exception:
                continue

    if dt is None:
        raise ValueError(f"cannot parse timestamp: {value}")

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp())


def _find_col(headers: List[str], target: str) -> Optional[str]:
    aliases = _COL_ALIASES[target]
    for h in headers:
        if _norm(h) in aliases:
            return h
    # accept exact target name too
    for h in headers:
        if _norm(h) == target:
            return h
    return None


def load_candles_csv(
    path: str,
    *,
    limit: Optional[int] = None,
    sort_by_ts: bool = True,
    drop_duplicates: bool = True,
) -> List[Dict[str, Any]]:
    """
    Load XAUUSD historical CSV into standard candle dicts:
      {"ts": int, "o": float, "h": float, "l": float, "c": float, "v": float}

    Supports MT5 export:
      <DATE>,<TIME>,<OPEN>,<HIGH>,<LOW>,<CLOSE>,<TICKVOL>,<VOL>,<SPREAD>

    Notes:
    - Delimiter auto-detected.
    - Volume optional (defaults to 0.0).
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        sample = f.read(4096)
        f.seek(0)
        delim = _detect_delimiter(sample)

        reader = csv.DictReader(f, delimiter=delim)
        if reader.fieldnames is None:
            raise ValueError("CSV has no header row")

        headers = list(reader.fieldnames)

        # Detect MT5 split date/time columns
        col_date = None
        col_time = None
        for h in headers:
            nh = _norm(h)
            if nh == "date":
                col_date = h
            elif nh == "time":
                col_time = h

        use_split_dt = (col_date is not None and col_time is not None)

        # OHLCV columns
        col_t = _find_col(headers, "time")  # for non-split timestamp column
        col_o = _find_col(headers, "open")
        col_h = _find_col(headers, "high")
        col_l = _find_col(headers, "low")
        col_c = _find_col(headers, "close")
        col_v = _find_col(headers, "volume")  # optional

        # Required columns check
        missing = []
        if not use_split_dt and col_t is None:
            missing.append("time")
        if col_o is None:
            missing.append("open")
        if col_h is None:
            missing.append("high")
        if col_l is None:
            missing.append("low")
        if col_c is None:
            missing.append("close")

        if missing:
            raise ValueError(f"Missing required columns: {missing}. Found headers={headers}")

        out: List[Candle] = []
        for i, row in enumerate(reader):
            if limit is not None and i >= limit:
                break

            try:
                if use_split_dt:
                    ts = _parse_ts(f"{row[col_date]} {row[col_time]}")  # type: ignore[index]
                else:
                    ts = _parse_ts(row[col_t])  # type: ignore[index]

                o = float(row[col_o])  # type: ignore[index]
                h = float(row[col_h])  # type: ignore[index]
                l = float(row[col_l])  # type: ignore[index]
                c = float(row[col_c])  # type: ignore[index]

                v = 0.0
                if col_v is not None:
                    vv = (row.get(col_v, "") or "").strip()
                    if vv != "":
                        v = float(vv)
            except Exception as e:
                raise ValueError(f"Bad row at line={i+2}: {row}. Err={e}") from e

            # sanity OHLC
            if not (l <= o <= h and l <= c <= h):
                raise ValueError(f"Invalid OHLC range at line={i+2}: {row}")

            out.append(Candle(ts=ts, o=o, h=h, l=l, c=c, v=v))

    if sort_by_ts:
        out.sort(key=lambda x: x.ts)

    if drop_duplicates:
        dedup: List[Candle] = []
        last_ts: Optional[int] = None
        for c in out:
            if last_ts is not None and c.ts == last_ts:
                dedup[-1] = c  # keep later
            else:
                dedup.append(c)
            last_ts = c.ts
        out = dedup

    return [c.to_dict() for c in out]
