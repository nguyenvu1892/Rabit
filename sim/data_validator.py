# sim/data_validator.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class DataReport:
    n: int
    ts_start: Optional[int]
    ts_end: Optional[int]

    inferred_tf_sec: Optional[int]
    duplicates: int
    out_of_order: int
    gap_count: int
    max_gap_sec: int

    bad_ohlc: int
    missing_volume: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n": self.n,
            "ts_start": self.ts_start,
            "ts_end": self.ts_end,
            "inferred_tf_sec": self.inferred_tf_sec,
            "duplicates": self.duplicates,
            "out_of_order": self.out_of_order,
            "gap_count": self.gap_count,
            "max_gap_sec": self.max_gap_sec,
            "bad_ohlc": self.bad_ohlc,
            "missing_volume": self.missing_volume,
        }


def infer_timeframe_seconds(candles: List[Dict[str, Any]], max_scan: int = 5000) -> Optional[int]:
    if len(candles) < 3:
        return None
    deltas = {}
    prev = None
    for c in candles[:max_scan]:
        ts = int(c["ts"])
        if prev is not None:
            d = ts - prev
            if d > 0:
                deltas[d] = deltas.get(d, 0) + 1
        prev = ts
    if not deltas:
        return None
    # mode delta
    return max(deltas.items(), key=lambda kv: kv[1])[0]


def validate_candles(candles: List[Dict[str, Any]]) -> DataReport:
    n = len(candles)
    if n == 0:
        return DataReport(
            n=0,
            ts_start=None,
            ts_end=None,
            inferred_tf_sec=None,
            duplicates=0,
            out_of_order=0,
            gap_count=0,
            max_gap_sec=0,
            bad_ohlc=0,
            missing_volume=0,
        )

    inferred_tf = infer_timeframe_seconds(candles)

    duplicates = 0
    out_of_order = 0
    gap_count = 0
    max_gap_sec = 0
    bad_ohlc = 0
    missing_volume = 0

    prev_ts: Optional[int] = None
    seen = set()

    for c in candles:
        ts = int(c["ts"])
        o = float(c["o"]); h = float(c["h"]); l = float(c["l"]); cl = float(c["c"])
        v = c.get("v", None)

        if v is None:
            missing_volume += 1

        if ts in seen:
            duplicates += 1
        else:
            seen.add(ts)

        if prev_ts is not None:
            if ts < prev_ts:
                out_of_order += 1
            else:
                d = ts - prev_ts
                if inferred_tf is not None and d > inferred_tf:
                    gap_count += 1
                    if d > max_gap_sec:
                        max_gap_sec = d
        prev_ts = ts

        if not (l <= o <= h and l <= cl <= h):
            bad_ohlc += 1

    return DataReport(
        n=n,
        ts_start=int(candles[0]["ts"]),
        ts_end=int(candles[-1]["ts"]),
        inferred_tf_sec=inferred_tf,
        duplicates=duplicates,
        out_of_order=out_of_order,
        gap_count=gap_count,
        max_gap_sec=max_gap_sec,
        bad_ohlc=bad_ohlc,
        missing_volume=missing_volume,
    )
