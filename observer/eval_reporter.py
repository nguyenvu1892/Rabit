from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional, Tuple, List


def _now_ts() -> float:
    return time.time()


def _ts_label(ts: Optional[float] = None) -> str:
    ts = ts if ts is not None else _now_ts()
    return time.strftime("%Y%m%d_%H%M%S", time.localtime(ts))


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _flatten_weights(weights: Dict[str, Dict[str, float]]) -> List[Tuple[str, str, float]]:
    out: List[Tuple[str, str, float]] = []
    for bucket, sub in (weights or {}).items():
        if not isinstance(sub, dict):
            continue
        for key, val in sub.items():
            out.append((str(bucket), str(key), _safe_float(val, 1.0)))
    return out


def _top_bottom(flat: List[Tuple[str, str, float]], k: int = 5) -> Dict[str, Any]:
    if not flat:
        return {"top": [], "bottom": []}
    flat_sorted = sorted(flat, key=lambda x: x[2], reverse=True)
    top = flat_sorted[:k]
    bottom = list(reversed(flat_sorted[-k:]))
    return {
        "top": [{"bucket": b, "key": kk, "w": v} for (b, kk, v) in top],
        "bottom": [{"bucket": b, "key": kk, "w": v} for (b, kk, v) in bottom],
    }


def _drift_score(prev: Dict[str, Dict[str, float]], curr: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    """
    Drift score: L1 mean abs diff over overlapping keys + new-key ratio.
    Lightweight & stable.
    """
    prev_flat = {(b, k): _safe_float(v, 1.0) for b, sub in (prev or {}).items() for k, v in (sub or {}).items()}
    curr_flat = {(b, k): _safe_float(v, 1.0) for b, sub in (curr or {}).items() for k, v in (sub or {}).items()}

    overlap = set(prev_flat.keys()) & set(curr_flat.keys())
    if overlap:
        l1 = sum(abs(curr_flat[x] - prev_flat[x]) for x in overlap)
        mean_l1 = l1 / max(1, len(overlap))
    else:
        mean_l1 = 0.0

    new_keys = len(set(curr_flat.keys()) - set(prev_flat.keys()))
    new_ratio = new_keys / max(1, len(curr_flat))

    return {"mean_abs_delta": float(mean_l1), "new_key_ratio": float(new_ratio)}


@dataclass
class EvalRunMeta:
    run_id: str
    created_ts: float
    source: str = "shadow_run"


class EvalReporter:
    """
    5.1.1: Report schema + writer.

    You can:
      - call snapshot_weights(weight_store) at start/end to compute drift
      - call record_summary(summary_dict) after shadow_run completes
      - write_report(...) to JSON file
    """

    def __init__(self, out_dir: str = "data/reports", run_id: Optional[str] = None) -> None:
        self.out_dir = out_dir
        _ensure_dir(self.out_dir)

        ts = _now_ts()
        self.meta = EvalRunMeta(
            run_id=run_id or f"run_{_ts_label(ts)}",
            created_ts=ts,
        )

        self._weights_before: Dict[str, Dict[str, float]] = {}
        self._weights_after: Dict[str, Dict[str, float]] = {}
        self._summary: Dict[str, Any] = {}

    def snapshot_weights_before(self, weight_store: Any) -> None:
        self._weights_before = self._read_weights(weight_store)

    def snapshot_weights_after(self, weight_store: Any) -> None:
        self._weights_after = self._read_weights(weight_store)

    def record_summary(self, summary: Dict[str, Any]) -> None:
        self._summary = dict(summary or {})

    def build_report(self) -> Dict[str, Any]:
        flat_after = _flatten_weights(self._weights_after)
        report: Dict[str, Any] = {
            "meta": asdict(self.meta),
            "summary": self._summary,
            "weights": {
                "after_top_bottom": _top_bottom(flat_after, k=5),
                "drift": _drift_score(self._weights_before, self._weights_after),
                "counts": {
                    "buckets": len(self._weights_after or {}),
                    "keys_total": len(flat_after),
                },
            },
        }
        return report

    def write_report(self, filename: Optional[str] = None) -> str:
        report = self.build_report()
        fname = filename or f"{self.meta.run_id}.json"
        path = os.path.join(self.out_dir, fname)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False, sort_keys=True)
        return path

    # -----------------------
    # internals
    # -----------------------
    def _read_weights(self, weight_store: Any) -> Dict[str, Dict[str, float]]:
        # Support multiple WeightStore implementations
        if weight_store is None:
            return {}
        # common: internal dict
        for attr in ("_w", "weights", "store", "data"):
            if hasattr(weight_store, attr):
                try:
                    w = getattr(weight_store, attr)
                    if isinstance(w, dict):
                        # nested dict expected
                        return {str(b): {str(k): _safe_float(v, 1.0) for k, v in (sub or {}).items()}
                                for b, sub in w.items()
                                if isinstance(sub, dict)}
                except Exception:
                    pass
        # fallback: try export method
        for fn in ("to_dict", "dump", "export"):
            if hasattr(weight_store, fn):
                try:
                    w = getattr(weight_store, fn)()
                    if isinstance(w, dict):
                        return {str(b): {str(k): _safe_float(v, 1.0) for k, v in (sub or {}).items()}
                                for b, sub in w.items()
                                if isinstance(sub, dict)}
                except Exception:
                    pass
        return {}
