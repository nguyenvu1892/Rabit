# observer/eval_reporter.py
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class EvalRecord:
    ts: float
    run_id: str
    stats: Dict[str, Any]
    extra: Dict[str, Any]


class EvalReporter:
    """
    Minimal evaluation reporter.
    - Can print a compact summary
    - Can append to a JSONL file (one record per run)
    - Optional: snapshot weights before/after a run
    """

    def __init__(
        self,
        out_path: Optional[str] = None,
        run_id: Optional[str] = None,
        out_dir: Optional[str] = None,
        filename: str = "eval_report.jsonl",
    ) -> None:
        self.run_id = run_id or f"run_{int(time.time())}"
        self.out_path: Optional[str] = None

        # Resolve output path
        if out_path:
            self.out_path = str(out_path)
        elif out_dir:
            self.out_path = str(Path(out_dir) / filename)
        else:
            self.out_path = None

        if self.out_path:
            Path(self.out_path).parent.mkdir(parents=True, exist_ok=True)

        self._weights_before: Optional[Dict[str, Any]] = None
        self._weights_after: Optional[Dict[str, Any]] = None

    # -------------------------
    # Weight snapshots (used by shadow_run)
    # -------------------------
    def snapshot_weights_before(self, weight_store: Any) -> None:
        self._weights_before = self._snapshot_weights(weight_store)

    def snapshot_weights_after(self, weight_store: Any) -> None:
        self._weights_after = self._snapshot_weights(weight_store)

    def _snapshot_weights(self, weight_store: Any) -> Dict[str, Any]:
        if weight_store is None:
            return {}

        # Duck-typing to avoid tight coupling
        for name in ("to_dict", "as_dict", "dump", "export", "snapshot"):
            fn = getattr(weight_store, name, None)
            if callable(fn):
                try:
                    data = fn()
                    if isinstance(data, dict):
                        return data
                except Exception:
                    pass

        # fallback: try known common attribute
        w = getattr(weight_store, "weights", None)
        if isinstance(w, dict):
            return w

        return {}

    # -------------------------
    # Reporting
    # -------------------------
    def summarize(self, stats: Dict[str, Any]) -> Dict[str, Any]:
        # Keep compatible with ShadowRunner stats keys
        decisions = int(stats.get("decisions", 0) or 0)
        allow = int(stats.get("allow", 0) or 0)
        deny = int(stats.get("deny", 0) or 0)
        errors = int(stats.get("errors", 0) or 0)

        wins = int(stats.get("wins", 0) or 0)
        losses = int(stats.get("losses", 0) or 0)
        total_pnl = float(stats.get("total_pnl", 0.0) or 0.0)

        return {
            "decisions": decisions,
            "allow": allow,
            "deny": deny,
            "errors": errors,
            "wins": wins,
            "losses": losses,
            "total_pnl": total_pnl,
        }

    def write(self, stats: Dict[str, Any], extra: Optional[Dict[str, Any]] = None) -> Optional[EvalRecord]:
        extra = extra or {}
        record = EvalRecord(
            ts=time.time(),
            run_id=self.run_id,
            stats=self.summarize(stats),
            extra=extra,
        )

        payload: Dict[str, Any] = {
            "ts": record.ts,
            "run_id": record.run_id,
            "stats": record.stats,
            "extra": record.extra,
        }

        if self._weights_before is not None:
            payload["weights_before"] = self._weights_before
        if self._weights_after is not None:
            payload["weights_after"] = self._weights_after

        if self.out_path:
            with open(self.out_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")

        return record

    def print_summary(self, stats: Dict[str, Any]) -> None:
        s = self.summarize(stats)
        print(
            f"[EvalReporter] decisions={s['decisions']} allow={s['allow']} deny={s['deny']} "
            f"errors={s['errors']} wins={s['wins']} losses={s['losses']} pnl={s['total_pnl']:.4f}"
        )
