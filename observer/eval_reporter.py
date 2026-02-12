# observer/eval_reporter.py
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _get_meta(decision: Any) -> Dict[str, Any]:
    if decision is None:
        return {}
    if isinstance(decision, dict):
        return decision.get("meta") or {}
    return getattr(decision, "meta", None) or {}


def _get_expert(decision: Any) -> str:
    if decision is None:
        return "NONE"
    if isinstance(decision, dict):
        return str(decision.get("expert") or "NONE")
    return str(getattr(decision, "expert", "NONE"))


def _get_allow(decision: Any) -> bool:
    if decision is None:
        return False
    if isinstance(decision, dict):
        return bool(decision.get("allow", False))
    return bool(getattr(decision, "allow", False))


def _get_score(decision: Any) -> float:
    if decision is None:
        return 0.0
    if isinstance(decision, dict):
        return _safe_float(decision.get("score", 0.0), 0.0)
    return _safe_float(getattr(decision, "score", 0.0), 0.0)

@dataclass
class EvalRecord:
    ts: float
    run_id: str
    stats: Dict[str, Any]
    extra: Dict[str, Any]


class EvalReporter:
    """
    Minimal evaluation reporter (backward compatible)
    - Print compact summary
    - Append JSONL (one record per run)
    - Optional snapshot weights before/after
    - Optional: persist regime_breakdown if provided by stats/extra
    """
    report_dir: Path = Path("reports")
    filename: str = "eval_report.jsonl"

    allow: int = 0
    deny: int = 0
    outcomes: int = 0
    wins: int = 0
    losses: int = 0
    total_pnl: float = 0.0
    def __init__(
        self,
        out_path: Optional[str] = None,
        run_id: Optional[str] = None,
        out_dir: Optional[str] = None,
        filename: str = "eval_report.jsonl",
    ) -> None:
        self.run_id = run_id or f"run_{int(time.time())}"

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
    # Weight snapshots
    # -------------------------
    def snapshot_weights_before(self, weight_store: Any) -> None:
        self._weights_before = self._snapshot_weights(weight_store)

    def snapshot_weights_after(self, weight_store: Any) -> None:
        self._weights_after = self._snapshot_weights(weight_store)

    def _snapshot_weights(self, weight_store: Any) -> Dict[str, Any]:
        if weight_store is None:
            return {}
        # Duck-typing
        for name in ("to_dict", "as_dict", "dump", "export", "snapshot"):
            fn = getattr(weight_store, name, None)
            if callable(fn):
                try:
                    data = fn()
                    if isinstance(data, dict):
                        return data
                except Exception:
                    pass
        w = getattr(weight_store, "weights", None)
        return w if isinstance(w, dict) else {}

    # -------------------------
    # Regime breakdown helpers
    # -------------------------
    def _get_regime_breakdown(self, stats: Dict[str, Any], extra: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        # prefer stats, then extra
        rb = stats.get("regime_breakdown")
        if isinstance(rb, dict):
            return rb
        rb = extra.get("regime_breakdown")
        if isinstance(rb, dict):
            return rb
        # allow alt key names (future-proof)
        for k in ("by_regime", "regime_stats", "regime_report"):
            rb = stats.get(k)
            if isinstance(rb, dict):
                return rb
            rb = extra.get(k)
            if isinstance(rb, dict):
                return rb
        return None

    def _top_regime_by_pnl(self, regime_breakdown: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        # Expect: { regime: {pnl: float, ...}, ... }
        best = None
        best_pnl = None
        for regime, row in regime_breakdown.items():
            if not isinstance(row, dict):
                continue
            pnl = row.get("pnl")
            try:
                pnl_f = float(pnl)
            except Exception:
                continue
            if best_pnl is None or pnl_f > best_pnl:
                best_pnl = pnl_f
                best = {"regime": regime, "pnl": pnl_f}
        return best

    # -------------------------
    # Reporting
    # -------------------------
    def summarize(self, stats: Dict[str, Any], extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        extra = extra or {}

        decisions = int(stats.get("decisions", 0) or 0)
        allow = int(stats.get("allow", 0) or 0)
        deny = int(stats.get("deny", 0) or 0)
        errors = int(stats.get("errors", 0) or 0)
        wins = int(stats.get("wins", 0) or 0)
        losses = int(stats.get("losses", 0) or 0)
        total_pnl = float(stats.get("total_pnl", 0.0) or 0.0)

        out: Dict[str, Any] = {
            "decisions": decisions,
            "allow": allow,
            "deny": deny,
            "errors": errors,
            "wins": wins,
            "losses": losses,
            "total_pnl": total_pnl,
        }

        # Attach light regime hint into summary (optional)
        rb = self._get_regime_breakdown(stats, extra)
        if isinstance(rb, dict) and rb:
            top = self._top_regime_by_pnl(rb)
            if top:
                out["top_regime"] = top

        return out

    def write(self, stats: Dict[str, Any], extra: Optional[Dict[str, Any]] = None) -> Optional[EvalRecord]:
        extra = extra or {}

        record = EvalRecord(
            ts=time.time(),
            run_id=self.run_id,
            stats=self.summarize(stats, extra),
            extra=extra,
        )

        payload: Dict[str, Any] = {
            "ts": record.ts,
            "run_id": record.run_id,
            "stats": record.stats,
            "extra": record.extra,
        }

        # Keep old behavior
        if self._weights_before is not None:
            payload["weights_before"] = self._weights_before
        if self._weights_after is not None:
            payload["weights_after"] = self._weights_after

        # NEW: persist regime breakdown if present
        rb = self._get_regime_breakdown(stats, extra)
        if isinstance(rb, dict) and rb:
            payload["regime_breakdown"] = rb

        if self.out_path:
            with open(self.out_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")

        return record

    def print_summary(self, stats: Dict[str, Any], extra: Optional[Dict[str, Any]] = None) -> None:
        s = self.summarize(stats, extra or {})
        msg = (
            f"[EvalReporter] decisions={s['decisions']} allow={s['allow']} deny={s['deny']} "
            f"errors={s['errors']} wins={s['wins']} losses={s['losses']} pnl={s['total_pnl']:.4f}"
        )
        if "top_regime" in s:
            tr = s["top_regime"]
            msg += f" | top_regime={tr.get('regime')} pnl={float(tr.get('pnl', 0.0)):.4f}"
        print(msg)

    def __post_init__(self) -> None:
        self.report_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.report_dir / self.filename

    def record_step(
        self,
        step: int,
        context: Dict[str, Any],
        decision: Any,
        pnl: Optional[float] = None,
    ) -> None:
        regime = str(context.get("regime", "unknown"))
        bucket = context.get("bucket")

        expert = _get_expert(decision)
        allow = _get_allow(decision)
        score = _get_score(decision)
        meta = _get_meta(decision)

        if allow:
            self.allow += 1
        else:
            self.deny += 1

        rec: Dict[str, Any] = {
            "step": step,
            "regime": regime,
            "bucket": bucket,
            "best_expert": expert,
            "best_score": score,
            "allow": allow,
            "topk": meta.get("topk", []),
        }

        # outcome fields (available at close/update time)
        if pnl is not None:
            self.outcomes += 1
            pnl_f = _safe_float(pnl, 0.0)
            self.total_pnl += pnl_f

            if pnl_f >= 0:
                self.wins += 1
            else:
                self.losses += 1

            self.regime_pnl[regime] = float(self.regime_pnl.get(regime, 0.0)) + pnl_f

            rec["pnl"] = pnl_f
            rec["confidence"] = meta.get("confidence")
            rec["scaled_reward"] = meta.get("scaled_reward")

        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def summary(self) -> Dict[str, Any]:
        return {
            "allow": self.allow,
            "deny": self.deny,
            "outcomes": self.outcomes,
            "wins": self.wins,
            "losses": self.losses,
            "total_pnl": self.total_pnl,
            "regime_breakdown": dict(self.regime_pnl),
            "report_path": str(self.path),
        }