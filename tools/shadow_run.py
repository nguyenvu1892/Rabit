# tools/shadow_run.py
from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
from typing import Any, Dict, Optional

from brain.decision_engine import DecisionEngine
from brain.weight_store import WeightStore
from observer.eval_reporter import EvalReporter
from observer.outcome_updater import OutcomeUpdater
from sim.shadow_runner import ShadowRunner


def _safe_int(x, default: int) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _call_with_signature(fn, /, *args, **kwargs):
    """
    Call fn with kwargs filtered by its signature to avoid:
    - unexpected keyword argument
    - multiple values for argument
    """
    try:
        sig = inspect.signature(fn)
        params = sig.parameters
        accepts_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())

        if accepts_var_kw:
            return fn(*args, **kwargs)

        filtered = {}
        for k, v in kwargs.items():
            if k in params:
                filtered[k] = v
        return fn(*args, **filtered)
    except Exception:
        # last resort: raw call
        return fn(*args, **kwargs)


def load_csv_candles(csv_path: str, limit: int) -> list:
    # existing project already has loaders in other places,
    # but keep minimal fallback to not break.
    import csv

    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for i, row in enumerate(r):
            rows.append(row)
            if limit > 0 and i + 1 >= limit:
                break
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--limit", type=int, default=5000)
    ap.add_argument("--lookback", type=int, default=300)
    ap.add_argument("--max-steps", type=int, default=2000)
    ap.add_argument("--horizon", type=int, default=30)
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--epsilon", type=float, default=0.0)
    ap.add_argument("--epsilon-cooldown", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--journal", default=None)
    ap.add_argument("--weights", default=None)
    args = ap.parse_args()

    # weight store
    weight_store = WeightStore()
    if args.weights:
        try:
            weight_store.load(args.weights)
        except Exception:
            pass

    # decision engine
    de = DecisionEngine(
        risk_engine=None,
        weight_store=weight_store,
    )

    # reporter
    reporter = EvalReporter(out_dir="data/reports")

    # outcome updater (compat: accept learner/trade_memory kwargs if older versions call it)
    outcome_updater = OutcomeUpdater(
        weight_store=weight_store,
        autosave=True,
        weights_path=args.weights or "data/weights.json",
        journal_path=args.journal or "data/journal_train.jsonl",
    )

    # runner (compat: accept risk_engine kw)
    runner = ShadowRunner(
        decision_engine=de,
        risk_engine=None,
        outcome_updater=outcome_updater,
        seed=args.seed,
        train=args.train,
    )

    candles = load_csv_candles(args.csv, args.limit)

    run_kwargs: Dict[str, Any] = {
        "lookback": _safe_int(args.lookback, 300),
        "max_steps": _safe_int(args.max_steps, 2000),
        "horizon": _safe_int(args.horizon, 30),
        "train": bool(args.train),
        "epsilon": float(args.epsilon),
        "epsilon_cooldown": _safe_int(args.epsilon_cooldown, 0),
        "journal": None,  # keep journal None unless you have journal object
    }

    stats = _call_with_signature(runner.run, candles, **run_kwargs)

    # snapshot weights
    try:
        reporter.snapshot_weights_before(weight_store)
    except Exception:
        pass

    # finalize report
    try:
        reporter.finalize(stats.to_dict() if hasattr(stats, "to_dict") else dict(stats.__dict__))
    except Exception:
        pass

    print("=== SHADOW RUN DONE ===")
    # printing robust
    payload = stats.to_dict() if hasattr(stats, "to_dict") else dict(stats.__dict__)
    for k in ["steps", "decisions", "allow", "deny", "errors", "outcomes", "wins", "losses", "total_pnl", "forced_entries"]:
        if k in payload:
            print(f"{k}: {payload[k]}")
    if "regime_breakdown" in payload:
        print("regime_breakdown:", payload["regime_breakdown"])

    try:
        reporter.snapshot_weights_after(weight_store)
    except Exception:
        pass


if __name__ == "__main__":
    main()
