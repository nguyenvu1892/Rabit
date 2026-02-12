# tools/shadow_run.py
from __future__ import annotations

import argparse
import random
from typing import Any, Dict, Optional

from sim.candle_loader import load_candles_csv
from sim.shadow_runner import ShadowRunner

from brain.decision_engine import DecisionEngine
from brain.journal import Journal
from brain.trade_memory import TradeMemory
from brain.reinforcement_learner import ReinforcementLearner
from brain.weight_store import WeightStore

from observer.outcome_updater import OutcomeUpdater

# Optional: evaluation reporter (5.1.1)
try:
    from observer.eval_reporter import EvalReporter
except Exception:  # pragma: no cover
    EvalReporter = None  # type: ignore


def _import_risk_engine():
    # Project structure moved a few times; support both.
    try:
        from risk.risk_engine import RiskEngine  # type: ignore

        return RiskEngine
    except Exception:
        try:
            from brain.risk_engine import RiskEngine  # type: ignore

            return RiskEngine
        except Exception:
            from risk_engine import RiskEngine  # type: ignore

            return RiskEngine


def _stats_to_dict(stats: Any) -> Dict[str, Any]:
    if stats is None:
        return {}
    if isinstance(stats, dict):
        return stats
    if hasattr(stats, "to_dict") and callable(getattr(stats, "to_dict")):
        try:
            return dict(stats.to_dict())
        except Exception:
            pass

    d: Dict[str, Any] = {}
    for k in [
        "steps",
        "decisions",
        "allow",
        "deny",
        "errors",
        "outcomes",
        "wins",
        "losses",
        "total_pnl",
        "forced_entries",
    ]:
        if hasattr(stats, k):
            d[k] = getattr(stats, k)
    return d


def _weight_pair_count(ws: Optional[WeightStore]) -> int:
    if ws is None:
        return 0
    try:
        d = ws.to_dict()
        return sum(len(sub) for sub in d.values())
    except Exception:
        try:
            return len(ws)  # type: ignore
        except Exception:
            return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--lookback", type=int, default=300)
    ap.add_argument("--max-steps", type=int, default=2000)
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--horizon", type=int, default=30)

    # exploration (owned by ShadowRunner)
    ap.add_argument("--epsilon", type=float, default=0.0)
    ap.add_argument("--epsilon-cooldown", type=int, default=0)

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--journal", type=str, default=None, help="path to journal jsonl (append)")

    # weights (5.0.8.x)
    ap.add_argument("--weights", type=str, default=None, help="path to weights json (load+save)")

    # reporting (5.1.1)
    ap.add_argument("--report-dir", type=str, default="data/reports")

    args = ap.parse_args()

    random.seed(args.seed)

    candles = load_candles_csv(args.csv, limit=args.limit)
    journal = Journal(args.journal) if args.journal else None

    # Weight store
    weight_store: Optional[WeightStore] = None
    if args.weights:
        # init with path => auto-load if file exists
        weight_store = WeightStore(path=args.weights)

    # Reporter (optional)
    reporter = None
    if EvalReporter is not None and args.report_dir:
        try:
            reporter = EvalReporter(out_dir=args.report_dir)
            if weight_store is not None and hasattr(reporter, "snapshot_weights_before"):
                reporter.snapshot_weights_before(weight_store)
        except Exception:
            reporter = None

    trade_memory = TradeMemory() if args.train else None

    learner = None
    outcome_updater = None
    if args.train:
        learner = ReinforcementLearner(weight_store=weight_store)
        outcome_updater = OutcomeUpdater(
            learner=learner,
            trade_memory=trade_memory,
            weight_store=weight_store,
            weights_path=args.weights,
            autosave=True,
        )

    RiskEngine = _import_risk_engine()
    risk_engine = RiskEngine()

    de = DecisionEngine(
        risk_engine=risk_engine,
        weight_store=weight_store,
    )

    runner_kwargs = dict(
        weight_store=weight_store,
        outcome_updater=outcome_updater,
        reporter=reporter,
        train=args.train,
    )

    # Try newer kw name first, fallback to older ones
    try:
        runner = ShadowRunner(de, risk_engine=risk_engine, **runner_kwargs)
    except TypeError:
        try:
            runner = ShadowRunner(de, risk=risk_engine, **runner_kwargs)
        except TypeError:
            runner = ShadowRunner(de, risk_mgr=risk_engine, **runner_kwargs)

    stats = runner.run(
        candles=candles,
        lookback=args.lookback,
        max_steps=args.max_steps,
        horizon=args.horizon,
        train=args.train,
        epsilon=args.epsilon,
        epsilon_cooldown=args.epsilon_cooldown,
        journal=journal,
    )

    stats_dict = _stats_to_dict(stats)

    # Reporter finalize
    if reporter is not None:
        try:
            if weight_store is not None and hasattr(reporter, "snapshot_weights_after"):
                reporter.snapshot_weights_after(weight_store)

            if hasattr(reporter, "write") and callable(getattr(reporter, "write")):
                reporter.write(
                    stats=stats_dict,
                    extra={
                        "csv": args.csv,
                        "limit": args.limit,
                        "lookback": args.lookback,
                        "max_steps": args.max_steps,
                        "horizon": args.horizon,
                        "train": args.train,
                        "epsilon": args.epsilon,
                        "epsilon_cooldown": args.epsilon_cooldown,
                        "seed": args.seed,
                        "journal": args.journal,
                        "weights": args.weights,
                    },
                )
            elif hasattr(reporter, "finalize") and callable(getattr(reporter, "finalize")):
                reporter.finalize(stats=stats_dict)
        except Exception as e:
            print(f"[shadow_run] WARN: reporter finalize failed: {e}")

    print("=== SHADOW RUN DONE ===")
    print("WEIGHT SIZE:", _weight_pair_count(weight_store))
    for k, v in stats_dict.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()