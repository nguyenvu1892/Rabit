# tools/shadow_run.py
from __future__ import annotations

import argparse
import random
from typing import Optional

from sim.candle_loader import load_candles_csv
from sim.shadow_runner import ShadowRunner

from brain.decision_engine import DecisionEngine
from brain.journal import Journal
from observer.outcome_updater import OutcomeUpdater
from brain.reinforcement_learner import ReinforcementLearner

from brain.weight_store import WeightStore
from brain.trade_memory import TradeMemory

# Optional reporting (5.1.1)
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--lookback", type=int, default=300)
    ap.add_argument("--max-steps", type=int, default=2000)
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--horizon", type=int, default=30)
    ap.add_argument("--epsilon", type=float, default=0.0)
    ap.add_argument("--epsilon-cooldown", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--journal", type=str, default=None, help="path to journal jsonl (append)")
    ap.add_argument("--weights", type=str, default=None, help="path to weights json (load/save)")
    ap.add_argument("--reports-dir", type=str, default="data/reports", help="directory for eval reports")

    args = ap.parse_args()
    random.seed(args.seed)

    candles = load_candles_csv(args.csv, limit=args.limit)
    journal = Journal(args.journal) if args.journal else None

    # --- weight store (load if provided) ---
    weight_store = WeightStore()
    if args.weights_weights := args.weights:
        try:
            weight_store.load_json(args.weights)
        except Exception as e:
            print(f"[shadow_run] WARN: cannot load weights from {args.weights}: {e}")

    # --- reporter (optional) ---
    reporter = None
    if EvalReporter is not None:
        reporter = EvalReporter(out_dir=args.reports_dir)
        try:
            reporter.snapshot_weights_before(weight_store)
        except Exception as e:
            print(f"[shadow_run] WARN: snapshot_weights_before failed: {e}")

    # --- learner/outcome updater ---
    trade_memory = TradeMemory()
    learner = ReinforcementLearner(weight_store=weight_store) if args.train else None
    outcome_updater = OutcomeUpdater(
        learner=learner,
        trade_memory=trade_memory,
        weight_store=weight_store,
        weights_path=args.weights,
        autosave=True,
    ) if args.train else None

    RiskEngine = _import_risk_engine()
    risk_engine = RiskEngine()

    de = DecisionEngine(
        risk_engine=risk_engine,
        weight_store=weight_store,
    )

    runner = ShadowRunner(
        de,
        risk_engine=risk_engine,
        outcome_updater=outcome_updater,
        seed=args.seed,
        train=args.train,
    )

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

    # --- reporter end-of-run ---
    if reporter is not None:
        try:
            reporter.snapshot_weights_after(weight_store)
            reporter.append(stats, extra={"weights_path": args.weights})
        except Exception as e:
            print(f"[shadow_run] WARN: reporter finalize failed: {e}")

    # Console summary
    print("=== SHADOW RUN DONE ===")
    for k, v in stats.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
