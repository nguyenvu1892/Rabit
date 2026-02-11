from __future__ import annotations

import argparse
import random

from sim.candle_loader import load_candles_csv
from sim.shadow_runner import ShadowRunner

from brain.decision_engine import DecisionEngine
from brain.journal import Journal
from brain.reinforcement_learner import ReinforcementLearner
from brain.weight_store import WeightStore
from observer.outcome_updater import OutcomeUpdater
from brain.trade_memory import TradeMemory
from observer.eval_reporter import EvalReporter


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


def main():
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

    # 5.0.8.1: weight learning persistence
    ap.add_argument("--weights", type=str, default="data/weights.json", help="path to weights json")

    args = ap.parse_args()

    random.seed(args.seed)

    candles = load_candles_csv(args.csv, limit=args.limit)
    journal = Journal(args.journal) if args.journal else None

    # Learner + weight store only matter in train mode
    learner = ReinforcementLearner() if args.train else None
    # tạo weight_store nếu có --weights
    weight_store = WeightStore(path=args.weights) if args.weights else None
    if weight_store:
        weight_store.load_json(args.weights)
    trade_memory = TradeMemory()
    outcome_updater = (
        OutcomeUpdater(
            learner=learner,
            trade_memory=trade_memory,
            weight_store=weight_store,
            weights_path=args.weights,
            autosave=True,
        )
        if args.train
        else None
    )

    RiskEngineCls = _import_risk_engine()
    risk_engine = RiskEngineCls()

    # IMPORTANT: epsilon belongs to ShadowRunner exploration (not DecisionEngine constructor).
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

    print("=== SHADOW RUN DONE ===")
    for k, v in stats.to_dict().items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
