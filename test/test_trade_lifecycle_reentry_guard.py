# test/test_trade_lifecycle_reentry_guard.py
import os
import tempfile

from brain.journal import Journal
from brain.journal_logger import JournalLogger
from broker.mock_adapter import MockBrokerAdapter
from executor.order_builder import OrderBuilder
from executor.order_router import OrderRouter
from executor.reentry_guard import ReentryGuard
from observer.outcome_updater import OutcomeUpdater
from sim.replay_loop import ReplayLoop
from sim.trade_lifecycle import TradeLifecycleSim
from brain.feature.feature_set import FeatureSet

from brain.decision_engine import DecisionEngine
from brain.strategy_store import StrategyStore
from brain.strategy_policy import StrategyPolicy


class FakeRL:
    def get_weight(self, feature, value): return 1.0


class FakeTradeMemory:
    def __init__(self):
        self.memory = {("k",): {"wins": 0, "losses": 0, "total_pnl": 0.0, "samples": 0}}
    def get_stats(self, trade_features):
        e = self.memory[("k",)]
        s = e["samples"]
        return {"samples": s, "wins": e["wins"], "losses": e["losses"], "total_pnl": e["total_pnl"],
                "avg_pnl": (e["total_pnl"]/s) if s else 0.0, "win_rate": (e["wins"]/s) if s else 0.0,
                "drawdown": 0.0}
    def record(self, snapshot, outcome):
        e = self.memory[("k",)]
        e["samples"] += 1
        e["total_pnl"] += float(outcome.get("pnl", 0.0))
        if outcome.get("win", False): e["wins"] += 1
        else: e["losses"] += 1


class FakeMemoryIntel:
    def evaluate(self, memory_stats): return 0.7


class FakeContextIntel:
    def evaluate_context(self, trade_features): return {"strength": 1.0, "confidence": 0.7}


class FakeRiskEngine:
    def scale_risk(self, score, confidence): return 0.01
    def apply_policy(self, risk, policy_risk):
        out = dict(policy_risk); out["risk_per_trade"] = float(policy_risk.get("risk_per_trade", risk)); return out


class FakeLearner:
    def learn(self, snapshot, outcome): pass


def test_lifecycle_respects_reentry_guard():
    with tempfile.TemporaryDirectory() as d:
        journal_path = os.path.join(d, "journal.jsonl")
        best_path = os.path.join(d, "best_strategy.json")

        store = StrategyStore(best_path)
        store.save({"entry_threshold": 0.0, "sl_atr_mult": 1.5, "tp_atr_mult": 2.5, "risk_per_trade": 0.01,
                    "only_trend": False, "avoid_high_vol": False}, meta={"fitness": 1.0})

        j = Journal(journal_path)
        logger = JournalLogger(journal=j, run_id="run-guard")
        logger.set_strategy(store.load() or {})

        tm = FakeTradeMemory()
        de = DecisionEngine(
            risk_engine=FakeRiskEngine(),
            context_intel=FakeContextIntel(),
            trade_memory=tm,
            memory_intel=FakeMemoryIntel(),
            rl=FakeRL(),
            strategy_store=store,
            strategy_policy=StrategyPolicy(store.load() or {}),
            journal_logger=logger,
        )

        fs = FeatureSet(symbol="XAUUSD")
        loop = ReplayLoop(feature_set=fs, decision_engine=de, window=50)

        broker = MockBrokerAdapter()
        router = OrderRouter(broker=broker, journal_logger=logger)
        builder = OrderBuilder()

        updater = OutcomeUpdater(
            learner=FakeLearner(),
            trade_memory=tm,
            upgrade_scheduler=None,
            journal_logger=logger,
            decision_engine=de,
        )

        guard = ReentryGuard(cooldown_trades=1000)  # huge cooldown => allow only first order
        life = TradeLifecycleSim(loop, builder, router, updater, reentry_guard=guard)

        candles = []
        price = 100.0
        for _ in range(20):
            candles.append({"o": price, "h": price + 1, "l": price - 1, "c": price + 0.5, "v": 100})
            price += 0.2

        life.run(candles)

        assert j.count("order_plan") == 1
        assert j.count("execution") == 1
        assert j.count("outcome") == 1
