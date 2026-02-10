# test/test_shadow_trade_lifecycle.py
import random

from sim.shadow_runner import ShadowRunner


class AllowAllDecisionEngine:
    def evaluate_trade(self, trade_features):
        # allow always
        return True, 1.0, {"sl_atr_mult": 1.0, "tp_atr_mult": 1.0}


class DummyJournal:
    def log_decision(self, *a, **k): ...
    def log_outcome(self, *a, **k): ...
    def log_heartbeat(self, *a, **k): ...


def test_shadow_runner_produces_outcomes():
    # synthetic candles with movement
    candles = []
    ts = 0
    price = 100.0
    for i in range(200):
        o = price
        h = price + 1.0
        l = price - 1.0
        c = price + (0.2 if i % 2 == 0 else -0.2)
        v = 1.0
        candles.append({"ts": ts, "o": o, "h": h, "l": l, "c": c, "v": v})
        ts += 60
        price = c

    r = ShadowRunner(
        decision_engine=AllowAllDecisionEngine(),
        lookback=10,
        journal=DummyJournal(),
        heartbeat_every=50,
    )
    stats = r.run(candles, max_steps=100)

    assert stats.steps == 100
    assert stats.decisions > 0
    assert stats.outcomes > 0
    assert (stats.wins + stats.losses) >= 1
