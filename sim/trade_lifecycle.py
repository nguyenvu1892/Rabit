# sim/trade_lifecycle.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional
# add import
from executor.reentry_guard import ReentryGuard
from risk.session_guard import SessionRiskGuard
from risk.session_scheduler import SessionScheduler
from risk.session_scheduler_day import DaySessionScheduler

@dataclass
class TradeLifecycleResult:
    steps: int
    decisions: int
    orders: int
    executions: int
    outcomes: int


class TradeLifecycleSim:
    """
    Glue layer for SIM:
      candles -> features -> decision -> order -> execution -> outcome -> learning
    """

    def __init__(self, replay_loop, order_builder, order_router, outcome_updater, reentry_guard=None):
        self.replay_loop = replay_loop
        self.order_builder = order_builder
        self.order_router = order_router
        self.outcome_updater = outcome_updater
        self.reentry_guard = reentry_guard or ReentryGuard(cooldown_trades=5)
        self._trade_count = 0
        self._prev_fill_price: Optional[float] = None
        self.session_guard = session_guard or SessionRiskGuard()
        self._was_paused = False
        self.session_scheduler = session_scheduler or SessionScheduler(every_n_steps=1000)
        self.day_scheduler = day_scheduler or DaySessionScheduler()
        self.rl = getattr(self.replay_loop.decision_engine, "rl", None)
        self.trade_memory = getattr(self.outcome_updater, "trade_memory", None)
        self.session_guard = getattr(self, "session_guard", None)



    def step(self, candle: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        out = self.replay_loop.step(candle)
        self._trade_count += 1
        if self.session_scheduler.should_reset(self._trade_count):
            self.session_guard.reset_session()
            self._was_paused = False
            if getattr(self.order_router, "journal_logger", None) is not None:
                try:
                    self.order_router.journal_logger.log_session_reset({
                        "step": self._trade_count,
                        "symbol": str(feats.get("symbol", "XAUUSD")) if isinstance(feats, dict) else "UNKNOWN",
                        "reason": "every_n_steps",
                    })
                except Exception:
                    pass
        if self.day_scheduler is not None and self.day_scheduler.should_reset(candle):
            self.session_guard.reset_session()
            self._was_paused = False
            if getattr(self.order_router, "journal_logger", None) is not None:
                try:
                    self.order_router.journal_logger.log_session_reset({
                        "step": self._trade_count,
                        "symbol": str(feats.get("symbol", "XAUUSD")),
                        "reason": "new_day",
                        "day": self.day_scheduler.last_day,
                    })
                except Exception:
                    pass
               
        symbol = str(feats.get("symbol", "XAUUSD"))
        dec = self.reentry_guard.can_enter(symbol, self._trade_count)
        if not dec.allowed:
            return {"decision": out, "order": None, "execution": None, "outcome": None}

        if out is None:
            return None

        allow = bool(out["allow"])
        feats = out["features"]
        price = float(candle["c"])

        if not allow:
            return {"decision": out, "order": None, "execution": None, "outcome": None}
        self._trade_count += 1
        symbol = str(feats.get("symbol", "XAUUSD"))
        dec = self.reentry_guard.can_enter(symbol, self._trade_count)
        if not dec.allowed:
            return {"decision": out, "order": None, "execution": None, "outcome": None}

        gs = self.session_guard.can_trade(self._trade_count)

        # log resume edge
        if self._was_paused and gs.allowed:
            if getattr(self.order_router, "journal_logger", None) is not None:
                try:
                    self.order_router.journal_logger.log_risk_resume({
                        "step": self._trade_count,
                        "symbol": str(feats.get("symbol", "XAUUSD")),
                    })
                except Exception:
                    pass
            self._was_paused = False

        if not gs.allowed:
            # log pause (only on edge)
            if not self._was_paused:
                if getattr(self.order_router, "journal_logger", None) is not None:
                    try:
                        self.order_router.journal_logger.log_risk_pause({
                            "step": self._trade_count,
                            "symbol": str(feats.get("symbol", "XAUUSD")),
                            "reason": gs.reason,
                            "pause_remaining": gs.pause_remaining,
                        })
                    except Exception:
                        pass
            self._was_paused = True
            return {"decision": out, "order": None, "execution": None, "outcome": None}

        # Build order
        intent_id = None
        de = self.replay_loop.decision_engine
        if hasattr(de, "get_last_intent_id"):
            intent_id = de.get_last_intent_id()
        if intent_id is None:
            # fallback
            intent_id = "intent-unknown"

        risk_cfg = {}
        if hasattr(de, "get_last_risk_config"):
            risk_cfg = de.get_last_risk_config() or {}

        plan = self.order_builder.build(intent_id, feats, risk_cfg)
        if plan is None:
            return {"decision": out, "order": None, "execution": None, "outcome": None}

        rep = self.order_router.place(plan, price=price)

        # Create outcome (simple deterministic-ish):
        # pnl = (next_close - fill_price) * sign
        # since we don't have next candle here, we approximate using current close movement vs last fill
        fill_price = float(rep.fill_price or price)
        if self._prev_fill_price is None:
            move = 0.0
        else:
            move = price - self._prev_fill_price

        sign = 1.0 if plan.side == "buy" else -1.0
        pnl = move * sign
        win = pnl >= 0

        outcome = {
            "snapshot": {"features": feats},
            "pnl": float(pnl),
            "win": bool(win),
            "fill_price": fill_price,
            "side": plan.side,
            "symbol": plan.symbol,
        }

        self.outcome_updater.process_outcome(outcome)

        self._prev_fill_price = fill_price

        return {"decision": out, "order": plan, "execution": rep, "outcome": outcome}

    def run(self, candles):
        steps = 0
        decisions = 0
        orders = 0
        executions = 0
        outcomes = 0

        for c in candles:
            steps += 1
            r = self.step(c)
            if r is None:
                continue
            decisions += 1
            if r["order"] is not None:
                orders += 1
            if r["execution"] is not None:
                executions += 1
            if r["outcome"] is not None:
                outcomes += 1

        return TradeLifecycleResult(
            steps=steps,
            decisions=decisions,
            orders=orders,
            executions=executions,
            outcomes=outcomes,
        )

    def get_state(self) -> dict:
        return {
            "_prev_fill_price": self._prev_fill_price,
            "_trade_count": self._trade_count,
            "_was_paused": getattr(self, "_was_paused", False),
            "reentry_guard": self.reentry_guard.get_state() if hasattr(self, "reentry_guard") else {},
            "day_scheduler": self.day_scheduler.get_state() if getattr(self, "day_scheduler", None) is not None else {},
        }

    def set_state(self, state: dict) -> None:
        s = state or {}
        self._prev_fill_price = s.get("_prev_fill_price", None)
        self._trade_count = int(s.get("_trade_count", 0))
        self._was_paused = bool(s.get("_was_paused", False))

        if hasattr(self, "reentry_guard") and "reentry_guard" in s:
            try:
                self.reentry_guard.set_state(s.get("reentry_guard", {}) or {})
            except Exception:
                pass

        if getattr(self, "day_scheduler", None) is not None and "day_scheduler" in s:
            try:
                self.day_scheduler.set_state(s.get("day_scheduler", {}) or {})
            except Exception:
                pass
