# sim/paper_trading_loop.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from sim.loop_state import LoopState, LoopStateStore
from sim.metrics import Metrics
from persistence.state_manager import CoreStateManager


@dataclass
class LoopReport:
    steps: int
    decisions: int
    orders: int
    executions: int
    outcomes: int
    stopped: bool


class PaperTradingLoop:
    def __init__(
        self,
        data_source,
        lifecycle_sim,
        state_store: Optional[LoopStateStore] = None,
        run_id: str = "",
        strategy_hash: str = "",
        checkpoint_every: int = 10,
    ):
        self.data_source = data_source
        self.lifecycle_sim = lifecycle_sim
        self.state_store = state_store
        self.run_id = run_id
        self.strategy_hash = strategy_hash
        self.checkpoint_every = int(checkpoint_every)
        self.journal_logger = journal_logger
        self.heartbeat_every = int(heartbeat_every)
        self.metrics = Metrics()
        self.state_manager = state_manager
        self.state_every = int(state_every)
        self.state_manager.load_into(rl, trade_memory, session_guard, lifecycle=self.lifecycle_sim)
        self.state_manager.save(rl, trade_memory, session_guard, lifecycle=self.lifecycle_sim)
        self._stop = False

        self.steps = 0
        self.decisions = 0
        self.orders = 0
        self.executions = 0
        self.outcomes = 0

    def stop(self):
        self._stop = True

    def _save_state(self):
        if self.state_store is None:
            return
        st = LoopState(idx=self.data_source.pos(), run_id=self.run_id, strategy_hash=self.strategy_hash)
        self.state_store.save(st)

    def _maybe_restore_state(self):
        if self.state_store is None:
            return
        st = self.state_store.load()
        if st is None:
            return
        # resume data source index
        self.data_source.seek(st.idx)

    def run(self, max_steps: Optional[int] = None) -> LoopReport:
        self._maybe_restore_state()
        # Auto-resume core state (RL/trade_memory/session_guard) if manager provided
        if self.state_manager is not None:
            try:
                # lifecycle_sim should expose references
                rl = getattr(self.lifecycle_sim, "rl", None)
                trade_memory = getattr(self.lifecycle_sim, "trade_memory", None)
                session_guard = getattr(self.lifecycle_sim, "session_guard", None)
                if rl is not None and trade_memory is not None and session_guard is not None:
                    self.state_manager.load_into(rl, trade_memory, session_guard)
            except Exception:
                pass
        if self.state_manager is not None and self.state_every > 0:
            if self.steps % self.state_every == 0:
                try:
                    rl = getattr(self.lifecycle_sim, "rl", None)
                    trade_memory = getattr(self.lifecycle_sim, "trade_memory", None)
                    session_guard = getattr(self.lifecycle_sim, "session_guard", None)
                    if rl is not None and trade_memory is not None and session_guard is not None:
                        self.state_manager.save(rl, trade_memory, session_guard)
                except Exception:
                    pass
        if self.state_manager is not None:
            try:
                rl = getattr(self.lifecycle_sim, "rl", None)
                trade_memory = getattr(self.lifecycle_sim, "trade_memory", None)
                session_guard = getattr(self.lifecycle_sim, "session_guard", None)
                if rl is not None and trade_memory is not None and session_guard is not None:
                    self.state_manager.save(rl, trade_memory, session_guard)
            except Exception:
                pass
            
        self.metrics.on_step()

        while not self._stop:
            if max_steps is not None and self.steps >= int(max_steps):
                break

            candle = self.data_source.next()
            if candle is None:
                break

            self.steps += 1
            out = self.lifecycle_sim.step(candle)
            if out is None:
                if self.checkpoint_every > 0 and (self.steps % self.checkpoint_every == 0):
                    self._save_state()
                continue

            self.metrics.on_decision()
            if out.get("order") is not None:
                self.metrics.on_order()
            if out.get("execution") is not None:
                self.metrics.on_execution()
            if out.get("outcome") is not None:
                oc = out["outcome"]
                self.metrics.on_outcome(pnl=float(oc.get("pnl", 0.0)), win=bool(oc.get("win", False)))
            if self.checkpoint_every > 0 and (self.steps % self.checkpoint_every == 0):
                self._save_state()
            if self.journal_logger is not None and self.heartbeat_every > 0:
                if self.metrics.steps % self.heartbeat_every == 0:
                    try:
                        self.journal_logger.log_heartbeat({
                            "run_id": self.run_id,
                            "strategy_hash": self.strategy_hash,
                            "idx": self.data_source.pos(),
                            "metrics": self.metrics.to_dict(),
                        })
                    except Exception:
                         pass
        # final save on exit
        self._save_state()

        return LoopReport(
            steps=self.steps,
            decisions=self.decisions,
            orders=self.orders,
            executions=self.executions,
            outcomes=self.outcomes,
            stopped=bool(self._stop),
        )
