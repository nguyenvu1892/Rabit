# brain/journal_logger.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from brain.journal import Journal, JournalEvent
from brain.event_schemas import (
    new_run_id,
    new_intent_id,
    make_snapshot_event,
    make_decision_event,
    make_order_plan_event,
    make_execution_event,
    make_outcome_event,
)
from brain.strategy_hash import genome_hash


@dataclass
class JournalContext:
    run_id: str
    strategy_hash: str


class JournalLogger:
    """
    Centralized logging wrapper to keep all journal events consistent.
    """

    def __init__(self, journal: Optional[Journal] = None, run_id: Optional[str] = None):
        self.journal = journal or Journal()
        self.run_id = run_id or new_run_id()

    def set_strategy(self, genome: Dict[str, Any]) -> str:
        sh = genome_hash(genome)
        self._strategy_hash = sh
        return sh

    def get_context(self) -> JournalContext:
        sh = getattr(self, "_strategy_hash", "unknown")
        return JournalContext(run_id=self.run_id, strategy_hash=sh)

    def new_intent_id(self) -> str:
        return new_intent_id()

    # ---- log methods ----

    def log_snapshot(self, symbol: str, snapshot_payload: Dict[str, Any]) -> JournalEvent:
        ctx = self.get_context()
        payload = make_snapshot_event(ctx.run_id, ctx.strategy_hash, symbol, snapshot_payload)
        return self.journal.append("snapshot", payload)

    def log_decision(self, intent_id: str, decision_payload: Dict[str, Any]) -> JournalEvent:
        ctx = self.get_context()
        payload = make_decision_event(ctx.run_id, intent_id, ctx.strategy_hash, decision_payload)
        return self.journal.append("decision", payload)

    def log_order_plan(self, intent_id: str, order_payload: Dict[str, Any]) -> JournalEvent:
        ctx = self.get_context()
        payload = make_order_plan_event(ctx.run_id, intent_id, ctx.strategy_hash, order_payload)
        return self.journal.append("order_plan", payload)

    def log_execution(self, intent_id: str, exec_payload: Dict[str, Any]) -> JournalEvent:
        ctx = self.get_context()
        payload = make_execution_event(ctx.run_id, intent_id, ctx.strategy_hash, exec_payload)
        return self.journal.append("execution", payload)

    def log_outcome(self, intent_id: str, outcome_payload: Dict[str, Any]) -> JournalEvent:
        ctx = self.get_context()
        payload = make_outcome_event(ctx.run_id, intent_id, ctx.strategy_hash, outcome_payload)
        return self.journal.append("outcome", payload)

    def log_upgrade(self, report_payload: Dict[str, Any]) -> JournalEvent:
        ctx = self.get_context()
        payload = {"run_id": ctx.run_id, "strategy_hash": ctx.strategy_hash, **report_payload}
        return self.journal.append("upgrade", payload)

    def log_rollback(self, report_payload: Dict[str, Any]) -> JournalEvent:
        ctx = self.get_context()
        payload = {"run_id": ctx.run_id, "strategy_hash": ctx.strategy_hash, **report_payload}
        return self.journal.append("rollback", payload)
        
    def log_heartbeat(self, payload: dict):
        self._write_event("heartbeat", payload)

    def log_risk_pause(self, payload: dict):
        ctx = self.get_context()
        data = {"run_id": ctx.run_id, "strategy_hash": ctx.strategy_hash, **payload}
        return self.journal.append("risk_pause", data)

    def log_risk_resume(self, payload: dict):
        ctx = self.get_context()
        data = {"run_id": ctx.run_id, "strategy_hash": ctx.strategy_hash, **payload}
        return self.journal.append("risk_resume", data)

    def log_session_reset(self, payload: dict):
        ctx = self.get_context()
        data = {"run_id": ctx.run_id, "strategy_hash": ctx.strategy_hash, **payload}
        return self.journal.append("session_reset", data)
