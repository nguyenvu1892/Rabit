# brain/event_schemas.py
from __future__ import annotations

import time
import uuid
from typing import Any, Dict, Optional


def new_run_id() -> str:
    return str(uuid.uuid4())


def new_intent_id() -> str:
    return str(uuid.uuid4())


def make_snapshot_event(run_id: str, strategy_hash: str, symbol: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "run_id": run_id,
        "strategy_hash": strategy_hash,
        "symbol": symbol,
        **payload,
    }


def make_decision_event(run_id: str, intent_id: str, strategy_hash: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "run_id": run_id,
        "intent_id": intent_id,
        "strategy_hash": strategy_hash,
        **payload,
    }


def make_order_plan_event(run_id: str, intent_id: str, strategy_hash: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "run_id": run_id,
        "intent_id": intent_id,
        "strategy_hash": strategy_hash,
        **payload,
    }


def make_execution_event(run_id: str, intent_id: str, strategy_hash: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "run_id": run_id,
        "intent_id": intent_id,
        "strategy_hash": strategy_hash,
        **payload,
    }


def make_outcome_event(run_id: str, intent_id: str, strategy_hash: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "run_id": run_id,
        "intent_id": intent_id,
        "strategy_hash": strategy_hash,
        **payload,
    }
