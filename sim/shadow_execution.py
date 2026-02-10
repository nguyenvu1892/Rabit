# sim/shadow_execution.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple


@dataclass
class SimTrade:
    intent_id: str
    side: str  # "buy" / "sell"
    entry_price: float
    sl: float
    tp: float
    entry_ts: int
    exit_price: Optional[float] = None
    exit_ts: Optional[int] = None
    result: Optional[str] = None  # "win" / "loss" / "flat"


def _hit_tp_sl(side: str, candle: Dict[str, Any], tp: float, sl: float) -> Optional[str]:
    h = float(candle["h"])
    l = float(candle["l"])

    if side == "buy":
        if l <= sl:
            return "sl"
        if h >= tp:
            return "tp"
    else:  # sell
        if h >= sl:
            return "sl"
        if l <= tp:
            return "tp"
    return None


def open_trade(intent_id: str, side: str, entry_price: float, sl: float, tp: float, ts: int) -> SimTrade:
    return SimTrade(
        intent_id=intent_id,
        side=side,
        entry_price=float(entry_price),
        sl=float(sl),
        tp=float(tp),
        entry_ts=int(ts),
    )


def step_trade(trade: SimTrade, candle: Dict[str, Any]) -> Tuple[bool, Optional[Dict[str, Any]]]:
    """
    Returns: (closed, outcome_dict_or_none)
    outcome_dict:
      { "intent_id", "result", "pnl", "entry", "exit", "entry_ts", "exit_ts" }
    """
    hit = _hit_tp_sl(trade.side, candle, trade.tp, trade.sl)
    if hit is None:
        return False, None

    exit_price = trade.tp if hit == "tp" else trade.sl
    trade.exit_price = float(exit_price)
    trade.exit_ts = int(candle["ts"])

    # simple pnl (1 unit volume)
    if trade.side == "buy":
        pnl = trade.exit_price - trade.entry_price
    else:
        pnl = trade.entry_price - trade.exit_price

    if pnl > 0:
        result = "win"
    elif pnl < 0:
        result = "loss"
    else:
        result = "flat"

    trade.result = result

    outcome = {
        "intent_id": trade.intent_id,
        "result": result,
        "pnl": float(pnl),
        "entry": float(trade.entry_price),
        "exit": float(trade.exit_price),
        "entry_ts": int(trade.entry_ts),
        "exit_ts": int(trade.exit_ts),
    }
    return True, outcome
