# broker/contracts.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Literal


Side = Literal["buy", "sell"]
OrderType = Literal["market"]
OrderStatus = Literal["filled", "rejected"]


@dataclass(frozen=True)
class OrderPlan:
    intent_id: str
    symbol: str
    side: Side
    order_type: OrderType = "market"

    volume: float = 0.01
    sl: Optional[float] = None
    tp: Optional[float] = None

    max_slippage: float = 0.0  # v1


@dataclass(frozen=True)
class ExecutionReport:
    intent_id: str
    symbol: str
    side: Side
    status: OrderStatus
    fill_price: Optional[float] = None
    message: str = ""
