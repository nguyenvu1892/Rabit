# broker/mock_adapter.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from broker.contracts import OrderPlan, ExecutionReport


@dataclass
class MockBrokerAdapter:
    """
    Deterministic paper broker:
      - fills immediately at provided price
      - idempotent by intent_id (same intent returns same report)
    """
    slippage: float = 0.0

    def __post_init__(self):
        self._seen: Dict[str, ExecutionReport] = {}

    def place_order(self, plan: OrderPlan, price: float) -> ExecutionReport:
        if plan.intent_id in self._seen:
            return self._seen[plan.intent_id]

        # v1 always fill
        fill_price = float(price) + float(self.slippage)
        rep = ExecutionReport(
            intent_id=plan.intent_id,
            symbol=plan.symbol,
            side=plan.side,
            status="filled",
            fill_price=fill_price,
            message="mock_filled",
        )
        self._seen[plan.intent_id] = rep
        return rep
