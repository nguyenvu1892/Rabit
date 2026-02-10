# executor/order_router.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Any

from broker.contracts import OrderPlan, ExecutionReport
from brain.journal_logger import JournalLogger


@dataclass
class OrderRouter:
    broker: Any
    journal_logger: Optional[JournalLogger] = None  # optional

    def place(self, plan: OrderPlan, price: float) -> ExecutionReport:
        if self.journal_logger is not None:
            try:
                self.journal_logger.log_order_plan(
                    plan.intent_id,
                    {
                        "symbol": plan.symbol,
                        "side": plan.side,
                        "volume": plan.volume,
                        "sl": plan.sl,
                        "tp": plan.tp,
                        "order_type": plan.order_type,
                        "max_slippage": plan.max_slippage,
                    },
                )
            except Exception:
                pass

        # ✅ gọi đúng method của MockBrokerAdapter
        rep = self.broker.place_order(plan, price=price)

        if self.journal_logger is not None:
            try:
                self.journal_logger.log_execution(
                    plan.intent_id,
                    {
                        "symbol": rep.symbol,
                        "side": rep.side,
                        "status": rep.status,
                        "fill_price": rep.fill_price,
                        "message": rep.message,
                    },
                )
            except Exception:
                pass

        return rep
