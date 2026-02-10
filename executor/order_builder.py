# executor/order_builder.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from broker.contracts import OrderPlan


@dataclass
class OrderBuilder:
    """
    V1 OrderBuilder: build market orders from trade_features + risk_config.

    side logic (V1):
      - trend_state == "up"  -> buy
      - trend_state == "down"-> sell
      - else -> None (no order)
    """
    default_volume: float = 0.01
    default_max_slippage: float = 0.0

    def build(
        self,
        intent_id: str,
        trade_features: Dict[str, Any],
        risk_config: Optional[Dict[str, Any]] = None,
    ) -> Optional[OrderPlan]:
        symbol = str(trade_features.get("symbol", "XAUUSD"))
        trend = trade_features.get("trend_state")

        if trend == "up":
            side = "buy"
        elif trend == "down":
            side = "sell"
        else:
            return None

        rc = risk_config or {}
        volume = float(rc.get("volume", self.default_volume))
        sl = rc.get("sl")
        tp = rc.get("tp")

        # keep sl/tp optional; broker adapter may ignore if None
        plan = OrderPlan(
            intent_id=intent_id,
            symbol=symbol,
            side=side,
            volume=volume,
            sl=sl,
            tp=tp,
            max_slippage=float(rc.get("max_slippage", self.default_max_slippage)),
        )
        return plan
