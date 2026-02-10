class TradeOutcomeTracker:

    def __init__(self):
        self.active_trades = {}

    # -------------------------------------------------

    def register_trade(self, trade_id, snapshot, entry_price):
        """
        Called when trade opens
        """
        self.active_trades[trade_id] = {
            "snapshot": snapshot,
            "entry_price": entry_price,
            "max_drawdown": 0,
            "max_profit": 0
        }

    # -------------------------------------------------

    def update_trade(self, trade_id, current_price):
        """
        Called during trade lifecycle
        """
        if trade_id not in self.active_trades:
            return

        trade = self.active_trades[trade_id]

        entry = trade["entry_price"]
        pnl = current_price - entry

        trade["max_profit"] = max(trade["max_profit"], pnl)
        trade["max_drawdown"] = min(trade["max_drawdown"], pnl)

    # -------------------------------------------------

    def close_trade(self, trade_id, exit_price, duration):

        if trade_id not in self.active_trades:
            return None

        trade = self.active_trades.pop(trade_id)

        entry = trade["entry_price"]
        pnl = exit_price - entry

        outcome = {
            "pnl": pnl,
            "win": pnl > 0,
            "duration": duration,
            "max_drawdown": trade["max_drawdown"],
            "max_profit": trade["max_profit"],
            "snapshot": trade["snapshot"]
        }

        return outcome
