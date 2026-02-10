# brain/strategy_policy.py
from __future__ import annotations
from typing import Dict, Any


class StrategyPolicy:
    """
    Convert a strategy genome into decision rules usable by DecisionEngine.
    V1 is conservative and non-invasive.
    """

    def __init__(self, genome: Dict[str, Any]):
        self.genome = genome or {}

        # Defaults (safe fallbacks)
        self.entry_threshold = float(self.genome.get("entry_threshold", 0.5))
        self.sl_atr_mult = float(self.genome.get("sl_atr_mult", 1.5))
        self.tp_atr_mult = float(self.genome.get("tp_atr_mult", 3.0))
        self.risk_per_trade = float(self.genome.get("risk_per_trade", 0.01))
        self.only_trend = bool(self.genome.get("only_trend", 0))
        self.avoid_high_vol = bool(self.genome.get("avoid_high_vol", 0))

    # -------- Entry Gate --------
    def allow_entry(self, snapshot: Dict[str, Any]) -> bool:
        """
        Decide whether to allow entry based on snapshot signals.
        Expected snapshot keys (best-effort):
          - score (float 0..1)
          - trend_state (str/bool)
          - volatility_state (str)
        """
        score = float(snapshot.get("score", 0.0))
        if score < self.entry_threshold:
            return False

        if self.only_trend:
            # Accept common encodings
            trend_ok = snapshot.get("trend_state") in (True, "up", "down", "trend")
            if not trend_ok:
                return False

        if self.avoid_high_vol:
            if snapshot.get("volatility_state") in ("high", "extreme"):
                return False

        return True

    # -------- Risk Params --------
    def risk_params(self, snapshot: Dict[str, Any]) -> Dict[str, float]:
        """
        Return risk parameters for RiskEngine.
        """
        return {
            "sl_atr_mult": float(self.sl_atr_mult),
            "tp_atr_mult": float(self.tp_atr_mult),
            "risk_per_trade": float(self.risk_per_trade),
        }

    # -------- Debug --------
    def explain(self) -> Dict[str, Any]:
        return {
            "entry_threshold": self.entry_threshold,
            "sl_atr_mult": self.sl_atr_mult,
            "tp_atr_mult": self.tp_atr_mult,
            "risk_per_trade": self.risk_per_trade,
            "only_trend": self.only_trend,
            "avoid_high_vol": self.avoid_high_vol,
        }
