# sim/shadow_runner.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple


@dataclass
class ShadowStats:
    steps: int = 0
    decisions: int = 0
    allow: int = 0
    deny: int = 0
    errors: int = 0
    outcomes: int = 0
    wins: int = 0
    losses: int = 0
    total_pnl: float = 0.0
    forced_entries: int = 0


class ShadowRunner:
    """
    Runs simulation loop and calls DecisionEngine for each step.
    """

    def __init__(
        self,
        decision_engine: Any,
        broker: Any,
        lookback: int = 300,
        horizon: int = 30,
        logger: Optional[Any] = None,
    ) -> None:
        self.de = decision_engine
        self.broker = broker
        self.lookback = int(lookback)
        self.horizon = int(horizon)
        self.logger = logger

    def _build_trade_features(self) -> Dict[str, Any]:
        # IMPORTANT: provide candles in keys that meta/regime can read
        candles = self.broker.get_window(self.lookback)  # list of bars/candles

        return {
            # New canonical key
            "candles_window": candles,
            # Backward-compat keys
            "candles": candles,
            "window": candles,
        }

    def run(self, max_steps: int = 2000) -> Tuple[ShadowStats, Dict[str, Any]]:
        stats = ShadowStats()
        regime_breakdown: Dict[str, Any] = {}

        for _ in range(int(max_steps)):
            stats.steps += 1

            features = self._build_trade_features()

            try:
                allow, score, risk_cfg = self.de.evaluate_trade(features)
                stats.decisions += 1
            except Exception as e:
                stats.errors += 1
                if self.logger:
                    self.logger.error(f"[ShadowRunner] evaluate_trade error: {e}")
                continue

            # regime key should be string (NOT RegimeResult object)
            regime = str(risk_cfg.get("regime") or features.get("regime") or "unknown")

            bucket = regime_breakdown.setdefault(
                regime,
                {
                    "decisions": 0,
                    "allow": 0,
                    "deny": 0,
                    "errors": 0,
                    "outcomes": 0,
                    "wins": 0,
                    "losses": 0,
                    "pnl": 0.0,
                    "forced": 0,
                    "conf_sum": 0.0,
                    "conf_n": 0,
                },
            )
            bucket["decisions"] += 1

            if allow:
                stats.allow += 1
                bucket["allow"] += 1

                # Execute simulated trade (broker decides pnl later)
                outcome = self.broker.execute_trade(risk_cfg=risk_cfg, horizon=self.horizon)
                if outcome is not None:
                    stats.outcomes += 1
                    bucket["outcomes"] += 1
                    pnl = float(outcome.get("pnl", 0.0) or 0.0)
                    stats.total_pnl += pnl
                    bucket["pnl"] += pnl
                    if pnl > 0:
                        stats.wins += 1
                        bucket["wins"] += 1
                    else:
                        stats.losses += 1
                        bucket["losses"] += 1

                # update learner/weights if engine supports it
                try:
                    self.de.on_outcome(outcome)
                except Exception:
                    pass
            else:
                stats.deny += 1
                bucket["deny"] += 1

        return stats, {"regime_breakdown": regime_breakdown}
