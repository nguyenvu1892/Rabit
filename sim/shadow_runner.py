# sim/shadow_runner.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional
import random
import traceback


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

    def to_dict(self) -> Dict[str, Any]:
        return {
            "steps": self.steps,
            "decisions": self.decisions,
            "allow": self.allow,
            "deny": self.deny,
            "errors": self.errors,
            "outcomes": self.outcomes,
            "wins": self.wins,
            "losses": self.losses,
            "total_pnl": self.total_pnl,
            "forced_entries": self.forced_entries,
        }


class ShadowRunner:
    def __init__(
        self,
        decision_engine,
        risk_engine=None,
        outcome_updater=None,
        seed: Optional[int] = None,
        train: bool = False,
    ) -> None:
        self.de = decision_engine
        self.risk_engine = risk_engine
        self.outcome_updater = outcome_updater
        self.train = bool(train)
        self.rng = random.Random(seed)

    def simulate_outcome(self, score: float) -> Dict[str, Any]:
        # simple sim: higher score -> higher win chance
        p_win = max(0.05, min(0.95, 0.5 + 0.4 * (score - 0.5)))
        win = self.rng.random() < p_win
        pnl = self.rng.uniform(0.2, 1.2) if win else -self.rng.uniform(0.2, 1.2)
        return {"win": win, "pnl": pnl}

    def run(
        self,
        candles,
        lookback: int = 300,
        max_steps: int = 2000,
        horizon: int = 30,
        train: Optional[bool] = None,
        epsilon: float = 0.0,
        epsilon_cooldown: int = 0,
        journal=None,
    ) -> ShadowStats:
        # unify train flag (không bị “lạc”)
        train_mode = self.train if train is None else bool(train)

        stats = ShadowStats()
        n = len(candles)

        # push exploration config into decision engine (nếu có)
        if hasattr(self.de, "set_exploration"):
            try:
                # new signature: set_exploration(epsilon=..., cooldown=...)
                self.de.set_exploration(epsilon=float(epsilon), cooldown=int(epsilon_cooldown))
            except TypeError:
                # old signature: set_exploration(epsilon)
                try:
                    self.de.set_exploration(float(epsilon))
                except Exception:
                    pass
            except Exception:
                pass

        start = max(int(lookback), 0)
        end = min(n - int(horizon), start + int(max_steps))

        if end <= start:
            return stats

        for i in range(start, end):
            stats.steps += 1
            window = candles[i - lookback : i]

            trade_features = {
                "candles": window,
                "step": i,
            }

            try:
                allow, score, risk_cfg = self.de.evaluate_trade(trade_features)
                stats.decisions += 1

                risk_cfg = risk_cfg or {}
                forced = bool(risk_cfg.get("forced", False))

                if bool(allow):
                    stats.allow += 1
                    if forced:
                        stats.forced_entries += 1
                else:
                    stats.deny += 1

                # journal decision
                if journal is not None:
                    try:
                        journal.log_decision(
                            step=i,
                            allow=bool(allow),
                            score=float(score),
                            risk=risk_cfg,
                            forced=forced,
                            payload={"candles_len": len(window)},
                        )
                    except Exception:
                        pass

                # training outcome
                if train_mode and bool(allow):
                    outcome = self.simulate_outcome(float(score))

                    stats.outcomes += 1
                    if outcome.get("win"):
                        stats.wins += 1
                    else:
                        stats.losses += 1
                    stats.total_pnl += float(outcome.get("pnl", 0.0))

                    # snapshot đầy đủ cho OutcomeUpdater học expert/regime/forced
                    snapshot = {
                        "step": i,
                        "features": trade_features,
                        "risk_cfg": risk_cfg,
                        "meta": (risk_cfg.get("meta", {}) or {}),
                    }

                    if self.outcome_updater is not None:
                        try:
                            self.outcome_updater.process_outcome(snapshot, outcome)
                        except Exception:
                            pass

                    if journal is not None:
                        try:
                            journal.log_outcome(step=i, outcome=outcome)
                        except Exception:
                            pass

            except Exception as e:
                stats.errors += 1

                # debug: in 3 lỗi đầu tiên ra terminal cho dễ bắt gốc
                if stats.errors <= 3:
                    print("[ShadowRunner] FIRST ERROR:", repr(e))
                    traceback.print_exc()

                if journal is not None:
                    try:
                        journal.log_error(step=i, error=traceback.format_exc())
                    except Exception:
                        pass

                continue

        return stats
