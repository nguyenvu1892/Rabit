# sim/shadow_runner.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional
import random
import traceback


# =========================
# COMPAT: FeaturePackV1 (optional)
# =========================
try:
    from brain.features.feature_pack_v1 import FeaturePackV1  # type: ignore
except Exception:
    FeaturePackV1 = None  # type: ignore


def _regime_key(risk_cfg: Dict[str, Any]) -> str:
    """Best-effort regime key extraction (compat across versions)."""
    if not isinstance(risk_cfg, dict):
        return "unknown"
    r = risk_cfg.get("regime") or risk_cfg.get("market_regime") or risk_cfg.get("state")
    return str(r) if r is not None else "unknown"


def _regime_conf(risk_cfg: Dict[str, Any]) -> float:
    """Best-effort regime confidence extraction (compat across versions)."""
    if not isinstance(risk_cfg, dict):
        return 0.0
    c = risk_cfg.get("regime_conf") or risk_cfg.get("confidence") or 0.0
    try:
        return float(c)
    except Exception:
        return 0.0


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

    # per-regime aggregation (schema used by tools/shadow_run.py prints)
    regime_breakdown: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def _rb_row(self, regime: str) -> Dict[str, Any]:
        row = self.regime_breakdown.get(regime)
        if row is None:
            row = {
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
            }
            self.regime_breakdown[regime] = row
        return row

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
            "regime_breakdown": self.regime_breakdown,
        }


class ShadowRunner:
    """
    Compat-first ShadowRunner.

    Goals:
    - DO NOT break old call sites.
    - Accept extra kwargs (risk_engine, journal, debug...) safely.
    - Keep the original evaluate loop + stats schema.

    Supported call patterns:
      runner = ShadowRunner(decision_engine=de, risk_engine=..., outcome_updater=..., train=..., debug=...)
      stats  = runner.run(candles, lookback=..., max_steps=..., horizon=..., train=..., epsilon=..., epsilon_cooldown=..., journal=...)
      stats  = runner.run(candles=..., ...)
      stats  = runner.run(rows=..., ...)
      stats  = runner.run(data=..., ...)
    """

    def __init__(
        self,
        decision_engine=None,
        risk_engine=None,
        outcome_updater=None,
        seed: Optional[int] = None,
        train: bool = False,
        debug: bool = False,
        **kwargs,
    ) -> None:
        # -------------------------
        # COMPAT: accept both "decision_engine" and legacy "de" via kwargs
        # -------------------------
        if decision_engine is None:
            decision_engine = kwargs.get("de") or kwargs.get("engine")

        self.decision_engine = decision_engine
        self.de = decision_engine  # alias (some code expects self.de)

        self.risk_engine = risk_engine  # kept for forward-compat; may be unused here
        self.outcome_updater = outcome_updater

        self.train = bool(train)
        self.debug = bool(debug)  # some error handlers check self.debug

        self.rng = random.Random(seed)

    # -------------------------
    # Core sim outcome (kept simple + deterministic-ish)
    # -------------------------
    def simulate_outcome(self, score: float) -> Dict[str, Any]:
        # Higher score -> higher win chance
        p_win = max(0.05, min(0.95, 0.5 + 0.4 * (score - 0.5)))
        win = self.rng.random() < p_win
        pnl = self.rng.uniform(0.2, 1.2) if win else -self.rng.uniform(0.2, 1.2)
        return {"win": win, "pnl": pnl}

    def run(
        self,
        candles=None,
        *,
        rows=None,
        data=None,
        lookback: int = 300,
        max_steps: int = 2000,
        horizon: int = 30,
        train: Optional[bool] = None,
        epsilon: float = 0.0,
        epsilon_cooldown: int = 0,
        journal=None,
        **kwargs,
    ) -> ShadowStats:
        """
        Main loop.

        IMPORTANT compat behavior:
        - accepts journal kwarg (tools/shadow_run passes it)
        - accepts arbitrary **kwargs to avoid 'unexpected keyword' explosions
        """

        # -------------------------
        # COMPAT: input selection
        # -------------------------
        if candles is None:
            candles = rows if rows is not None else data

        stats = ShadowStats()
        if candles is None:
            return stats

        de = self.de or self.decision_engine
        if de is None:
            # No decision engine => can't evaluate anything
            return stats

        train_mode = self.train if train is None else bool(train)

        # -------------------------
        # COMPAT: push exploration if supported
        # -------------------------
        if hasattr(de, "set_exploration"):
            try:
                de.set_exploration(epsilon=float(epsilon), cooldown=int(epsilon_cooldown))
            except TypeError:
                # older signature: set_exploration(float)
                try:
                    de.set_exploration(float(epsilon))
                except Exception:
                    pass
            except Exception:
                pass

        n = len(candles)
        start = max(int(lookback), 0)
        end = min(n - int(horizon), start + int(max_steps))
        if end <= start:
            return stats

        # -------------------------
        # Core evaluation loop (kept)
        # -------------------------
        for i in range(start, end):
            stats.steps += 1

            try:
                window = candles[i - lookback : i]

                # Keep schema: candles + step
                trade_features: Dict[str, Any] = {"candles": window, "step": i}

                # -------------------------
                # COMPAT: FeaturePack injection (non-breaking)
                # -------------------------
                if FeaturePackV1 is not None and "candles" in trade_features:
                    try:
                        trade_features.update(FeaturePackV1.compute(trade_features["candles"]))
                    except Exception:
                        pass

                # DecisionEngine API (current): allow, score, risk_cfg
                allow, score, risk_cfg = de.evaluate_trade(trade_features)
                stats.decisions += 1

                risk_cfg = risk_cfg or {}
                forced = bool(risk_cfg.get("forced", False))

                regime = _regime_key(risk_cfg)
                conf = _regime_conf(risk_cfg)
                row = stats._rb_row(regime)

                row["decisions"] += 1
                if conf > 0:
                    row["conf_sum"] += float(conf)
                    row["conf_n"] += 1

                if bool(allow):
                    stats.allow += 1
                    row["allow"] += 1
                    if forced:
                        stats.forced_entries += 1
                        row["forced"] += 1
                else:
                    stats.deny += 1
                    row["deny"] += 1

                # journal decision (optional)
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

                # training outcome only on allowed trades
                if train_mode and bool(allow):
                    outcome = self.simulate_outcome(float(score))

                    stats.outcomes += 1
                    row["outcomes"] += 1

                    if outcome.get("win"):
                        stats.wins += 1
                        row["wins"] += 1
                    else:
                        stats.losses += 1
                        row["losses"] += 1

                    pnl = float(outcome.get("pnl", 0.0))
                    stats.total_pnl += pnl
                    row["pnl"] += pnl

                    snapshot = {
                        "step": i,
                        "features": trade_features,
                        "risk_cfg": risk_cfg,
                        "meta": (risk_cfg.get("meta", {}) or {}),
                    }

                    # outcome updater hook (optional)
                    if self.outcome_updater is not None:
                        try:
                            # prefer process_outcome if present
                            if hasattr(self.outcome_updater, "process_outcome"):
                                self.outcome_updater.process_outcome(snapshot, outcome)
                            elif hasattr(self.outcome_updater, "on_outcome"):
                                self.outcome_updater.on_outcome(snapshot, outcome)
                        except Exception:
                            pass

                    # journal outcome (optional)
                    if journal is not None:
                        try:
                            journal.log_outcome(step=i, outcome=outcome)
                        except Exception:
                            pass

            except Exception as e:
                stats.errors += 1
                try:
                    risk_cfg_local = locals().get("risk_cfg") or {}
                    regime = _regime_key(risk_cfg_local)
                    stats._rb_row(regime)["errors"] += 1
                except Exception:
                    pass

                # Print only first few errors to keep console readable
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