# sim/shadow_runner.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional
import random
import traceback

# NEW: FeaturePackV1 injection (compat-safe)
try:
    from brain.features.feature_pack_v1 import FeaturePackV1  # type: ignore
except Exception:
    FeaturePackV1 = None  # type: ignore


def _regime_key(risk_cfg: Dict[str, Any]) -> str:
    if not isinstance(risk_cfg, dict):
        return "unknown"
    r = risk_cfg.get("regime") or risk_cfg.get("market_regime") or risk_cfg.get("state")
    return str(r) if r is not None else "unknown"


def _regime_conf(risk_cfg: Dict[str, Any]) -> float:
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
    Compat-first ShadowRunner:
    - __init__ accepts risk_engine (optional) to avoid 'unexpected keyword'
    - run() accepts candles in many forms:
        run(candles, ...) OR run(candles=...) OR run(rows=...) OR run(data=...)
    - run() accepts journal kw (and ignores unknown kwargs safely)
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
        # ---- COMPAT: accept different ctor param names (without breaking old code) ----
        # tools/shadow_run may pass decision_engine=...
        if decision_engine is None and "de" in kwargs:
            decision_engine = kwargs.get("de")
        if decision_engine is None and "engine" in kwargs:
            decision_engine = kwargs.get("engine")

        # keep legacy field name used by existing code
        self.de = decision_engine
        # extra alias for safety (some code may reference these)
        self.decision_engine = decision_engine
        self.engine = decision_engine

        self.risk_engine = risk_engine
        self.outcome_updater = outcome_updater
        self.train = bool(train)
        self.debug = bool(debug)  # <-- fix "no attribute debug"
        self.rng = random.Random(seed)

        # stash any extra compat kwargs (do not crash)
        self._compat_kwargs = dict(kwargs)

    def simulate_outcome(self, score: float) -> Dict[str, Any]:
        # simple sim: higher score -> higher win chance
        p_win = max(0.05, min(0.95, 0.5 + 0.4 * (score - 0.5)))
        win = self.rng.random() < p_win
        pnl = self.rng.uniform(0.2, 1.2) if win else -self.rng.uniform(0.2, 1.2)
        return {"win": win, "pnl": pnl}

    # -------------------------------------------------------------------------
    # COMPAT ADD: inject regime into risk_cfg if missing (fix UNKNOWN breakdown)
    # Priority:
    # 1) decision_engine.regime_detector.detect(features) if exists
    # 2) features['regime'] / features['regime_conf'] if present
    # -------------------------------------------------------------------------
    def _ensure_regime_in_risk_cfg(self, trade_features: Dict[str, Any], risk_cfg: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(risk_cfg, dict):
            risk_cfg = {}

        # if already set -> do nothing
        if (risk_cfg.get("regime") is not None) and (risk_cfg.get("regime") != ""):
            return risk_cfg

        # try DecisionEngine.regime_detector
        try:
            rd = getattr(self.de, "regime_detector", None)
            if rd is not None and hasattr(rd, "detect"):
                rr = rd.detect(trade_features)
                # rr could be dataclass or dict-like
                regime = getattr(rr, "regime", None) if rr is not None else None
                conf = getattr(rr, "confidence", None) if rr is not None else None
                if regime:
                    risk_cfg["regime"] = str(regime)
                if conf is not None:
                    try:
                        risk_cfg["regime_conf"] = float(conf)
                    except Exception:
                        pass
                return risk_cfg
        except Exception:
            pass

        # fallback: from trade_features (if feature pack provided)
        try:
            if "regime" in trade_features and trade_features.get("regime") is not None:
                risk_cfg["regime"] = str(trade_features.get("regime"))
            if "regime_conf" in trade_features and trade_features.get("regime_conf") is not None:
                try:
                    risk_cfg["regime_conf"] = float(trade_features.get("regime_conf"))
                except Exception:
                    pass
            elif "confidence" in trade_features and trade_features.get("confidence") is not None:
                try:
                    risk_cfg["regime_conf"] = float(trade_features.get("confidence"))
                except Exception:
                    pass
        except Exception:
            pass

        return risk_cfg

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
        # compat input selection
        if candles is None:
            candles = rows if rows is not None else data

        stats = ShadowStats()
        if candles is None:
            return stats

        train_mode = self.train if train is None else bool(train)

        # push exploration config into decision engine (if supported)
        if hasattr(self.de, "set_exploration"):
            try:
                self.de.set_exploration(epsilon=float(epsilon), cooldown=int(epsilon_cooldown))
            except TypeError:
                try:
                    self.de.set_exploration(float(epsilon))
                except Exception:
                    pass
            except Exception:
                pass

        n = len(candles)
        start = max(int(lookback), 0)
        end = min(n - int(horizon), start + int(max_steps))
        if end <= start:
            return stats

        for i in range(start, end):
            stats.steps += 1
            window = candles[i - lookback : i]
            trade_features = {"candles": window, "step": i}

            # ===== FeaturePack injection (DO NOT break old schema) =====
            if FeaturePackV1 is not None and "candles" in trade_features:
                try:
                    trade_features.update(FeaturePackV1.compute(trade_features["candles"]))
                except Exception:
                    pass

            try:
                allow, score, risk_cfg = self.de.evaluate_trade(trade_features)
                stats.decisions += 1

                risk_cfg = risk_cfg or {}
                # ---- COMPAT: fix UNKNOWN by injecting regime if missing ----
                risk_cfg = self._ensure_regime_in_risk_cfg(trade_features, risk_cfg)

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
                try:
                    risk_cfg_local = locals().get("risk_cfg") or {}
                    regime = _regime_key(risk_cfg_local)
                    stats._rb_row(regime)["errors"] += 1
                except Exception:
                    pass

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