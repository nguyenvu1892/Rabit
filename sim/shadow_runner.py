# sim/shadow_runner.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional
import math
import random
import traceback

from brain.features.feature_pack_v1 import FeaturePackV1

# ----------------------------
# Legacy helpers (KEEP)
# ----------------------------
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


# ----------------------------
# NEW compat helpers (ADD)
# ----------------------------
def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _sigmoid(x: float) -> float:
    # stable sigmoid
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _score_to_01(score: Any) -> float:
    """
    Compat layer:
      - If score already in [0,1] -> keep.
      - Else treat as "logit-like / unbounded score" -> map via sigmoid to (0,1).
    """
    s = _safe_float(score, 0.0)
    if 0.0 <= s <= 1.0:
        return s
    return _sigmoid(s)


def _extract_expert(risk_cfg: Dict[str, Any]) -> str:
    """
    Best-effort extract expert name for learning snapshot.
    We DO NOT change schema; we only enrich snapshot/meta.
    """
    if not isinstance(risk_cfg, dict):
        return "UNKNOWN"
    e = risk_cfg.get("expert")
    if isinstance(e, str) and e.strip():
        return e.strip()
    m = risk_cfg.get("meta")
    if isinstance(m, dict):
        e2 = m.get("expert") or m.get("expert_name")
        if isinstance(e2, str) and e2.strip():
            return e2.strip()
    return "UNKNOWN"


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
    """

    def __init__(
        self,
        decision_engine,
        risk_engine=None,
        outcome_updater=None,
        seed: Optional[int] = None,
        train: bool = False,
        **kwargs,
    ) -> None:
        self.de = decision_engine
        self.risk_engine = risk_engine
        self.outcome_updater = outcome_updater
        self.train = bool(train)
        self.rng = random.Random(seed)

    # ----------------------------
    # Legacy sim (KEEP) + compat normalization (ADD)
    # ----------------------------
    def simulate_outcome(self, score: float) -> Dict[str, Any]:
        """
        Legacy expectation: score in [0..1].
        Compat: normalize incoming score (which might be unbounded) to score01.
        """
        score01 = _score_to_01(score)
        # simple sim: higher score -> higher win chance
        # KEEP legacy formula but feed normalized score
        p_win = max(0.05, min(0.95, 0.5 + 0.4 * (score01 - 0.5)))
        win = self.rng.random() < p_win
        pnl = self.rng.uniform(0.2, 1.2) if win else -self.rng.uniform(0.2, 1.2)
        return {"win": win, "pnl": pnl, "p_win": p_win, "score01": score01}

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
                # legacy signature: set_exploration(float)
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

            # base features
            trade_features = {"candles": window, "step": i}

            # FeaturePack-v1 injection (already existing style, keep)
            if "candles" in trade_features:
                trade_features.update(FeaturePackV1.compute(trade_features["candles"]))

            try:
                allow, score, risk_cfg = self.de.evaluate_trade(trade_features)
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

                # Journal decision (KEEP)
                if journal is not None:
                    try:
                        journal.log_decision(
                            step=i,
                            allow=bool(allow),
                            score=float(_safe_float(score, 0.0)),
                            risk=risk_cfg,
                            forced=forced,
                            payload={"candles_len": len(window)},
                        )
                    except Exception:
                        pass

                # training outcome
                if train_mode and bool(allow):
                    # simulator expects score01 -> handled in simulate_outcome
                    outcome = self.simulate_outcome(_safe_float(score, 0.0))
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

                    # ----------------------------
                    # COMPAT snapshot enrichment (ADD, schema-safe)
                    # ----------------------------
                    meta = (risk_cfg.get("meta", {}) or {})
                    if not isinstance(meta, dict):
                        meta = {}

                    expert = _extract_expert(risk_cfg)

                    meta.setdefault("expert", expert)
                    meta.setdefault("regime", regime)
                    meta.setdefault("regime_conf", conf)
                    meta.setdefault("score_raw", _safe_float(score, 0.0))
                    meta.setdefault("score01", _score_to_01(score))

                    snapshot = {
                        "step": i,
                        "features": trade_features,
                        "risk_cfg": risk_cfg,
                        "meta": meta,
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
                    regime_local = _regime_key(risk_cfg_local)
                    stats._rb_row(regime_local)["errors"] += 1
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