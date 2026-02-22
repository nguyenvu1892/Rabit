# observer/outcome_updater.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def _pick_regime(risk_cfg: Any) -> str:
    if not isinstance(risk_cfg, dict):
        return "unknown"
    r = risk_cfg.get("regime") or risk_cfg.get("market_regime") or risk_cfg.get("state")
    return str(r) if r is not None else "unknown"


def _pick_expert(risk_cfg: Any, meta: Any) -> str:
    # prefer explicit risk_cfg.expert, then meta.expert, then UNKNOWN
    if isinstance(risk_cfg, dict):
        e = risk_cfg.get("expert") or (risk_cfg.get("meta", {}) or {}).get("expert")
        if e:
            return str(e)
    if isinstance(meta, dict):
        e2 = meta.get("expert")
        if e2:
            return str(e2)
    return "UNKNOWN"


@dataclass
class OutcomeUpdater:
    """
    Compat-first OutcomeUpdater:
    - Accepts extra kwargs without crashing (autosave, learner, weights_path...)
    - Keeps API: process_outcome(snapshot, outcome)
    """
    weight_store: Any
    journal_path: str = "data/journal_train.jsonl"
    autosave: bool = True

    # NOTE: dataclass would generate __init__ without **kwargs.
    # We add compat __init__ manually to avoid unexpected keyword errors.
    def __init__(self, weight_store: Any, journal_path: str = "data/journal_train.jsonl", autosave: bool = True, **kwargs) -> None:
        self.weight_store = weight_store

        # compat aliases
        self.journal_path = str(kwargs.get("journal") or kwargs.get("journal_path") or journal_path)
        self.autosave = bool(kwargs.get("autosave", autosave))

        # accept but ignore (compat): learner, weights_path, etc.
        # (do NOT remove: keeps shadow_run older/newer versions working)
        self._compat_extra = dict(kwargs)

    def process_outcome(self, snapshot: Dict[str, Any], outcome: Dict[str, Any]) -> None:
        """
        Update weight_store by reward * confidence (very light stability).
        We DO NOT change schemas; only consume existing fields.
        """
        features = snapshot.get("features") or {}
        risk_cfg = snapshot.get("risk_cfg") or {}
        meta = snapshot.get("meta") or {}

        expert = _pick_expert(risk_cfg, meta)
        regime = _pick_regime(risk_cfg)

        win = bool(outcome.get("win", False))
        pnl = _safe_float(outcome.get("pnl"), 0.0)

        # reward shaping (simple, stable):
        # - win => +1, loss => -1
        # - scaled by small factor + optional confidence
        reward = 1.0 if win else -1.0

        conf = _safe_float(risk_cfg.get("regime_conf") or risk_cfg.get("confidence"), 0.0)
        conf = max(0.0, min(1.0, conf if conf > 0 else 1.0))  # default 1.0 if missing

        # delta small to avoid exploding weights
        delta = 0.02 * reward * conf

        if hasattr(self.weight_store, "update"):
            try:
                self.weight_store.update(expert, regime, delta)
            except Exception:
                # fallback: set/get style
                try:
                    cur = 1.0
                    if hasattr(self.weight_store, "get"):
                        cur = float(self.weight_store.get(expert, regime, 1.0))
                    if hasattr(self.weight_store, "set"):
                        self.weight_store.set(expert, regime, cur + delta)
                except Exception:
                    pass

        if self.autosave and hasattr(self.weight_store, "save"):
            try:
                self.weight_store.save()
            except Exception:
                pass

        # journal is optional; keep compat (do not enforce)
        j = self._compat_extra.get("journal_obj")
        if j is not None and hasattr(j, "log_outcome"):
            try:
                j.log_outcome(step=snapshot.get("step"), outcome=outcome)
            except Exception:
                pass