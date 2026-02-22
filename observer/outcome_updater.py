# observer/outcome_updater.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional
import time

from brain.weight_store import WeightStore


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _get_meta(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    m = snapshot.get("meta")
    if isinstance(m, dict):
        return m
    return {}


def _get_risk_cfg(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    r = snapshot.get("risk_cfg")
    if isinstance(r, dict):
        return r
    return {}


def _extract_regime(snapshot: Dict[str, Any]) -> str:
    risk_cfg = _get_risk_cfg(snapshot)
    meta = _get_meta(snapshot)

    r = risk_cfg.get("regime") or risk_cfg.get("market_regime") or risk_cfg.get("state")
    if isinstance(r, str) and r.strip():
        return r.strip()

    r2 = meta.get("regime") or meta.get("market_regime") or meta.get("state")
    if isinstance(r2, str) and r2.strip():
        return r2.strip()

    return "unknown"


def _extract_expert(snapshot: Dict[str, Any]) -> str:
    """
    Compat-first:
    - prefer risk_cfg['expert']
    - else risk_cfg['meta']['expert']
    - else snapshot['meta']['expert']
    - else fallback "UNKNOWN"
    """
    risk_cfg = _get_risk_cfg(snapshot)
    meta = _get_meta(snapshot)

    e = risk_cfg.get("expert")
    if isinstance(e, str) and e.strip():
        return e.strip()

    rm = risk_cfg.get("meta")
    if isinstance(rm, dict):
        e2 = rm.get("expert") or rm.get("expert_name")
        if isinstance(e2, str) and e2.strip():
            return e2.strip()

    e3 = meta.get("expert") or meta.get("expert_name")
    if isinstance(e3, str) and e3.strip():
        return e3.strip()

    return "UNKNOWN"


@dataclass
class OutcomeUpdater:
    """
    Online learning v1:
    - update WeightStore.expert_regime with delta = lr * reward * confidence
    - keep schema unchanged
    """

    weight_store: WeightStore
    lr: float = 0.01
    min_w: float = 0.05
    max_w: float = 10.0
    debug: bool = False

    # existing (used by tools/shadow_run.py)
    journal_path: Optional[str] = None
    weights_path: Optional[str] = None

    # ----------------------------
    # COMPAT ADD: autosave flag
    # tools/shadow_run.py may pass autosave=...
    # ----------------------------
    autosave: bool = False

    def __post_init__(self) -> None:
        # compat: ensure numeric
        try:
            self.lr = float(self.lr)
        except Exception:
            self.lr = 0.01
        try:
            self.min_w = float(self.min_w)
        except Exception:
            self.min_w = 0.05
        try:
            self.max_w = float(self.max_w)
        except Exception:
            self.max_w = 10.0

        # compat: autosave normalization
        self.autosave = bool(self.autosave)

    # ----------------------------
    # KEEP API
    # ----------------------------
    def process_outcome(self, snapshot: Dict[str, Any], outcome: Dict[str, Any]) -> None:
        expert = _extract_expert(snapshot)
        regime = _extract_regime(snapshot)

        # reward signal
        win = bool(outcome.get("win", False))
        pnl = _safe_float(outcome.get("pnl", 0.0), 0.0)

        # keep it simple: reward = sign(pnl) (or win/loss)
        reward = 1.0 if pnl > 0 else (-1.0 if pnl < 0 else (1.0 if win else -1.0))

        # confidence: prefer regime_conf/confidence/score01 if provided
        risk_cfg = _get_risk_cfg(snapshot)
        meta = _get_meta(snapshot)

        conf = (
            risk_cfg.get("regime_conf")
            or risk_cfg.get("confidence")
            or meta.get("regime_conf")
            or meta.get("confidence")
            or meta.get("score01")
            or 0.0
        )
        conf_f = max(0.0, min(1.0, _safe_float(conf, 0.0)))

        key = f"{expert}|{regime}"

        # update weight
        delta = float(self.lr) * float(reward) * float(conf_f)

        cur = self.weight_store.get("expert_regime", key, default=1.0)
        nxt = cur + delta
        if nxt < self.min_w:
            nxt = self.min_w
        if nxt > self.max_w:
            nxt = self.max_w

        self.weight_store.set("expert_regime", key, nxt)

        # update meta timestamps (schema-safe)
        d = self.weight_store.data.get("meta")
        if isinstance(d, dict):
            d["updated_at"] = time.time()

        # ----------------------------
        # COMPAT SAVE LOGIC (ADD)
        # - old behavior: save if weights_path exists
        # - new compat: save if autosave=True and weights_path exists
        # ----------------------------
        if self.weights_path and (self.autosave or True):
            # keep legacy "always save when weights_path is provided"
            try:
                self.weight_store.save(self.weights_path)
            except Exception:
                pass

        if self.debug:
            print(
                f"[OutcomeUpdater] {key} cur={cur:.4f} delta={delta:.4f} nxt={nxt:.4f} "
                f"conf={conf_f:.3f} pnl={pnl:.3f} autosave={self.autosave}"
            )