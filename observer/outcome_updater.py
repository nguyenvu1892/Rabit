# observer/outcome_updater.py
from __future__ import annotations

from typing import Any, Dict, Optional
import time

# ----------------------------
# Legacy helpers (KEEP)
# ----------------------------
def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _ensure_dict(x: Any) -> Dict[str, Any]:
    return x if isinstance(x, dict) else {}


# ----------------------------
# OutcomeUpdater
# ----------------------------
class OutcomeUpdater:
    """
    KEEP existing behavior:
      - process_outcome(snapshot, outcome) updates weight_store
      - can write journal lines if journal_path provided

    COMPAT (ADD):
      - accept unexpected kwargs: learner, autosave, weights_path, save_every...
      - autosave kw from tools/shadow_run.py no longer crashes
    """

    def __init__(
        self,
        weight_store,
        journal_path: Optional[str] = None,
        # legacy / compat params:
        weights_path: Optional[str] = None,
        autosave: bool = False,
        save_every: int = 1,
        learner=None,
        **kwargs,
    ) -> None:
        self.weight_store = weight_store
        self.journal_path = journal_path
        self.weights_path = weights_path
        self.autosave = bool(autosave)
        self.save_every = max(1, int(save_every))
        self.learner = learner  # kept for forward-compat (not required)

        self._n = 0
        self._last_save_ts = 0.0

        # COMPAT: accept alternative names without breaking callers
        # (do NOT change schema; only map)
        if "weights" in kwargs and self.weights_path is None:
            try:
                self.weights_path = str(kwargs["weights"])
            except Exception:
                pass
        if "journal" in kwargs and self.journal_path is None:
            try:
                self.journal_path = str(kwargs["journal"])
            except Exception:
                pass
        if "autosave_every" in kwargs:
            try:
                self.save_every = max(1, int(kwargs["autosave_every"]))
            except Exception:
                pass

    # ----------------------------
    # Extraction helpers (KEEP + improve robustness)
    # ----------------------------
    def _extract_expert(self, snapshot: Dict[str, Any]) -> str:
        risk_cfg = _ensure_dict(snapshot.get("risk_cfg"))
        e = risk_cfg.get("expert")
        if isinstance(e, str) and e.strip():
            return e.strip()
        meta = _ensure_dict(risk_cfg.get("meta"))
        e2 = meta.get("expert") or meta.get("expert_name")
        if isinstance(e2, str) and e2.strip():
            return e2.strip()
        return "UNKNOWN"

    def _extract_regime(self, snapshot: Dict[str, Any]) -> str:
        risk_cfg = _ensure_dict(snapshot.get("risk_cfg"))
        r = risk_cfg.get("regime") or risk_cfg.get("market_regime") or risk_cfg.get("state")
        if isinstance(r, str) and r.strip():
            return r.strip()
        meta = _ensure_dict(risk_cfg.get("meta"))
        r2 = meta.get("regime") or meta.get("market_regime") or meta.get("state")
        if isinstance(r2, str) and r2.strip():
            return r2.strip()
        return "unknown"

    def _extract_confidence(self, snapshot: Dict[str, Any]) -> float:
        risk_cfg = _ensure_dict(snapshot.get("risk_cfg"))
        c = risk_cfg.get("regime_conf") or risk_cfg.get("confidence")
        if c is None:
            meta = _ensure_dict(risk_cfg.get("meta"))
            c = meta.get("regime_conf") or meta.get("confidence") or meta.get("score01")
        return _safe_float(c, 0.0)

    # ----------------------------
    # Journal writing (KEEP, best-effort)
    # ----------------------------
    def _append_journal(self, row: Dict[str, Any]) -> None:
        if not self.journal_path:
            return
        try:
            import json

            with open(self.journal_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        except Exception:
            pass

    # ----------------------------
    # Weight saving (KEEP + compat)
    # ----------------------------
    def _maybe_save(self) -> None:
        if not self.weights_path:
            return
        # autosave: every save_every outcomes OR if autosave True (still respects save_every)
        if (self._n % self.save_every) != 0:
            return
        try:
            # avoid overly frequent saves in tight loops
            now = time.time()
            if now - self._last_save_ts < 0.2:
                return
            self._last_save_ts = now
            self.weight_store.save(self.weights_path)
        except Exception:
            pass

    # ----------------------------
    # Core API (KEEP)
    # ----------------------------
    def process_outcome(self, snapshot: Dict[str, Any], outcome: Dict[str, Any]) -> None:
        """
        snapshot schema: {"step":..., "features":..., "risk_cfg":..., "meta":...}
        outcome schema: {"win": bool, "pnl": float, ...}

        Updates weight_store with delta ~ reward * confidence.
        """
        self._n += 1
        snapshot = _ensure_dict(snapshot)
        outcome = _ensure_dict(outcome)

        expert = self._extract_expert(snapshot)
        regime = self._extract_regime(snapshot)
        conf = self._extract_confidence(snapshot)

        pnl = _safe_float(outcome.get("pnl", 0.0), 0.0)
        win = bool(outcome.get("win", pnl > 0))

        # Legacy-like reward: use pnl directly (already signed)
        reward = pnl

        # COMPAT stability layer: clamp conf into [0..1] softly (without assuming score range)
        if conf < 0.0:
            conf = 0.0
        if conf > 1.0 and conf < 10.0:
            # if conf is like "score raw", shrink (still monotonic)
            conf = min(1.0, conf / 10.0)
        if conf > 1.0:
            conf = 1.0

        delta = reward * conf

        # update store (KEEP method name expected by current weight_store.py)
        try:
            if hasattr(self.weight_store, "update_expert_regime"):
                self.weight_store.update_expert_regime(expert=expert, regime=regime, delta=delta)
            else:
                # last resort: try generic update(key, delta)
                key = f"{expert}|{regime}"
                if hasattr(self.weight_store, "update"):
                    self.weight_store.update(key, delta)  # type: ignore
        except Exception:
            pass

        # journal row (best effort)
        self._append_journal(
            {
                "t": time.time(),
                "expert": expert,
                "regime": regime,
                "conf": conf,
                "reward": reward,
                "delta": delta,
                "win": win,
                "pnl": pnl,
            }
        )

        # save weights
        if self.autosave or self.weights_path:
            self._maybe_save()