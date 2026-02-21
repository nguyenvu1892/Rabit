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


def _get_field(d: Any, key: str, default: Any = None) -> Any:
    if isinstance(d, dict):
        return d.get(key, default)
    return default


@dataclass
class OutcomeUpdateResult:
    updated: bool
    reason: str
    expert: str
    regime: str
    reward: float


class OutcomeUpdater:
    """
    Online updater (compat + stable):

    ShadowRunner in this repo calls:
      outcome_updater.process_outcome(snapshot, outcome)

    So we MUST provide process_outcome() as the wiring entrypoint.
    """

    def __init__(
        self,
        weight_store: Any = None,
        *,
        lr: float = 0.05,
        use_confidence: bool = True,
        autosave: bool = True,
        debug: bool = False,
        weights_path: Optional[str] = None,   # compat
        journal_path: Optional[str] = None,   # compat (unused here)
        learner: Any = None,                  # compat (unused here)
        **kwargs: Any,                        # absorb future params
    ) -> None:
        self.weight_store = weight_store
        self.lr = float(lr)
        self.use_confidence = bool(use_confidence)
        self.autosave = bool(autosave)
        self.debug = bool(debug)

        self.weights_path = weights_path
        self.journal_path = journal_path
        self.learner = learner
        self.extra = dict(kwargs) if kwargs else {}

    # ---------------------------------------------------------
    # ✅ CRITICAL COMPAT WIRING: ShadowRunner uses process_outcome
    # ---------------------------------------------------------
    def process_outcome(self, snapshot: Dict[str, Any], outcome: Dict[str, Any]) -> OutcomeUpdateResult:
        # ShadowRunner signature: (snapshot, outcome)
        return self.on_outcome(outcome, snapshot)

    # -----------------------------
    # Extractors (snapshot schema tolerant)
    # -----------------------------
    def _extract_expert(self, snapshot: Dict[str, Any]) -> str:
        # snapshot from ShadowRunner currently includes:
        #  { step, features, risk_cfg, meta }
        risk_cfg = _get_field(snapshot, "risk_cfg", {}) or {}
        meta = _get_field(snapshot, "meta", {}) or {}

        # best effort: try these fields
        exp = _get_field(risk_cfg, "expert", None)
        if exp:
            return str(exp)

        exp = _get_field(meta, "expert", None)
        if exp:
            return str(exp)

        # sometimes nested
        m2 = _get_field(risk_cfg, "meta", {}) or {}
        exp = _get_field(m2, "expert", None)
        if exp:
            return str(exp)

        return ""

    def _extract_regime(self, snapshot: Dict[str, Any]) -> str:
        risk_cfg = _get_field(snapshot, "risk_cfg", {}) or {}
        meta = _get_field(snapshot, "meta", {}) or {}

        rg = _get_field(risk_cfg, "regime", None)
        if rg:
            return str(rg)

        rg = _get_field(meta, "regime", None)
        if rg:
            return str(rg)

        m2 = _get_field(risk_cfg, "meta", {}) or {}
        rg = _get_field(m2, "regime", None)
        if rg:
            return str(rg)

        return "unknown"

    def _extract_conf(self, snapshot: Dict[str, Any]) -> float:
        risk_cfg = _get_field(snapshot, "risk_cfg", {}) or {}
        meta = _get_field(snapshot, "meta", {}) or {}

        c = _get_field(risk_cfg, "regime_conf", None)
        if c is None:
            c = _get_field(meta, "regime_conf", None)

        m2 = _get_field(risk_cfg, "meta", {}) or {}
        if c is None:
            c = _get_field(m2, "regime_conf", None)

        return _safe_float(c, 1.0)

    # -----------------------------
    # Main (generic)
    # -----------------------------
    def on_outcome(self, outcome: Dict[str, Any], snapshot: Dict[str, Any]) -> OutcomeUpdateResult:
        # reward extraction
        reward = None
        for k in ("reward", "r", "pnl", "profit"):
            if isinstance(outcome, dict) and k in outcome and outcome[k] is not None:
                reward = outcome[k]
                break
        base_reward = _safe_float(reward, 0.0)

        expert = self._extract_expert(snapshot)
        regime = self._extract_regime(snapshot)
        conf = self._extract_conf(snapshot) if self.use_confidence else 1.0

        shaped = base_reward * conf

        if self.debug:
            try:
                print(f"[OutcomeUpdater] expert={expert!r} regime={regime!r} base_reward={base_reward:.6f} conf={conf:.4f} shaped={shaped:.6f}")
            except Exception:
                pass

        if not expert:
            if self.debug:
                try:
                    print("[OutcomeUpdater] missing expert; snapshot keys:", list(snapshot.keys()))
                    print("[OutcomeUpdater] risk_cfg keys:", list((_get_field(snapshot, 'risk_cfg', {}) or {}).keys()))
                    print("[OutcomeUpdater] meta keys:", list((_get_field(snapshot, 'meta', {}) or {}).keys()))
                except Exception:
                    pass
            return OutcomeUpdateResult(False, "missing_expert", "", regime, shaped)

        ws = self.weight_store
        if ws is None:
            return OutcomeUpdateResult(False, "no_weight_store", expert, regime, shaped)

        if not hasattr(ws, "update"):
            if self.debug:
                print("[OutcomeUpdater] weight_store has no update(); type=", type(ws))
            return OutcomeUpdateResult(False, "no_weight_store_update", expert, regime, shaped)

        try:
            ws.update(
                expert,
                regime,
                shaped,
                lr=self.lr,
                autosave=self.autosave,
                log=self.debug,
                meta={
                    "base_reward": base_reward,
                    "conf": conf,
                    "regime": regime,
                    "expert": expert,
                },
                save_path=self.weights_path,
            )
            return OutcomeUpdateResult(True, "ok", expert, regime, shaped)
        except Exception as e:
            if self.debug:
                try:
                    print("[OutcomeUpdater] update failed:", repr(e))
                except Exception:
                    pass
            return OutcomeUpdateResult(False, f"exception:{type(e).__name__}", expert, regime, shaped)