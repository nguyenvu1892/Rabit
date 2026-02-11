# observer/outcome_updater.py
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Optional

from brain.trade_memory import TradeMemory

if TYPE_CHECKING:
    from brain.reinforcement_learner import ReinforcementLearner
    from brain.weight_store import WeightStore


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _sign(x: float) -> float:
    if x > 0:
        return 1.0
    if x < 0:
        return -1.0
    return 0.0


@dataclass
class OUConfig:
    # base update strength
    lr: float = 0.05

    # stabilization
    min_w: float = 0.2
    max_w: float = 5.0
    decay_rate: float = 0.002
    decay_target: float = 1.0
    normalize_target_mean: float = 1.0

    # periodic jobs
    save_every: int = 50
    decay_every: int = 50
    normalize_every: int = 200
    log_every: int = 200

    # safety
    forced_penalty: float = 0.15  # reduce learning impact if forced trade
    pnl_clip: float = 2.0         # clip pnl signal for stability
    atr_scale: float = 0.0        # if >0, scale by 1/(1+atr*atr_scale)


class OutcomeUpdater:
    """
    Updates:
      - ReinforcementLearner (existing behavior)
      - TradeMemory
      - WeightStore (Smart bucket intelligence)

    Smart buckets updated:
      1) expert bucket:   weights["expert"][expert]
      2) regime bucket:   weights[f"regime:{regime}"][expert]
      3) tag buckets:     weights["session"][session], ["pattern"][pattern], ["structure"][structure], ["trend"][trend]
    """

    def __init__(
        self,
        learner: Optional["ReinforcementLearner"],
        trade_memory: TradeMemory,
        weight_store: Optional["WeightStore"] = None,
        weights_path: Optional[str] = None,
        autosave: bool = True,
        cfg: Optional[OUConfig] = None,
    ) -> None:
        self.learner = learner
        self.trade_memory = trade_memory
        self.weight_store = weight_store
        self.weights_path = weights_path
        self.autosave = bool(autosave)
        self.cfg = cfg or OUConfig()

        self._n_updates = 0

        # allow runtime override of clamp range in WeightStore
        if self.weight_store is not None:
            try:
                self.weight_store.min_w = float(self.cfg.min_w)
                self.weight_store.max_w = float(self.cfg.max_w)
            except Exception:
                pass

    def on_outcome(self, outcome: Dict[str, Any]) -> None:
        """
        outcome expected to contain (best effort):
          - pnl (float)
          - win (bool) or win-like
          - forced (bool)
          - meta: { expert, regime, session, pattern, structure, trend, atr? ... }
        """
        self._n_updates += 1

        # 1) trade memory
        try:
            self.trade_memory.add_outcome(outcome)
        except Exception:
            pass

        # 2) learner update (keep existing pipeline)
        if self.learner is not None:
            try:
                self.learner.on_outcome(outcome)
            except Exception:
                pass

        # 3) weight learning
        if self.weight_store is not None:
            self._update_weights(outcome)

        # periodic maintenance
        if self.weight_store is not None:
            self._maintenance()

    # ---------------------------
    # Smart bucket intelligence
    # ---------------------------
    def _update_weights(self, outcome: Dict[str, Any]) -> None:
        meta = outcome.get("meta") or {}
        if not isinstance(meta, dict):
            meta = {}

        expert = str(meta.get("expert") or outcome.get("expert") or "").strip()
        regime = str(meta.get("regime") or outcome.get("regime") or "").strip()

        # tag buckets (these are NOT experts; these are labels)
        session = meta.get("session")
        pattern = meta.get("pattern")
        structure = meta.get("structure")
        trend = meta.get("trend")

        pnl = _safe_float(outcome.get("pnl"), 0.0)
        win_raw = outcome.get("win", None)
        win = bool(win_raw) if win_raw is not None else (pnl > 0)

        forced = bool(outcome.get("forced") or meta.get("forced") or False)

        # optional ATR scaling
        atr = _safe_float(meta.get("atr", outcome.get("atr", 0.0)), 0.0)

        # --- compute stable delta ---
        # signal from pnl + win
        pnl_sig = _safe_float(pnl, 0.0)
        # clip extreme
        if pnl_sig > self.cfg.pnl_clip:
            pnl_sig = self.cfg.pnl_clip
        if pnl_sig < -self.cfg.pnl_clip:
            pnl_sig = -self.cfg.pnl_clip

        # combine: win gives direction if pnl tiny
        direction = _sign(pnl_sig)
        if direction == 0.0:
            direction = 1.0 if win else -1.0

        mag = abs(pnl_sig) / max(1e-9, self.cfg.pnl_clip)  # 0..1
        base = self.cfg.lr * (0.35 + 0.65 * mag) * direction

        if forced:
            base *= (1.0 - float(self.cfg.forced_penalty))

        if self.cfg.atr_scale and atr > 0:
            base *= 1.0 / (1.0 + atr * float(self.cfg.atr_scale))

        # --- apply to buckets ---
        ws = self.weight_store
        assert ws is not None

        # expert bucket
        if expert:
            ws.bump(key=expert, bucket="expert", delta=base, clamp=True)

        # regime-specific bucket for expert
        if expert and regime:
            ws.bump(key=expert, bucket=f"regime:{regime}", delta=base, clamp=True)

        # tag buckets (update tag weights with smaller power so they don’t explode)
        tag_delta = base * 0.5

        if session:
            ws.bump(key=str(session), bucket="session", delta=tag_delta, clamp=True)
        if pattern:
            ws.bump(key=str(pattern), bucket="pattern", delta=tag_delta, clamp=True)
        if structure:
            ws.bump(key=str(structure), bucket="structure", delta=tag_delta, clamp=True)
        if trend:
            ws.bump(key=str(trend), bucket="trend", delta=tag_delta, clamp=True)

    def _maintenance(self) -> None:
        ws = self.weight_store
        if ws is None:
            return

        n = self._n_updates
        cfg = self.cfg

        # decay toward 1.0 (anti-overfit)
        if cfg.decay_every > 0 and (n % cfg.decay_every == 0):
            # decay all buckets
            try:
                for b in list(ws.weights.keys()):
                    ws.decay_bucket_toward(b, target=cfg.decay_target, rate=cfg.decay_rate, clamp=True)
            except Exception:
                pass

        # normalize mean bucket -> 1.0 (optional, less frequent)
        if cfg.normalize_every > 0 and (n % cfg.normalize_every == 0):
            try:
                for b in list(ws.weights.keys()):
                    ws.normalize_bucket_mean(b, target_mean=cfg.normalize_target_mean)
            except Exception:
                pass

        # log summary
        if cfg.log_every > 0 and (n % cfg.log_every == 0):
            try:
                # print top/bottom for key buckets
                for b in ["expert", "pattern", "structure", "trend", "session"]:
                    if b in ws.weights and ws.weights[b]:
                        top = ws.topk(b, 3)
                        bot = ws.bottomk(b, 3)
                        print(f"[WeightStore] bucket={b} top={top} bottom={bot}")
            except Exception:
                pass

        # periodic save
        if self.autosave and cfg.save_every > 0 and (n % cfg.save_every == 0):
            try:
                if self.weights_path:
                    ws.save_json(self.weights_path)
                elif ws.path:
                    ws.save_json(ws.path)
            except Exception:
                pass
