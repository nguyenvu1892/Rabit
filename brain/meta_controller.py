# brain/meta_controller.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


@dataclass
class RegimeMetaState:
    # EMA of confidence (score) and outcome
    ema_conf: float = 0.0
    ema_reward: float = 0.0  # +1/-1-ish reward
    ema_winrate: float = 0.5

    # Counts
    n: int = 0
    n_allow: int = 0
    n_deny: int = 0

    # Adaptive threshold (what we want to learn)
    score_threshold: float = 0.05  # default tiny > 0

    # Optional exploration by regime
    epsilon: float = 0.05

    def update_decision(self, allow: bool, conf: float, alpha: float = 0.05) -> None:
        self.n += 1
        if allow:
            self.n_allow += 1
        else:
            self.n_deny += 1

        conf = max(0.0, min(1.0, _safe_float(conf, 0.0)))
        self.ema_conf = (1 - alpha) * self.ema_conf + alpha * conf

    def update_outcome(self, reward: float, alpha: float = 0.05) -> None:
        # reward expected in [-1, +1]
        r = max(-1.0, min(1.0, _safe_float(reward, 0.0)))
        self.ema_reward = (1 - alpha) * self.ema_reward + alpha * r

        win = 1.0 if r > 0 else 0.0 if r < 0 else 0.5
        self.ema_winrate = (1 - alpha) * self.ema_winrate + alpha * win


class MetaController:
    """
    5.1.8 Meta Intelligence:
    - Track decision confidence by regime
    - Track outcome reward by regime
    - Adapt per-regime score_threshold (gating)
    - Optionally adapt epsilon by regime

    IMPORTANT:
    - Keep backward compatibility: all methods are best-effort and safe.
    """

    def __init__(
        self,
        base_threshold: float = 0.05,
        min_threshold: float = 0.0,
        max_threshold: float = 1.0,
        base_epsilon: float = 0.05,
        alpha: float = 0.05,
        **kwargs,
    ) -> None:
        self.base_threshold = float(base_threshold)
        self.min_threshold = float(min_threshold)
        self.max_threshold = float(max_threshold)
        self.base_epsilon = float(base_epsilon)
        self.alpha = float(alpha)

        self.by_regime: Dict[str, RegimeMetaState] = {}

    def _state(self, regime: Optional[str]) -> RegimeMetaState:
        r = str(regime or "unknown")
        if r not in self.by_regime:
            self.by_regime[r] = RegimeMetaState(
                score_threshold=self.base_threshold,
                epsilon=self.base_epsilon,
            )
        return self.by_regime[r]

    # -----------------------
    # Read policy
    # -----------------------
    def get_policy(self, regime: Optional[str]) -> Dict[str, float]:
        st = self._state(regime)
        return {
            "score_threshold": float(st.score_threshold),
            "epsilon": float(st.epsilon),
            "ema_conf": float(st.ema_conf),
            "ema_reward": float(st.ema_reward),
            "ema_winrate": float(st.ema_winrate),
            "n": int(st.n),
            "n_allow": int(st.n_allow),
            "n_deny": int(st.n_deny),
        }

    def get_score_threshold(self, regime: Optional[str]) -> float:
        return float(self._state(regime).score_threshold)

    def get_epsilon(self, regime: Optional[str]) -> float:
        return float(self._state(regime).epsilon)

    # -----------------------
    # Update from decision trace
    # -----------------------
    def on_decision(self, regime: Optional[str], allow: bool, confidence: float) -> None:
        st = self._state(regime)
        st.update_decision(bool(allow), _safe_float(confidence, 0.0), alpha=self.alpha)

        # Light shaping: if we deny too much but confidence EMA is healthy -> lower threshold
        # if allow too much but reward EMA negative -> raise threshold
        self._adapt_threshold(regime)

    # -----------------------
    # Update from realized outcome
    # -----------------------
    def on_outcome(self, regime: Optional[str], reward: float) -> None:
        st = self._state(regime)
        st.update_outcome(_safe_float(reward, 0.0), alpha=self.alpha)

        self._adapt_threshold(regime)
        self._adapt_epsilon(regime)

    # -----------------------
    # Internal adaptation rules (simple & stable)
    # -----------------------
    def _adapt_threshold(self, regime: Optional[str]) -> None:
        st = self._state(regime)

        # signal
        conf = float(st.ema_conf)
        rew = float(st.ema_reward)
        wr = float(st.ema_winrate)

        # small step
        step = 0.01

        # If reward is negative, become stricter
        if rew < -0.10:
            st.score_threshold += step
        # If reward positive and confidence decent, relax a bit
        elif rew > 0.10 and conf > 0.05:
            st.score_threshold -= step
        # If deny ratio huge while confidence is not terrible => relax
        if st.n >= 50:
            deny_ratio = st.n_deny / max(1, st.n)
            if deny_ratio > 0.85 and conf > 0.03:
                st.score_threshold -= step

        # Clamp
        st.score_threshold = max(self.min_threshold, min(self.max_threshold, st.score_threshold))

        # If winrate is extremely low, clamp higher floor a bit (avoid overtrading)
        if st.n >= 50 and wr < 0.35:
            st.score_threshold = max(st.score_threshold, 0.08)

    def _adapt_epsilon(self, regime: Optional[str]) -> None:
        st = self._state(regime)
        # If doing well -> explore less; if doing bad -> explore more (bounded)
        wr = float(st.ema_winrate)
        step = 0.01

        if st.n < 50:
            return

        if wr > 0.60:
            st.epsilon = max(0.0, st.epsilon - step)
        elif wr < 0.45:
            st.epsilon = min(0.20, st.epsilon + step)

    def snapshot(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for r, st in self.by_regime.items():
            out[r] = {
                "ema_conf": st.ema_conf,
                "ema_reward": st.ema_reward,
                "ema_winrate": st.ema_winrate,
                "n": st.n,
                "n_allow": st.n_allow,
                "n_deny": st.n_deny,
                "score_threshold": st.score_threshold,
                "epsilon": st.epsilon,
            }
        return out
