# brain/experts/expert_gate.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from brain.experts.expert_base import ExpertDecision


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _ensure_dict(maybe: Any) -> Dict[str, Any]:
    return maybe if isinstance(maybe, dict) else {}


def _coerce_decision(raw: Any, fallback_expert: str) -> ExpertDecision:
    """
    Compat: accept various return formats from experts:
      - ExpertDecision
      - dict {allow, score, risk_cfg, meta}
      - tuple/list (allow, score, risk_cfg)
    """
    if isinstance(raw, ExpertDecision):
        return raw

    # dict-form
    if isinstance(raw, dict):
        allow = bool(raw.get("allow", False))
        score = _safe_float(raw.get("score", 0.0), 0.0)
        risk_cfg = _ensure_dict(raw.get("risk_cfg"))
        meta = _ensure_dict(raw.get("meta"))
        return ExpertDecision(
            expert=fallback_expert,
            allow=allow,
            score=score,
            risk_cfg=risk_cfg,
            meta=meta,
        )

    # tuple/list form
    if isinstance(raw, (tuple, list)) and len(raw) >= 2:
        allow = bool(raw[0])
        score = _safe_float(raw[1], 0.0)
        risk_cfg = _ensure_dict(raw[2]) if len(raw) >= 3 else {}
        return ExpertDecision(
            expert=fallback_expert,
            allow=allow,
            score=score,
            risk_cfg=risk_cfg,
            meta={},
        )

    # fallback
    return ExpertDecision(expert=fallback_expert, allow=False, score=0.0, risk_cfg={}, meta={})


@dataclass
class ExpertGate:
    registry: Any
    weight_store: Any = None
    debug: bool = False

    def _get_all(self) -> List[Any]:
        # compat for registry.get_all()
        if hasattr(self.registry, "get_all"):
            try:
                xs = self.registry.get_all()
                return list(xs) if xs is not None else []
            except Exception:
                return []
        return []

    def pick(
        self,
        features: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[bool, float, Dict[str, Any]]:
        """
        Returns:
          (allow, score, risk_cfg)
        """
        ctx = context or {}
        regime = ctx.get("regime") or ctx.get("market_regime") or ctx.get("state") or "unknown"
        regime_conf = _safe_float(ctx.get("regime_conf") or ctx.get("confidence") or 0.0, 0.0)

        experts = self._get_all()
        best: Optional[ExpertDecision] = None

        for exp in experts:
            name = getattr(exp, "name", None) or getattr(exp, "__name__", None) or exp.__class__.__name__
            try:
                raw = exp.decide(features, ctx) if hasattr(exp, "decide") else exp(features, ctx)
            except Exception:
                raw = None

            dec = _coerce_decision(raw, fallback_expert=str(name))

            # ensure expert name is set
            if not getattr(dec, "expert", None):
                dec.expert = str(name)

            # choose by score
            if best is None or float(dec.score) > float(best.score):
                best = dec

        if best is None:
            # hard fallback
            best = ExpertDecision(expert="FALLBACK", allow=False, score=0.0, risk_cfg={}, meta={})

        # weight by expert|regime if available
        w = 1.0
        if self.weight_store is not None and hasattr(self.weight_store, "get_expert_regime_weight"):
            try:
                w = float(self.weight_store.get_expert_regime_weight(best.expert, str(regime)))
            except Exception:
                w = 1.0

        raw_score = _safe_float(best.score, 0.0)
        score = raw_score * w

        # attach meta (schema-safe)
        meta = _ensure_dict(getattr(best, "meta", None))
        meta["expert"] = best.expert
        meta["regime"] = str(regime)
        meta["regime_conf"] = float(regime_conf)
        meta["w"] = float(w)
        meta["raw_score"] = float(raw_score)

        risk_cfg = _ensure_dict(getattr(best, "risk_cfg", None))
        risk_cfg["meta"] = meta  # keep compat: risk_cfg carries meta

        if self.debug:
            try:
                print(
                    f"[ExpertGate] best={best.expert} regime={regime} "
                    f"allow={best.allow} raw={raw_score:.4f} w={w:.4f} score={score:.4f}"
                )
            except Exception:
                pass

        return bool(best.allow), float(score), risk_cfg