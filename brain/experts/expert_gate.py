# brain/experts/expert_gate.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from brain.experts.expert_base import ExpertDecision


@dataclass
class GateOutput:
    allow: bool
    score: float
    risk_cfg: Dict[str, Any]
    expert: str
    meta: Dict[str, Any]


class ExpertGate:
    """
    Select best expert decision from registry.
    Returns (allow, score, risk_cfg).
    """

    def __init__(self, registry: Any, weight_store: Optional[Any] = None, debug: bool = False):
        self.registry = registry
        self.weight_store = weight_store
        self.debug = debug

    def _get_experts(self) -> List[Any]:
        xs: List[Any] = []
        try:
            if hasattr(self.registry, "get_all"):
                tmp = self.registry.get_all()
                if isinstance(tmp, list):
                    xs = tmp
        except Exception:
            xs = []

        # ---- COMPAT BLOCK: never allow empty registry to brick the system ----
        # If registry is empty, we still want the pipeline to run deterministically.
        # Try fallback import DEFAULT_EXPERTS (baseline set).
        if not xs:
            try:
                from brain.experts.experts_basic import DEFAULT_EXPERTS  # type: ignore
                if isinstance(DEFAULT_EXPERTS, list) and DEFAULT_EXPERTS:
                    xs = DEFAULT_EXPERTS
            except Exception:
                xs = []
        # ---------------------------------------------------------------------

        return xs

    def decide(self, features: Dict[str, Any]) -> GateOutput:
        experts = self._get_experts()

        best: Optional[ExpertDecision] = None
        best_raw: float = -1e18
        best_name: str = "UNKNOWN"
        best_meta: Dict[str, Any] = {}

        for e in experts:
            try:
                # ---- COMPAT: support both decide() and evaluate() on expert ----
                if hasattr(e, "decide"):
                    dec = e.decide(features)
                elif hasattr(e, "evaluate"):
                    dec = e.evaluate(features)
                else:
                    dec = None
                # ----------------------------------------------------------------

                d = ExpertDecision.coerce(dec)
                name = getattr(e, "name", None) or getattr(d, "expert", None) or "UNKNOWN"
                d.expert = str(name)

                raw_score = float(getattr(d, "score", 0.0) or 0.0)

                # apply weight if available
                w = 1.0
                try:
                    if self.weight_store is not None:
                        # weight key convention: EXPERT|REGIME
                        regime = (
                            features.get("market_regime")
                            or features.get("regime")
                            or features.get("state")
                            or "unknown"
                        )
                        key = f"{d.expert}|{regime}"
                        w = float(self.weight_store.get(key, 1.0))
                except Exception:
                    w = 1.0

                weighted = raw_score * w

                if weighted > best_raw:
                    best_raw = weighted
                    best = d
                    best_name = d.expert
                    best_meta = d.meta if isinstance(d.meta, dict) else {}
            except Exception:
                continue

        if best is None:
            # keep deterministic fallback
            best = ExpertDecision(allow=False, score=0.0, expert="UNKNOWN", meta={})

        allow = bool(best.allow)
        score = float(best_raw if best_raw > -1e18 else (best.score or 0.0))

        # risk_cfg contract: dict
        risk_cfg: Dict[str, Any] = {}
        try:
            if isinstance(getattr(best, "risk_cfg", None), dict):
                risk_cfg.update(best.risk_cfg)
        except Exception:
            pass

        # attach meta for downstream
        meta: Dict[str, Any] = {}
        meta.update(best_meta or {})
        meta.setdefault("expert", best_name)
        meta.setdefault("raw_score", float(getattr(best, "score", 0.0) or 0.0))
        meta.setdefault("weighted_score", float(score))

        risk_cfg.setdefault("meta", {})
        if isinstance(risk_cfg["meta"], dict):
            risk_cfg["meta"].update(meta)

        return GateOutput(allow=allow, score=score, risk_cfg=risk_cfg, expert=best_name, meta=meta)

    # ---- COMPAT BLOCK --------------------------------------------------------
    # Older DecisionEngine called gate.evaluate(features)
    def evaluate(self, features: Dict[str, Any]) -> Tuple[bool, float, Dict[str, Any]]:
        out = self.decide(features)
        return out.allow, out.score, out.risk_cfg
    # -------------------------------------------------------------------------