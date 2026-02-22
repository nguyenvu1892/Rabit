# brain/experts/expert_base.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


# ============================================================
# Legacy / compat-safe helpers (DO NOT REMOVE)
# ============================================================

def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


# ============================================================
# LEGACY DATACLASSES (kept for backward compatibility)
# - We keep these around to avoid breaking older imports.
# ============================================================

@dataclass
class LegacyExpertDecision:
    """
    Legacy schema that some modules may still import/expect.
    """
    expert: str = "UNKNOWN"
    allow: bool = True
    score: float = 0.0
    action: str = "hold"
    meta: Dict[str, Any] = field(default_factory=dict)


# ============================================================
# CURRENT ExpertDecision (with compat __init__)
# ============================================================

@dataclass
class ExpertDecision:
    """
    Canonical decision object across the project.

    Compat rules:
    - accept risk_cfg=... (old name) and store into meta["risk_cfg"]
    - accept risk=... or risk_config=... as aliases (store into meta)
    - accept any extra kwargs without crashing (store into meta["_extra"])
    """
    expert: str = "UNKNOWN"
    allow: bool = True
    score: float = 0.0
    action: str = "hold"
    meta: Dict[str, Any] = field(default_factory=dict)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        # Support positional dataclass-like init (rare)
        if args:
            # Fallback: map positional in the canonical order
            # (expert, allow, score, action, meta)
            expert = args[0] if len(args) > 0 else kwargs.pop("expert", "UNKNOWN")
            allow = args[1] if len(args) > 1 else kwargs.pop("allow", True)
            score = args[2] if len(args) > 2 else kwargs.pop("score", 0.0)
            action = args[3] if len(args) > 3 else kwargs.pop("action", "hold")
            meta = args[4] if len(args) > 4 else kwargs.pop("meta", None)
            kwargs.setdefault("expert", expert)
            kwargs.setdefault("allow", allow)
            kwargs.setdefault("score", score)
            kwargs.setdefault("action", action)
            if meta is not None:
                kwargs.setdefault("meta", meta)

        # ---- Known fields ----
        expert = kwargs.pop("expert", "UNKNOWN")
        allow = kwargs.pop("allow", True)
        score = kwargs.pop("score", 0.0)
        action = kwargs.pop("action", "hold")
        meta = kwargs.pop("meta", None)
        if not isinstance(meta, dict):
            meta = {} if meta is None else {"_meta": meta}

        # ---- Compat aliases ----
        # Old naming: risk_cfg
        if "risk_cfg" in kwargs:
            meta["risk_cfg"] = kwargs.pop("risk_cfg")
        if "risk" in kwargs:
            meta["risk"] = kwargs.pop("risk")
        if "risk_config" in kwargs:
            meta["risk_config"] = kwargs.pop("risk_config")

        # Keep any extra fields so we don't lose info
        if kwargs:
            meta.setdefault("_extra", {})
            try:
                meta["_extra"].update(kwargs)
            except Exception:
                meta["_extra"] = {"_raw": str(kwargs)}

        # Assign
        self.expert = str(expert) if expert is not None else "UNKNOWN"
        self.allow = bool(allow)
        self.score = _safe_float(score, 0.0)
        self.action = str(action) if action is not None else "hold"
        self.meta = meta


# ============================================================
# Expert base class (stable)
# ============================================================

class ExpertBase:
    name: str = "BASE"

    def decide(self, features: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> ExpertDecision:
        # Default safe: allow True with tiny score (baseline-like)
        return ExpertDecision(expert=self.name, allow=True, score=0.0001, action="hold", meta={"reason": "expert_base_default"})


# Backward-compatible alias names (some modules import these)
BaseExpert = ExpertBase