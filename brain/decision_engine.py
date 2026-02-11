# brain/decision_engine.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from brain.regime_detector import detect_regime

# NOTE:
# - Avoid circular imports: keep these imports local when needed.
# - DecisionEngine must work even if meta layer is absent / optional.


@dataclass
class EngineDecision:
    allow: bool
    score: float
    expert: str = "UNKNOWN_EXPERT"
    meta: Optional[Dict[str, Any]] = None


class DecisionEngine:
    """
    Coordinates:
    - Regime detection
    - ExpertGate picking (uses WeightStore if provided)
    - (Optional) MetaController hooks
    """

    def __init__(
        self,
        risk_engine: Any,
        weight_store: Optional[Any] = None,
        meta_controller: Optional[Any] = None,
        enable_meta: bool = True,
        *,
        epsilon: float = 0.0,
        epsilon_cooldown: int = 0,
        seed: Optional[int] = None,
    ) -> None:
        self.risk_engine = risk_engine
        self.weight_store = weight_store

        # --- gate / registry ---
        self.registry = self._build_registry()
        self.gate = self._build_gate(
            registry=self.registry,
            weight_store=self.weight_store,
            epsilon=epsilon,
            epsilon_cooldown=epsilon_cooldown,
            seed=seed,
        )

        # --- meta (optional) ---
        self.meta = None
        if enable_meta:
            self.meta = self._build_meta(meta_controller)

    # -------------------------
    # Builders (robust)
    # -------------------------
    def _build_registry(self) -> Any:
        """
        Create ExpertRegistry and register experts.
        Keep imports local to avoid circular dependencies.
        """
        from brain.experts.expert_registry import ExpertRegistry

        reg = ExpertRegistry()

        # Register basic experts if available
        try:
            from brain.experts.experts_basic import register_basic_experts  # type: ignore

            register_basic_experts(reg)
        except Exception:
            # Fallback: try importing module side-effects
            try:
                import brain.experts.experts_basic  # noqa: F401
            except Exception:
                pass

        # Register simple experts if available
        try:
            from brain.experts.simple_experts import register_simple_experts  # type: ignore

            register_simple_experts(reg)
        except Exception:
            try:
                import brain.experts.simple_experts  # noqa: F401
            except Exception:
                pass

        return reg

    def _build_gate(
        self,
        registry: Any,
        weight_store: Optional[Any],
        epsilon: float,
        epsilon_cooldown: int,
        seed: Optional[int],
    ) -> Any:
        from brain.experts.expert_gate import ExpertGate

        # Some versions accept rng, some accept seed, some accept weight_store.
        kwargs: Dict[str, Any] = {
            "registry": registry,
            "epsilon": float(epsilon),
            "epsilon_cooldown": int(epsilon_cooldown),
        }
        if weight_store is not None:
            kwargs["weight_store"] = weight_store

        # RNG/seed compatibility
        if seed is not None:
            try:
                import random

                kwargs["rng"] = random.Random(int(seed))
            except Exception:
                pass

        try:
            return ExpertGate(**kwargs)
        except TypeError:
            # fallback: remove unknown keys one by one
            for k in ["weight_store", "rng", "epsilon_cooldown"]:
                if k in kwargs:
                    tmp = dict(kwargs)
                    tmp.pop(k, None)
                    try:
                        return ExpertGate(**tmp)
                    except TypeError:
                        continue
            # last resort
            return ExpertGate(registry=registry)

    def _build_meta(self, meta_controller: Optional[Any]) -> Optional[Any]:
        if meta_controller is not None:
            return meta_controller

        # Try auto-create MetaController if module exists
        try:
            from brain.meta_controller import MetaController  # type: ignore

            try:
                return MetaController()
            except TypeError:
                # some versions might require args; disable if cannot instantiate
                return None
        except Exception:
            return None

    # -------------------------
    # Exploration control
    # -------------------------
    def set_exploration(self, epsilon: float, cooldown: int = 0) -> None:
        """
        Called by ShadowRunner periodically.
        """
        # keep on engine (useful for logging/debug)
        self.epsilon = float(epsilon)
        self.epsilon_cooldown = int(cooldown)

        # pass-through to gate if available
        if hasattr(self.gate, "set_epsilon"):
            try:
                self.gate.set_epsilon(float(epsilon))
            except Exception:
                pass

        # some gate versions expose epsilon_cooldown directly
        try:
            if hasattr(self.gate, "epsilon_cooldown"):
                setattr(self.gate, "epsilon_cooldown", int(cooldown))
        except Exception:
            pass

        # some gate versions have tick/cooldown internal
        # (ShadowRunner should call gate.tick() each step if it exists)

    # -------------------------
    # Core evaluation
    # -------------------------
    def evaluate_trade(self, trade_features: Dict[str, Any]) -> Tuple[bool, float, Dict[str, Any]]:
        """
        Return: (allow, score, risk_cfg)
        """
        candles: List[Any] = trade_features.get("candles") or trade_features.get("window") or []
        context: Dict[str, Any] = detect_regime(candles) if candles is not None else {"regime": "UNKNOWN"}

        # Meta pre-hook (optional)
        if self.meta is not None and hasattr(self.meta, "pre_decision"):
            try:
                self.meta.pre_decision(trade_features=trade_features, context=context)
            except Exception:
                pass

        best, all_decisions = self.gate.pick(trade_features, context)

        # Normalize best decision shape
        if isinstance(best, dict):
            allow = bool(best.get("allow", False))
            score = float(best.get("score", 0.0))
            expert_name = str(best.get("expert", "UNKNOWN_EXPERT"))
            meta = best.get("meta") if isinstance(best.get("meta"), dict) else None
        else:
            allow = bool(getattr(best, "allow", False))
            score = float(getattr(best, "score", 0.0))
            expert_name = str(getattr(best, "expert", "UNKNOWN_EXPERT"))
            meta = getattr(best, "meta", None)
            if meta is not None and not isinstance(meta, dict):
                meta = None

        # Build risk cfg
        risk_cfg: Dict[str, Any] = {
            "expert": expert_name,
            "regime": context.get("regime", "UNKNOWN"),
        }
        if meta:
            risk_cfg.update({"meta": meta})

        # Meta post-hook (optional)
        if self.meta is not None and hasattr(self.meta, "post_decision"):
            try:
                self.meta.post_decision(
                    trade_features=trade_features,
                    context=context,
                    decision={"allow": allow, "score": score, "expert": expert_name, "meta": meta},
                )
            except Exception:
                pass

        return allow, score, risk_cfg
