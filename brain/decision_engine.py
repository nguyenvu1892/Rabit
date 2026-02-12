# brain/decision_engine.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from brain.experts.expert_gate import ExpertGate

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

        # store exploration params (debug)
        self.epsilon = float(epsilon)
        self.epsilon_cooldown = int(epsilon_cooldown)

    # -------------------------
    # Builders (robust)
    # -------------------------
    def _build_registry(self) -> Any:
        """
        Create ExpertRegistry and register experts.

        Robust strategy:
        - import experts modules
        - auto-discover register_* functions
        - accept common API patterns
        - fallback baseline expert if registry stays empty
        """
        from brain.experts.expert_registry import ExpertRegistry
        from brain.experts.experts_basic import DEFAULT_EXPERTS
        reg = ExpertRegistry()
        def _reg_add(expert_obj: Any) -> bool:
            for meth in ("register", "add", "add_expert", "register_expert"):
                if hasattr(reg, meth):
                    try:
                        getattr(reg, meth)(expert_obj)
                        return True
                    except Exception:
                        continue
            return False

        def _get_all() -> List[Any]:
            for meth in ("get_all", "all", "list", "values"):
                if hasattr(reg, meth):
                    try:
                        xs = getattr(reg, meth)()
                        return list(xs) if xs is not None else []
                    except Exception:
                        pass
            return []

        def _maybe_register_module(mod: Any) -> None:
            # known names
            for fn_name in (
                "register",
                "register_experts",
                "register_basic_experts",
                "register_simple_experts",
            ):
                fn = getattr(mod, fn_name, None)
                if callable(fn):
                    try:
                        fn(reg)
                    except Exception:
                        pass

            # any register_* (1 arg)
            for name in dir(mod):
                if not name.startswith("register_"):
                    continue
                fn = getattr(mod, name, None)
                if not callable(fn):
                    continue
                try:
                    fn(reg)
                except TypeError:
                    continue
                except Exception:
                    continue

            # EXPERTS iterable
            exps = getattr(mod, "EXPERTS", None)
            if isinstance(exps, (list, tuple)):
                for e in exps:
                    _reg_add(e)

            # builder funcs returning list
            for fn_name in ("build_experts", "make_experts", "create_experts"):
                fn = getattr(mod, fn_name, None)
                if callable(fn):
                    try:
                        xs = fn()
                        if isinstance(xs, (list, tuple)):
                            for e in xs:
                                _reg_add(e)
                    except Exception:
                        pass

        # Try import modules
        for mod_path in (
            "brain.experts.experts_basic",
            "brain.experts.simple_experts",
            "brain.experts.experts",
        ):
            try:
                mod = __import__(mod_path, fromlist=["*"])
                _maybe_register_module(mod)
            except Exception:
                continue

        # Fallback baseline expert
        if len(_get_all()) == 0:
            from brain.experts.expert_base import ExpertDecision

            class BaselineExpert:
                name = "BASELINE"

                def decide(self, trade_features: Dict[str, Any], context: Dict[str, Any]) -> ExpertDecision:
                    return ExpertDecision(
                        expert=self.name,
                        allow=True,
                        score=0.01,
                        meta={"fallback": True, "reason": "registry_empty"},
                    )

            _reg_add(BaselineExpert())

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

        # Try create with richest signature first
        try:
            return ExpertGate(**kwargs)
        except TypeError:
            for k in ("weight_store", "rng", "epsilon_cooldown"):
                if k in kwargs:
                    tmp = dict(kwargs)
                    tmp.pop(k, None)
                    try:
                        return ExpertGate(**tmp)
                    except TypeError:
                        continue
            return ExpertGate(registry=registry)

    def _build_meta(self, meta_controller: Optional[Any]) -> Optional[Any]:
        if meta_controller is not None:
            return meta_controller
        try:
            from brain.meta_controller import MetaController  # type: ignore
            try:
                return MetaController()
            except TypeError:
                return None
        except Exception:
            return None

    # -------------------------
    # Exploration control
    # -------------------------
    def set_exploration(self, epsilon: float, cooldown: int = 0) -> None:
        self.epsilon = float(epsilon)
        self.epsilon_cooldown = int(cooldown)

        if hasattr(self.gate, "set_epsilon"):
            try:
                self.gate.set_epsilon(float(epsilon))
            except Exception:
                pass
        try:
            if hasattr(self.gate, "epsilon_cooldown"):
                setattr(self.gate, "epsilon_cooldown", int(cooldown))
        except Exception:
            pass

    # -------------------------
    # Core evaluation
    # -------------------------
    def evaluate_trade(self, trade_features: Dict[str, Any]) -> Tuple[bool, float, Dict[str, Any]]:
        """
        Return: (allow, score, risk_cfg)
        risk_cfg must include regime + expert so OutcomeUpdater can credit-assign.
        """
        from brain.regime_detector import detect_regime

        candles: List[Any] = trade_features.get("candles") or trade_features.get("window") or []
        context: Dict[str, Any] = detect_regime(candles) if candles is not None else {"regime": "UNKNOWN", "confidence": 0.0}

        # normalize keys (regime-first 5.1.3)
        regime = str(context.get("regime", "UNKNOWN"))
        regime_conf = float(context.get("confidence", 0.0))
        context["regime"] = regime
        context["regime_conf"] = regime_conf  # canonical key for downstream
        context.setdefault("vol", 0.0)
        context.setdefault("slope", 0.0)

        # Meta pre-hook
        if self.meta is not None and hasattr(self.meta, "pre_decision"):
            try:
                self.meta.pre_decision(trade_features=trade_features, context=context)
            except Exception:
                pass

        best, _all_decisions = self.gate.pick(trade_features, context)
        print("DEBUG BEST:", best)
        print("DEBUG BEST_SCORE:", getattr(best, "score", None))
        print("DEBUG BEST_EXPERT:", getattr(best, "expert", None))
        print("DEBUG BEST_META:", getattr(best, "meta", None))
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

        # risk_cfg is what OutcomeUpdater uses later
        regime = str(context.get("regime", "UNKNOWN"))
        risk_cfg: Dict[str, Any] = {
            "expert": expert_name,
            "regime": regime,
            "regime_conf": regime_conf,
        }
        if meta:
            risk_cfg["meta"] = meta

        meta = meta or {}
        meta["regime"] = regime
        meta["expert"] = expert_name
                # Meta post-hook
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
    