# tools/shadow_run.py
from __future__ import annotations

import argparse
import inspect
from typing import Any, Dict, Optional

from brain.decision_engine import DecisionEngine
from brain.weight_store import WeightStore
from observer.eval_reporter import EvalReporter
from observer.outcome_updater import OutcomeUpdater
from sim.shadow_runner import ShadowRunner

# Optional imports (best effort)
try:
    from brain.trade_memory import TradeMemory
except Exception:
    TradeMemory = None  # type: ignore


def _filter_kwargs_for_callable(fn: Any, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Return kwargs restricted to accepted parameters of fn (unless fn has **kwargs)."""
    try:
        sig = inspect.signature(fn)
        params = sig.parameters
        if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
            return dict(kwargs)
        allowed = set(params.keys())
        return {k: v for k, v in kwargs.items() if k in allowed}
    except Exception:
        return dict(kwargs)


def _call_shadowrunner_run_compat(runner: Any, candles: Any, run_kwargs: Dict[str, Any]) -> Any:
    """
    Compat wrapper for ShadowRunner.run across versions:

    vA: run(self, candles, lookback=..., max_steps=..., journal=..., train=..., ...)
    vB (5.1.8): run(self, max_steps=..., journal=..., train=...)  # candles come from runner/broker internally
    """
    run_fn = getattr(runner, "run", None)
    if run_fn is None:
        raise RuntimeError("ShadowRunner has no run()")

    try:
        sig = inspect.signature(run_fn)
        params = list(sig.parameters.values())
        # remove 'self'
        if params and params[0].name == "self":
            params = params[1:]
        names = [p.name for p in params]

        # If run accepts candles/rows/data -> old style
        if any(n in ("candles", "rows", "data") for n in names):
            filtered = _filter_kwargs_for_callable(run_fn, run_kwargs)

            # Prefer keyword if supported
            if "candles" in names:
                return run_fn(candles=candles, **filtered)
            if "rows" in names:
                return run_fn(rows=candles, **filtered)
            if "data" in names:
                return run_fn(data=candles, **filtered)

            # Fallback positional first arg (candles) ONLY if first param is not max_steps
            # to avoid "candles list -> max_steps"
            if names and names[0] not in ("max_steps", "steps"):
                return run_fn(candles, **filtered)

        # Otherwise new style (5.1.8): no candles param
        filtered = _filter_kwargs_for_callable(run_fn, run_kwargs)
        return run_fn(**filtered)

    except TypeError:
        # Last-resort attempts (do NOT pass candles as positional, it can break max_steps)
        filtered = _filter_kwargs_for_callable(run_fn, run_kwargs)
        return run_fn(**filtered)


def build_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--limit", type=int, default=5000)
    p.add_argument("--lookback", type=int, default=300)
    p.add_argument("--max-steps", type=int, default=2000)
    p.add_argument("--horizon", type=int, default=30)
    p.add_argument("--train", action="store_true")
    p.add_argument("--epsilon", type=float, default=0.05)
    p.add_argument("--epsilon-cooldown", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--journal", default="data/journal_train.jsonl")
    p.add_argument("--weights", default="data/weights.json")
    p.add_argument("--reports", default="data/reports")
    return p.parse_args()


def main() -> None:
    args = build_args()

    # --- weight store ---
    weight_store = WeightStore(path=args.weights)

    # --- decision engine ---
    # NOTE: your DecisionEngine signature may require risk_engine; keep explicit
    # If your DecisionEngine needs other deps, add them here (do not delete existing)
    de = DecisionEngine(
        risk_engine=None,           # keep compat; your engine may ignore
        weight_store=weight_store,
    )

    # --- optional trade memory ---
    trade_memory = None
    if TradeMemory is not None:
        try:
            trade_memory = TradeMemory()
        except Exception:
            trade_memory = None

    # --- outcome updater (best-effort compat) ---
    outcome_updater = OutcomeUpdater(
        learner=None,
        trade_memory=trade_memory,
        weight_store=weight_store,
        meta_controller=getattr(de, "meta_controller", None),
        weights_path=args.weights,
        autosave=True,
    )

    # --- eval reporter ---
    reporter = EvalReporter(out_dir=args.reports)

    # --- ShadowRunner init (compat by filtering kwargs) ---
    # ShadowRunner in 5.1.8 usually loads candles from broker internally.
    runner_init_kwargs = dict(
        decision_engine=de,
        csv=args.csv,
        limit=args.limit,
        lookback=args.lookback,
        horizon=args.horizon,
        outcome_updater=outcome_updater,
        reporter=reporter,
        seed=args.seed,
        epsilon=args.epsilon,
        epsilon_cooldown=args.epsilon_cooldown,
    )
    init_fn = ShadowRunner.__init__
    runner_init_kwargs = _filter_kwargs_for_callable(init_fn, runner_init_kwargs)

    runner = ShadowRunner(**runner_init_kwargs)

    # --- run compat ---
    run_kwargs = dict(
        max_steps=args.max_steps,
        journal=args.journal,
        train=args.train,
    )

    # Some older versions require candles from loader; 5.1.8 doesn't.
    candles = None
    stats = _call_shadowrunner_run_compat(runner, candles, run_kwargs)

    # --- summary print ---
    try:
        ws = weight_store.to_dict() if hasattr(weight_store, "to_dict") else {}
        print("\n=== SHADOW RUN DONE ===")
        print("WEIGHT SIZE:", len(ws) if isinstance(ws, dict) else 0)
    except Exception:
        print("\n=== SHADOW RUN DONE ===")

    # stats can be dict or dataclass-like
    if isinstance(stats, dict):
        for k in ["steps", "decisions", "allow", "deny", "errors", "outcomes", "wins", "losses", "total_pnl", "forced_entries", "regime_breakdown"]:
            if k in stats:
                print(f"{k}: {stats[k]}")
    else:
        # best-effort attributes
        for k in ["steps", "decisions", "allow", "deny", "errors", "outcomes", "wins", "losses", "total_pnl", "forced_entries", "regime_breakdown"]:
            if hasattr(stats, k):
                print(f"{k}: {getattr(stats, k)}")

    try:
        reporter.finalize(stats)
    except Exception as e:
        print("[shadow_run] WARN: reporter finalize failed:", repr(e))


if __name__ == "__main__":
    main()
