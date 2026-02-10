# persistence/state_manager.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from persistence.state_bundle import CoreStateBundle
from persistence.state_store import CoreStateStore


@dataclass
class CoreStateManager:
    store: CoreStateStore
    run_id: str
    strategy_hash: str

    def save(self, rl, trade_memory: Any, session_guard, lifecycle=None) -> None:
        ext = lifecycle.get_state() if (lifecycle is not None and hasattr(lifecycle, "get_state")) else {}

        bundle = CoreStateBundle(
            run_id=self.run_id,
            strategy_hash=self.strategy_hash,
            rl_weights=rl.get_state(),
            trade_memory=getattr(trade_memory, "memory", trade_memory),
            session_guard_state=session_guard.get_state(),
            extended_state=ext,
        )
        self.store.save(bundle)

    def load_into(self, rl, trade_memory: Any, session_guard, lifecycle=None) -> bool:
        bundle = self.store.load()
        if bundle is None:
            return False

        rl.set_state(bundle.rl_weights)

        # restore trade memory
        if hasattr(trade_memory, "memory"):
            trade_memory.memory = bundle.trade_memory

        session_guard.set_state(bundle.session_guard_state)

        # restore lifecycle extended state
        if lifecycle is not None and hasattr(lifecycle, "set_state"):
            lifecycle.set_state(bundle.extended_state or {})

        return True
