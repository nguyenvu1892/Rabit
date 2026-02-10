# sim/mock_data_source.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class MockCandleDataSource:
    candles: List[Dict[str, Any]]
    idx: int = 0

    def next(self) -> Optional[Dict[str, Any]]:
        if self.idx >= len(self.candles):
            return None
        c = self.candles[self.idx]
        self.idx += 1
        return c

    def seek(self, idx: int) -> None:
        self.idx = max(0, min(int(idx), len(self.candles)))

    def pos(self) -> int:
        return int(self.idx)
