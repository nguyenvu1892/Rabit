# features/pipeline.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Protocol, Sequence


Candle = Dict[str, Any]  # expected keys: o,h,l,c,(v optional)


class FeaturePlugin(Protocol):
    name: str
    def compute(self, candles: Sequence[Candle]) -> Dict[str, Any]: ...


@dataclass
class FeaturePipeline:
    plugins: List[FeaturePlugin]

    def compute(self, candles: Sequence[Candle]) -> Dict[str, Any]:
        """
        candles: sequence of dicts with at least keys: o,h,l,c. v optional.
        Returns merged feature dict with namespaced keys.
        """
        if candles is None or len(candles) == 0:
            return {}

        out: Dict[str, Any] = {}
        for p in self.plugins:
            feats = p.compute(candles) or {}
            # namespace keys to avoid collisions
            for k, v in feats.items():
                out[f"{p.name}.{k}"] = v
        return out
