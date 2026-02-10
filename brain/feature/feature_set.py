# brain/feature/feature_set.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Sequence

from brain.feature.pipeline import FeaturePipeline
from brain.feature.market_structure import MarketStructureFeatures
from brain.feature.price_action import PriceActionFeatures
from brain.feature.volume import VolumeFeatures


Candle = Dict[str, Any]


@dataclass
class FeatureSet:
    """
    High-level feature builder that returns:
      - core keys for DecisionEngine/Policy
      - plus namespaced plugin features for RL learning
    """
    symbol: str = "XAUUSD"
    vol_state_threshold: float = 0.01  # relative range threshold (tweak later)

    def __post_init__(self):
        self.pipeline = FeaturePipeline(
            plugins=[
                MarketStructureFeatures(),
                PriceActionFeatures(),
                VolumeFeatures(),
            ]
        )

    def compute(self, candles: Sequence[Candle]) -> Dict[str, Any]:
        feats = self.pipeline.compute(candles)

        # --- core mappings ---
        # trend_state from market structure
        trend_state = feats.get("ms.trend_state")

        # volatility_state (simple v1): use pa.range / last_close
        last_close = float(candles[-1]["c"]) if candles else 0.0
        last_range = float(feats.get("pa.range", 0.0))
        rel_range = 0.0 if abs(last_close) < 1e-9 else (last_range / abs(last_close))

        volatility_state = "high" if rel_range >= self.vol_state_threshold else "low"

        out = dict(feats)
        out["symbol"] = self.symbol
        out["trend_state"] = trend_state
        out["volatility_state"] = volatility_state
        out["rel_range"] = float(rel_range)

        return out
