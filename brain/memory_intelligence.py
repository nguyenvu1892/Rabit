# brain/memory_intelligence.py
from __future__ import annotations

class MemoryIntelligence:
    def evaluate(self, memory_stats: dict) -> float:
        """
        Return confidence score [0..1] based on past outcomes
        """
        if not memory_stats:
            return 0.5

        samples = memory_stats.get("samples", 0)
        wins = memory_stats.get("wins", 0)
        losses = memory_stats.get("losses", 0)
        avg_pnl = memory_stats.get("avg_pnl", 0.0)

        if samples <= 0:
            return 0.5

        # compute winrate safely
        winrate = wins / max(1, wins + losses)

        # simple confidence model (v1 – ổn định, không overfit)
        confidence = (
            0.5
            + 0.3 * (winrate - 0.5)
            + 0.2 * max(-1.0, min(1.0, avg_pnl))
        )

        # clamp
        return max(0.0, min(1.0, confidence))
