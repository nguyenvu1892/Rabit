class ContextIntelligence:

    def __init__(self, context_memory):
        self.context_memory = context_memory

    def evaluate_context(self, trade_features):

        stats = self.context_memory.get_stats(trade_features)

        if stats is None:
            return {
                "strength": 1.0,
                "confidence": 0.0,
                "samples": 0
            }

        strength = stats["avg_outcome"]
        confidence = min(1.0, stats["samples"] / 5.0)

        return {
            "strength": strength,
            "confidence": confidence,
            "samples": stats["samples"]
        }
