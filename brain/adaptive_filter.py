class AdaptiveFilterEngine:

    def __init__(self,
                 min_decision_score=1.0,
                 min_confidence=0.2,
                 min_market_strength=0.4):

        self.min_decision_score = min_decision_score
        self.min_confidence = min_confidence
        self.min_market_strength = min_market_strength

    # ============================
    # Main filter logic
    # ============================
    def evaluate(self,
                 decision_score,
                 context_confidence,
                 market_strength):

        allow = True
        risk_modifier = 1.0
        reasons = []

        # Decision quality check
        if decision_score < self.min_decision_score:
            allow = False
            reasons.append("Decision score too low")

        # Context confidence check
        if context_confidence < self.min_confidence:
            risk_modifier *= 0.7
            reasons.append("Low context confidence")

        # Market strength check
        if market_strength < self.min_market_strength:
            risk_modifier *= 0.7
            reasons.append("Weak market conditions")

        return {
            "allow_trade": allow,
            "risk_modifier": risk_modifier,
            "reasons": reasons
        }
