class MarketStateEngine:

    def __init__(self, session_detector):
        self.session_detector = session_detector

    # ============================
    # Main evaluation
    # ============================
    def evaluate(self, market_features):

        trend = self._detect_trend(market_features)
        volatility = self._detect_volatility(market_features)
        session = self.session_detector.detect(market_features)

        strength = self._compute_strength(trend, volatility)

        return {
            "trend_state": trend,
            "volatility_state": volatility,
            "session_state": session,
            "market_strength": strength
        }

    # ============================
    # Trend detection
    # ============================
    def _detect_trend(self, features):

        bos = features.get("bos", False)
        choch = features.get("choch", False)

        if bos:
            return "trend"
        elif choch:
            return "transition"
        else:
            return "range"

    # ============================
    # Volatility detection
    # ============================
    def _detect_volatility(self, features):

        atr = features.get("atr", 0)

        if atr > 2.0:
            return "high"
        elif atr < 1.0:
            return "low"
        else:
            return "normal"

        # ============================
    # Strength scoring
    # ============================
    def _compute_strength(self, trend, volatility):

        score = 0.5

        if trend == "trend":
            score += 0.3

        if volatility == "high":
            score += 0.2

        return min(score, 1.0)

