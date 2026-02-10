def extract_structure(df):

    closes = df["close"]

    trend_strength = closes.rolling(20).mean().iloc[-1] - closes.rolling(50).mean().iloc[-1]

    direction = 1 if trend_strength > 0 else -1

    return {
        "trend_strength": float(trend_strength),
        "trend_direction": int(direction)
    }
