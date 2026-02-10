def extract_microstructure(df):

    candle = df.iloc[-1]

    body = abs(candle["close"] - candle["open"])
    total = candle["high"] - candle["low"]

    body_ratio = body / (total + 1e-6)

    bullish = candle["close"] > candle["open"]

    return {
        "body_ratio": float(body_ratio),
        "bullish_candle": int(bullish)
    }
