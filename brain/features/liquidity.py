def extract_liquidity(df):

    candle = df.iloc[-1]

    body = abs(candle["close"] - candle["open"])
    wick = (candle["high"] - candle["low"]) - body

    wick_ratio = wick / (body + 1e-6)

    vol_ratio = candle["tick_volume"] / df["tick_volume"].rolling(20).mean().iloc[-1]

    return {
        "wick_ratio": float(wick_ratio),
        "volume_pressure": float(vol_ratio)
    }
