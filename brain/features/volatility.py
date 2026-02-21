def extract_volatility(df):

    atr = df["high"] - df["low"]

    return {
        "atr_mean": atr.mean()
    }
