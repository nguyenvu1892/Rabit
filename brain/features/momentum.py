import numpy as np


def extract_momentum(df):

    returns = df["close"].pct_change().dropna()

    velocity = returns.rolling(5).mean().iloc[-1]
    acceleration = returns.diff().rolling(5).mean().iloc[-1]

    return {
        "return_velocity": float(velocity),
        "return_acceleration": float(acceleration)
    }
