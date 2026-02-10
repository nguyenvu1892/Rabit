from brain.feature_modules.volatility import extract_volatility
from brain.feature_modules.momentum import extract_momentum
from brain.feature_modules.liquidity import extract_liquidity
from brain.feature_modules.structure import extract_structure
from brain.feature_modules.context import extract_context
from brain.feature_modules.microstructure import extract_microstructure


def extract_market_features(df):

    features = {}

    features["volatility"] = extract_volatility(df)
    features["momentum"] = extract_momentum(df)
    features["liquidity"] = extract_liquidity(df)
    features["structure"] = extract_structure(df)
    features["context"] = extract_context(df)
    features["microstructure"] = extract_microstructure(df)

    return features
