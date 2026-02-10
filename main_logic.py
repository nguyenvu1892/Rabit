from datetime import datetime
import pandas as pd

# ===== Brain Modules =====
from brain.trade_logger import init_logger, log_trade
from brain.session_detector import get_session
from brain.adaptive_filter import allow_trade
from brain.market_feature_engine import extract_market_features

# ===== Observer =====
from observer.trade_snapshot_builder import build_trade_snapshot

# ===== Data Loader =====
from data.historical_loader import connect_mt5, load_data


# ---------------------------------------------------
# Fake Signal Generator (Sau này thay bằng ML / RL)
# ---------------------------------------------------
def generate_signal():
    return {
        "type": "BUY",
        "entry_price": 2030,
        "sl": 2025,
        "tp": 2040,

        # ===== Features cho AI học =====
        "h1_bias": "BUY",
        "m5_structure": "BOS_UP",
        "price_vs_ema": "ABOVE",
        "volume_ratio": 1.5,
        "distance_to_ob": 5,
        "distance_to_fvg": 3,
        "volatility": 12,
        "ob_valid": True,
        "fvg_valid": True,
        "volume_confirm": True,
        "candle_pattern": "Bullish Engulf"
    }


# ---------------------------------------------------
# Load Market Data
# ---------------------------------------------------
def load_market_data():
    """
    Tạm load historical data từ MT5
    Sau này có thể replace bằng realtime feed
    """

    connect_mt5()

    df = load_data(
        symbol="XAUUSD",
        timeframe="M5",
        n_bars=200
    )

    return df


# ---------------------------------------------------
# MAIN BOT LOOP
# ---------------------------------------------------
def main():

    init_logger()
    print("🤖 AI Trader Started")

    # 1️⃣ Detect Session
    session = get_session()

    # 2️⃣ Load Market Data
    df = load_market_data()

    if df is None or df.empty:
        print("❌ No market data")
        return

    # 3️⃣ Extract Market Features
    market_features = extract_market_features(df)

    # 4️⃣ Generate Trading Signal
    signal = generate_signal()

    # 5️⃣ Adaptive Filter
    if not allow_trade(session, signal["type"]):
        print("⛔ Trade blocked by adaptive filter")
        return

    print("✅ Trade allowed → Logging snapshot")

    # 6️⃣ Build Snapshot
    snapshot = build_trade_snapshot(
        signal=signal,
        market_features=market_features,
        session=session
    )

    # 7️⃣ Log Trade Dataset
    log_trade(snapshot)

    print("📊 Snapshot logged successfully")


# ---------------------------------------------------
if __name__ == "__main__":
    main()
