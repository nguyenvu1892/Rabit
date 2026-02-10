from data.historical_loader import connect_mt5, load_data
import MetaTrader5 as mt5

from brain.trade_logger import log_trade
from brain.session_detector import get_session

from datetime import datetime


# ===== SIMULATE TRADE =====
def simulate_trade(entry_price, sl_price, tp_price, df_future, trade_type):

    for _, row in df_future.iterrows():

        high = row["high"]
        low = row["low"]

        # BUY
        if trade_type == "BUY":

            if low <= sl_price:
                return -1   # SL hit

            if high >= tp_price:
                return 2.5  # TP hit


        # SELL
        if trade_type == "SELL":

            if high >= sl_price:
                return -1

            if low <= tp_price:
                return 2.5

    return None


# ===== SIMPLE TEST STRATEGY =====
def simple_strategy(df, i):

    # dummy strategy (sau ta thay bằng SMC thật)
    if df["close"].iloc[i] > df["open"].iloc[i]:

        return {
            "type": "BUY",
            "sl": df["low"].iloc[i] - 1,
            "tp": df["close"].iloc[i] + 2
        }

    return None


# ===== RUN BACKTEST =====
def run_backtest():

    connect_mt5()

    df = load_data("XAUUSD", mt5.TIMEFRAME_M5, 3000)

    for i in range(50, len(df)-50):

        signal = simple_strategy(df, i)

        if signal is None:
            continue

        entry_price = df["close"].iloc[i]

        result = simulate_trade(
            entry_price,
            signal["sl"],
            signal["tp"],
            df.iloc[i+1:i+50],
            signal["type"]
        )

        log_trade({
            "time": df["time"].iloc[i],
            "symbol": "XAUUSD",
            "session": get_session(),

            "h1_bias": "UNKNOWN",
            "m5_structure": "TEST",

            "price_vs_ema": "TEST",
            "volume_ratio": 1,

            "distance_to_ob": 0,
            "distance_to_fvg": 0,

            "volatility": 0,

            "ob_valid": False,
            "fvg_valid": False,
            "volume_confirm": False,
            "candle_pattern": "TEST",

            "entry_type": signal["type"],
            "entry_price": entry_price,
            "sl_price": signal["sl"],
            "tp_price": signal["tp"],

            "result_R": result,
            "max_drawdown": None,
            "hold_minutes": None
        })

    print("Backtest finished")


if __name__ == "__main__":
    run_backtest()
