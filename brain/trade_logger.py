import pandas as pd
import os

LOG_FILE = "trade_log.csv"


# =============================
# INIT LOGGER
# =============================
def init_logger():

    if not os.path.exists(LOG_FILE):

        df = pd.DataFrame(columns=[

            # ===== Trade Info =====
            "ticket",
            "time",
            "symbol",
            "session",

            # ===== Market Context =====
            "h1_bias",
            "m5_structure",
            "price_vs_ema",
            "volume_ratio",

            "distance_to_ob",
            "distance_to_fvg",
            "volatility",

            # ===== Signal Validation =====
            "ob_valid",
            "fvg_valid",
            "volume_confirm",
            "candle_pattern",

            # ===== Entry Info =====
            "entry_type",
            "entry_price",
            "sl_price",
            "tp_price",

            # ===== Trade Outcome =====
            "outcome",          # win / loss / breakeven
            "rr_realized",      # R result
            "hold_minutes",
            "exit_price",
            "exit_time",

            # ===== Performance Tracking =====
            "equity",
            "peak_equity",
            "max_drawdown"
        ])

        df.to_csv(LOG_FILE, index=False)


# =============================
# LOG 1 TRADE SNAPSHOT
# =============================
def log_trade(data_dict):

    if not os.path.exists(LOG_FILE):
        init_logger()

    df = pd.read_csv(LOG_FILE)

    # đảm bảo snapshot có đủ key
    for col in df.columns:
        if col not in data_dict:
            data_dict[col] = None

    df = pd.concat([df, pd.DataFrame([data_dict])], ignore_index=True)

    df.to_csv(LOG_FILE, index=False)

    print("Trade logged")
