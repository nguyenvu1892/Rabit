import pandas as pd
from brain.session_detector import detect_session


def load_xauusd_5y(path="XAUUSD_M5.csv"):

    df = pd.read_csv(
        path,
        sep="\t"
    )

    # ===== CLEAN COLUMN NAME =====
    df.columns = [c.replace("<", "").replace(">", "").lower() for c in df.columns]

    # ===== MERGE DATE + TIME =====
    df["time"] = pd.to_datetime(df["date"] + " " + df["time"])

    # ===== DROP DATE =====
    df = df.drop(columns=["date"])

    # ===== SORT TIME =====
    df = df.sort_values("time").reset_index(drop=True)

    df["session"] = df["time"].apply(detect_session)

    print("Loaded candles from CSV:", len(df))
    print(df.head())


    return df
