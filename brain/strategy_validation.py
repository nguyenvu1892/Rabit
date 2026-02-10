import pandas as pd

LOG_PATH = "trade_log.csv"


def validate_strategy():

    df = pd.read_csv(LOG_PATH)

    if len(df) == 0:
        return {"valid": False, "reason": "No trades"}

    df = df.dropna(subset=["result_R"])

    if len(df) == 0:
        return {"valid": False, "reason": "No completed trades"}

    total = len(df)

    wins = len(df[df["result_R"] > 0])
    winrate = wins / total

    avg_R = df["result_R"].mean()

    max_dd = df["max_drawdown"].max(skipna=True)
    
    if pd.isna(max_dd):
        max_dd = 0


    valid = (
        winrate > 0.45
        and avg_R > 0
        and max_dd < 0.35
    )

    return {
        "total_trades": total,
        "winrate": round(winrate, 3),
        "avg_R": round(avg_R, 3),
        "max_drawdown": round(max_dd, 3),
        "valid": valid
    }
