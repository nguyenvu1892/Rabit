# tools/data_report.py
from __future__ import annotations

from sim.candle_loader import load_candles_csv
from sim.data_validator import validate_candles

import glob

def _auto_find_csv():
    cands = glob.glob("data/XAUUSD*.csv")
    if not cands:
        raise FileNotFoundError("No data/XAUUSD*.csv found")
    return sorted(cands)[0]

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="path to XAUUSD csv")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    candles = load_candles_csv(args.csv, limit=args.limit)
    rep = validate_candles(candles)

    d = rep.to_dict()
    print("=== DATA REPORT ===")
    for k in sorted(d.keys()):
        print(f"{k}: {d[k]}")


if __name__ == "__main__":
    main()
