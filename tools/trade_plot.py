from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Tuple, Optional, DefaultDict
from collections import defaultdict

import matplotlib.pyplot as plt


# -------- CSV loader (supports MT5 export tabs + <> headers) --------

def _normalize_header(h: str) -> str:
    h = h.strip()
    if h.startswith("<") and h.endswith(">"):
        h = h[1:-1]
    return h.strip().lower()

def load_ohlc_from_csv(path: str, limit: Optional[int] = None) -> Tuple[List[int], List[float]]:
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    # MT5 exports are typically tab-separated
    with open(path, "r", encoding="utf-8-sig") as f:
        header_line = f.readline()
        if not header_line:
            raise ValueError("Empty CSV")
        delim = "\t" if "\t" in header_line else ","
        headers = [h for h in header_line.strip().split(delim)]
        norm = [_normalize_header(h) for h in headers]
        col_map = {norm[i]: headers[i] for i in range(len(headers))}

        def pick(*names: str) -> str:
            for n in names:
                if n in col_map:
                    return col_map[n]
            raise ValueError(f"Missing required column among {names}. Found headers={headers}")

        col_close = pick("close", "c")
        col_time = None
        for cand in ("ts", "time", "datetime"):
            if cand in col_map:
                col_time = col_map[cand]
                break

        rows = []
        for line in f:
            if not line.strip():
                continue
            parts = line.rstrip("\n").split(delim)
            row = {headers[i]: parts[i] if i < len(parts) else "" for i in range(len(headers))}
            rows.append(row)
            if limit is not None and len(rows) >= int(limit):
                break

    x: List[int] = []
    close: List[float] = []
    for i, r in enumerate(rows):
        if col_time is None:
            x.append(i)
        else:
            try:
                x.append(int(float(r.get(col_time, i))))
            except Exception:
                x.append(i)
        close.append(float(r[col_close]))
    return x, close


# -------- Journal reader --------

def read_journal(path: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    events: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            try:
                ev = json.loads(raw)
            except Exception:
                continue
            events.append(ev)
            if limit is not None and len(events) >= int(limit):
                break
    return events


def plot_trades(
    csv_path: str,
    journal_path: str,
    out_path: str,
    limit: int = 5000,
    journal_limit: Optional[int] = None,
):
    x, close = load_ohlc_from_csv(csv_path, limit=limit)
    events = read_journal(journal_path, limit=journal_limit)

    # decision events
    decisions = [e for e in events if e.get("type") == "decision"]
    if not decisions:
        raise ValueError("No decision events loaded from journal (type='decision').")

    entries_by_expert: DefaultDict[str, List[int]] = defaultdict(list)
    wins: List[int] = []
    losses: List[int] = []

    # If you also log outcome events, we can mark them too:
    # expected outcome event format: {"type":"outcome","payload":{"step":int,"win":bool,"loss":bool}}
    outcomes = [e for e in events if e.get("type") == "outcome"]
    outcome_map: Dict[int, Dict[str, Any]] = {}
    for e in outcomes:
        p = e.get("payload") or {}
        step = p.get("step")
        if isinstance(step, int):
            outcome_map[step] = p

    # collect entries
    for e in decisions:
        p = e.get("payload") or {}
        step = p.get("step")
        allow = p.get("allow")
        if not (isinstance(step, int) and isinstance(allow, bool)):
            continue
        if not allow:
            continue
        expert = p.get("expert") or "UNKNOWN_EXPERT"
        entries_by_expert[str(expert)].append(step)

        # win/lose by outcome payload if present
        oc = outcome_map.get(step)
        if oc:
            if bool(oc.get("win", False)):
                wins.append(step)
            if bool(oc.get("loss", False)):
                losses.append(step)

    if not entries_by_expert:
        raise ValueError("Journal had decision events but no valid allow entries (payload.step/payload.allow).")

    plt.figure()
    plt.plot(x, close)

    # plot entries per expert with different marker
    markers = ["^", "v", "s", "D", "P", "X", "*", "o"]
    expert_names = sorted(entries_by_expert.keys())
    for i, ex in enumerate(expert_names):
        steps = entries_by_expert[ex]
        xs, ys = [], []
        for st in steps:
            if 0 <= st < len(close):
                xs.append(x[st])
                ys.append(close[st])
        if xs:
            plt.scatter(xs, ys, marker=markers[i % len(markers)], s=30, label=f"{ex} entries({len(xs)})")

    # plot wins/losses if available
    if wins:
        xs, ys = [], []
        for st in wins:
            if 0 <= st < len(close):
                xs.append(x[st]); ys.append(close[st])
        if xs:
            plt.scatter(xs, ys, marker="o", s=20, label=f"wins({len(xs)})")
    if losses:
        xs, ys = [], []
        for st in losses:
            if 0 <= st < len(close):
                xs.append(x[st]); ys.append(close[st])
        if xs:
            plt.scatter(xs, ys, marker="x", s=25, label=f"losses({len(xs)})")

    plt.title("Trade plot (entries by expert) + outcomes")
    plt.xlabel("step (candle index)")
    plt.ylabel("close")
    plt.legend(loc="best")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--journal", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=5000)
    ap.add_argument("--journal-limit", type=int, default=None)
    args = ap.parse_args()

    plot_trades(args.csv, args.journal, args.out, limit=args.limit, journal_limit=args.journal_limit)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
