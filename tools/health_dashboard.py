# tools/health_dashboard.py
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional
from dataclasses import dataclass
from typing import List

@dataclass
class Warning:
    code: str
    detail: dict


@dataclass
class HealthSummary:
    steps: int = 0
    decisions: int = 0
    orders: int = 0
    executions: int = 0
    outcomes: int = 0

    wins: int = 0
    losses: int = 0
    total_pnl: float = 0.0

    heartbeats: int = 0
    last_heartbeat: Optional[Dict[str, Any]] = None

    risk_pause: int = 0
    risk_resume: int = 0
    session_reset: int = 0

    last_heartbeat_step: int = 0
    max_seen_step: int = 0
 
    def win_rate(self) -> float:
        n = self.wins + self.losses
        return (self.wins / n) if n else 0.0


def read_journal(path: str) -> HealthSummary:
    s = HealthSummary()
    if not os.path.exists(path):
        return s
    idx = data.get("idx", None)
    if isinstance(idx, int):
        s.last_heartbeat_step = max(s.last_heartbeat_step, idx)

    m = (data or {}).get("metrics", {})
    if isinstance(m, dict):
        st = m.get("steps", None)
        if isinstance(st, int):
            s.max_seen_step = max(s.max_seen_step, st)

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except Exception:
                continue

            et = ev.get("type") or ev.get("event") or ev.get("name")
            data = ev.get("data", ev)

            if et == "heartbeat":
                s.heartbeats += 1
                s.last_heartbeat = data
                # metrics inside heartbeat
                m = (data or {}).get("metrics", {})
                if isinstance(m, dict):
                    s.steps = max(s.steps, int(m.get("steps", s.steps)))
                    s.decisions = max(s.decisions, int(m.get("decisions", s.decisions)))
                    s.orders = max(s.orders, int(m.get("orders", s.orders)))
                    s.executions = max(s.executions, int(m.get("executions", s.executions)))
                    s.outcomes = max(s.outcomes, int(m.get("outcomes", s.outcomes)))
                    s.wins = max(s.wins, int(m.get("wins", s.wins)))
                    s.losses = max(s.losses, int(m.get("losses", s.losses)))
                    s.total_pnl = float(m.get("total_pnl", s.total_pnl))

            elif et == "decision":
                s.decisions += 1
            elif et == "order_plan":
                s.orders += 1
            elif et == "execution":
                s.executions += 1
            elif et == "outcome":
                s.outcomes += 1
                pnl = float((data or {}).get("pnl", 0.0))
                win = bool((data or {}).get("win", False))
                s.total_pnl += pnl
                if win:
                    s.wins += 1
                else:
                    s.losses += 1

            elif et == "risk_pause":
                s.risk_pause += 1
            elif et == "risk_resume":
                s.risk_resume += 1
            elif et == "session_reset":
                s.session_reset += 1
        step = (data or {}).get("step", None)
        if isinstance(step, int):
            s.max_seen_step = max(s.max_seen_step, step)

    return s


def render_summary(s: HealthSummary) -> str:
    lines = []
    lines.append("=== XAU_AI_BOT HEALTH DASHBOARD ===")
    lines.append(f"steps={s.steps} decisions={s.decisions} orders={s.orders} executions={s.executions} outcomes={s.outcomes}")
    lines.append(f"wins={s.wins} losses={s.losses} win_rate={s.win_rate():.2%} total_pnl={s.total_pnl:.2f}")
    lines.append(f"heartbeats={s.heartbeats} risk_pause={s.risk_pause} risk_resume={s.risk_resume} session_reset={s.session_reset}")
    lines.append(f"last_heartbeat_step={s.last_heartbeat_step} max_seen_step={s.max_seen_step}")

    if s.last_heartbeat:
        idx = s.last_heartbeat.get("idx", None)
        run_id = s.last_heartbeat.get("run_id", "")
        sh = s.last_heartbeat.get("strategy_hash", "")
        lines.append(f"last_heartbeat: idx={idx} run_id={run_id} strategy_hash={sh}")

    return "\n".join(lines)


def detect_anomalies(s: HealthSummary,
                     heartbeat_expected_min: int = 1,
                     max_pause_rate: float = 0.10,
                     min_samples_for_pause_rate: int = 50,
                     min_outcomes_for_winrate: int = 30,
                     min_win_rate: float = 0.35,
                     max_daily_loss_abs: float = 0.0, heartbeat_stall_threshold: int = 200, current_step: int | None = None) -> list:
    """
    V1 warnings:
      - no_heartbeat: no heartbeat at all
      - excessive_risk_pause: risk_pause rate too high
      - low_win_rate: win_rate too low (requires enough outcomes)
      - losing_pnl: total_pnl < -max_daily_loss_abs if configured (>0)
    """
    warns = []

    # 1) heartbeat missing (simple V1)
    if s.heartbeats < heartbeat_expected_min:
        warns.append(Warning("no_heartbeat", {"heartbeats": s.heartbeats}))

    # 2) risk pause too often (only if enough samples)
    if s.steps >= min_samples_for_pause_rate and s.steps > 0:
        pause_rate = s.risk_pause / float(s.steps)
        if pause_rate > max_pause_rate:
            warns.append(Warning("excessive_risk_pause", {"pause_rate": pause_rate, "risk_pause": s.risk_pause, "steps": s.steps}))

    # 3) low win rate (only if enough outcomes)
    if s.outcomes >= min_outcomes_for_winrate:
        wr = s.win_rate()
        if wr < min_win_rate:
            warns.append(Warning("low_win_rate", {"win_rate": wr, "wins": s.wins, "losses": s.losses, "outcomes": s.outcomes}))

    # 4) losing pnl threshold (optional)
    if max_daily_loss_abs and max_daily_loss_abs > 0:
        if s.total_pnl <= -abs(max_daily_loss_abs):
            warns.append(Warning("losing_pnl", {"total_pnl": s.total_pnl, "threshold": -abs(max_daily_loss_abs)}))

    # Heartbeat stalled
    cur = current_step if isinstance(current_step, int) else s.max_seen_step
    last = s.last_heartbeat_step
    if last > 0 and cur > 0:
        gap = cur - last
        if gap > heartbeat_stall_threshold:
            warns.append(Warning("heartbeat_stalled", {"gap": gap, "last_heartbeat_step": last, "current_step": cur}))        

    return warns


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--journal", default="journal.jsonl")
    ap.add_argument("--current-step", type=int, default=None)
    ap.add_argument("--hb-stall", type=int, default=200)
    args = ap.parse_args()

    # 1️⃣ đọc journal → tạo s
    s = read_journal(args.journal)

    # 2️⃣ detect warnings
    warns = detect_anomalies(
        s,
        current_step=args.current_step,
        heartbeat_stall_threshold=args.hb_stall,
    )

    # 3️⃣ render
    print(render_summary(s))

    if warns:
        print("WARNINGS:")
        for w in warns:
            print(f"- {w.code} {w.detail}")


if __name__ == "__main__":
    main()
