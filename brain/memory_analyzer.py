import pandas as pd
import os


class MemoryAnalyzer:

    def __init__(self, trade_log_path="data/trade_outcomes.csv"):
        self.trade_log_path = trade_log_path
        self.df = None

    # =========================
    # LOAD MEMORY
    # =========================
    def load_memory(self):

        if not os.path.exists(self.trade_log_path):
            print("[Memory] File not found:", self.trade_log_path)
            self.df = pd.DataFrame()
            return

        self.df = pd.read_csv(self.trade_log_path)

        print("Loaded columns:", self.df.columns.tolist())

        # --- Column mapping ---
        if "result_R" in self.df.columns and "rr_realized" not in self.df.columns:
            self.df["rr_realized"] = self.df["result_R"]

        if "result_R" in self.df.columns and "result" not in self.df.columns:
            self.df["result"] = self.df["result_R"]

        if "result" in self.df.columns and "outcome" not in self.df.columns:
            self.df["outcome"] = self.df["result"]

        # --- Normalize outcome ---
        if "outcome" in self.df.columns:
            self.df["outcome"] = self.df["outcome"].astype(str).str.upper()

        print(f"[Memory] Loaded {len(self.df)} trades")

    # ===============================
    # CHECK REQUIRED COLUMNS
    # ===============================
    def _check_required_columns(self, cols):

        if self.df is None:
            return False

        for c in cols:
            if c not in self.df.columns:
                print(f"[Memory Warning] Missing column: {c}")
                return False

        return True

    # ===============================
    # BASIC PERFORMANCE
    # ===============================
    def basic_performance(self):

        if self.df is None or len(self.df) == 0:
            return {"error": "No memory loaded"}

        if "result_R" not in self.df.columns:
            return {"error": "Trade log missing result_R"}

        df = self.df

        total = len(df)
        wins = len(df[df["result_R"] > 0])
        losses = len(df[df["result_R"] < 0])
        breakeven = len(df[df["result_R"] == 0])

        winrate = wins / total if total else 0
        avg_rr = df["result_R"].mean()

        return {
            "total_trades": total,
            "wins": wins,
            "losses": losses,
            "breakeven": breakeven,
            "winrate": round(winrate, 3),
            "avg_rr": round(float(avg_rr), 2)
        }


    # ===============================
    # FEATURE EDGE ANALYSIS
    # ===============================
    def discover_edges(self, min_samples=5):

        if not self._check_required_columns(["result_R"]):
            return {}

        edge_report = {}

        for column in self.df.columns:

            if column == "result_R":
                continue

            grouped = self.df.groupby(column)

            for value, group in grouped:

                if len(group) < min_samples:
                    continue

                winrate = (group["result_R"] > 0).mean()
                avg_rr = group["result_R"].mean()

                edge_report[(column, value)] = {
                    "samples": len(group),
                    "winrate": round(winrate, 3),
                    "avg_rr": round(avg_rr, 2)
                }

        return edge_report

    # ===============================
    # FEATURE WINRATE
    # ===============================
    def feature_winrate(self, column):

        if not self._check_required_columns(["outcome"]):
            return {}

        if column not in self.df.columns:
            return {}

        return self.df.groupby(column)["outcome"].apply(
            lambda x: (x == "WIN").mean()
        ).to_dict()

    # ===============================
    # BUILD AI PROFILE
    # ===============================
    def build_performance_profile(self):

        if not self._check_required_columns(["outcome", "rr_realized"]):
            return {}

        features = [
            "session",
            "m5_structure",
            "fvg_valid",
            "ob_valid",
            "volume_confirm",
            "candle_pattern"
        ]

        profile = {}

        for f in features:

            if f not in self.df.columns:
                profile[f] = None
                continue

            winrate = self.df.groupby(f)["outcome"].apply(
                lambda x: (x == "WIN").mean()
            )

            rr = self.df.groupby(f)["rr_realized"].mean()

            profile[f] = {
                "winrate": winrate.to_dict(),
                "avg_rr": rr.to_dict()
            }

        return profile
