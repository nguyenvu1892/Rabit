import json
import random


class PortfolioManager:

    def __init__(self, portfolio_path="strategy_portfolio.json"):

        self.portfolio_path = portfolio_path
        self.strategies = self.load_portfolio()

    # -------------------------------------------------
    # LOAD / SAVE PORTFOLIO
    # -------------------------------------------------
    def load_portfolio(self):

        try:
            with open(self.portfolio_path, "r") as f:
                return json.load(f)
        except:
            return []

    def save_portfolio(self):

        with open(self.portfolio_path, "w") as f:
            json.dump(self.strategies, f, indent=4)

    # -------------------------------------------------
    # ADD VALIDATED STRATEGY
    # -------------------------------------------------
    def add_strategy(self, strategy, validation_result):

        entry = {
            "strategy": strategy,
            "score": validation_result["test"]["score"],
            "winrate": validation_result["test"]["winrate"],
            "avg_rr": validation_result["test"]["avg_rr"],
            "active": True
        }

        self.strategies.append(entry)
        self.save_portfolio()

    # -------------------------------------------------
    # DISABLE WEAK STRATEGIES
    # -------------------------------------------------
    def prune_strategies(self, min_score=0.8):

        for s in self.strategies:
            if s["score"] < min_score:
                s["active"] = False

        self.save_portfolio()

    # -------------------------------------------------
    # CAPITAL ALLOCATION
    # -------------------------------------------------
    def allocate_capital(self):

        active = [s for s in self.strategies if s["active"]]

        if not active:
            return []

        total_score = sum([s["score"] for s in active])

        for s in active:
            s["capital_weight"] = s["score"] / total_score

        return active

    # -------------------------------------------------
    # SELECT STRATEGY FOR TRADE
    # -------------------------------------------------
    def select_strategy(self):

        active = self.allocate_capital()

        if not active:
            return None

        weights = [s["capital_weight"] for s in active]

        selected = random.choices(active, weights=weights, k=1)[0]

        return selected["strategy"]
