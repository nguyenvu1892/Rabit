class OutcomeLearner:

    def __init__(self, rl, trade_memory):
        self.rl = rl
        self.trade_memory = trade_memory

    # -------------------------
    # Learn from trade outcome
    # -------------------------
    def learn(self, trade_features, pnl):
        reward = pnl

        for feature, value in trade_features.items():
            self.rl.update(feature, value, reward)

        self.trade_memory.update(trade_features, pnl)

    # -------------------------
    # Reward model
    # -------------------------
    def _calculate_reward(self, pnl):

        if pnl > 0:
            return 1.0
        elif pnl < 0:
            return -1.0
        return 0.0
