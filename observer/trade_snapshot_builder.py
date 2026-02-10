class TradeSnapshotBuilder:

    def __init__(self, registry):
        self.registry = registry

    def build_snapshot(self, trade_data):

        snapshot = {}

        active_features = self.registry.get_active_features()

        for feature in active_features:
            if feature in trade_data:
                snapshot[feature] = trade_data[feature]

        return snapshot
