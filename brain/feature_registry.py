class FeatureRegistry:

    def __init__(self):
        self.features = {}

    # ============================
    # Register feature
    # ============================
    def register(self, name, enabled=True):
        self.features[name] = {
            "enabled": enabled
        }

    # ============================
    # Enable / Disable
    # ============================
    def enable(self, name):
        if name in self.features:
            self.features[name]["enabled"] = True

    def disable(self, name):
        if name in self.features:
            self.features[name]["enabled"] = False

    # ============================
    # Get active features
    # ============================
    def get_active_features(self):
        return [f for f, cfg in self.features.items() if cfg["enabled"]]
