# brain/risk_engine.py

class RiskEngine:
    def __init__(self):
        self.base_risk = 0.01
        self.min_risk = 0.005
        self.max_risk = 0.02

    def scale_risk(self, score, confidence):
        risk = self.base_risk * score * confidence
        return max(self.min_risk, min(self.max_risk, risk))

    # --- NEW (2.4.4) ---
        # --- NEW (2.4.4) ---
    def apply_policy(self, scaled_risk: float, policy_risk: dict | None = None) -> dict:
        """
        Combine scaled_risk (from score/confidence) with strategy policy risk params.
        Returns a config dict for downstream executor.

        policy_risk expected keys:
          - risk_per_trade
          - sl_atr_mult
          - tp_atr_mult
        """
        policy_risk = policy_risk or {}

        # Strategy can override risk_per_trade; fallback to scaled_risk
        risk_per_trade = float(policy_risk.get("risk_per_trade", scaled_risk))

        return {
            "risk_per_trade": risk_per_trade,
            "sl_atr_mult": float(policy_risk.get("sl_atr_mult", 1.5)),
            "tp_atr_mult": float(policy_risk.get("tp_atr_mult", 3.0)),
        }
