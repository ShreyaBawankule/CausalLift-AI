"""
Counterfactual 'What-If' Recourse Simulator.
Explores Actionable Algorithmic Recourse:
    "What minimum modification to customer engagement (channel, recency, cross-category)
     transitions a Lost Cause into a High-Elasticity Persuadable?"
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List


class CounterfactualSimulator:
    """
    Simulates counterfactual scenarios across actionable customer dimensions.
    """
    def __init__(self, model, feature_names: List[str]):
        self.model = model
        self.feature_names = feature_names

    def _predict_tau(self, profile: Dict[str, Any]) -> float:
        df_in = pd.DataFrame([profile])
        if "customer_id" in df_in.columns:
            df_in = df_in.drop(columns=["customer_id"])
        df_enc = pd.get_dummies(df_in, drop_first=True, dtype=int)
        aligned = pd.DataFrame(0, index=[0], columns=self.feature_names)
        for col in df_enc.columns:
            if col in aligned.columns:
                aligned[col] = df_enc[col]
        return float(self.model.predict_uplift(aligned)[0])

    def simulate_channel_scenarios(self, base_profile: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Simulates how changing the primary shopping channel impacts treatment effect.
        """
        scenarios = []
        for ch in ["Web", "Phone", "Multichannel"]:
            temp = base_profile.copy()
            temp["channel"] = ch
            tau = self._predict_tau(temp)
            
            if tau < -0.005:
                seg = "Sleeping Dog"
            elif tau >= 0.015:
                seg = "Persuadable"
            elif temp.get("history", 0) > 300:
                seg = "Sure Thing"
            else:
                seg = "Lost Cause"
                
            scenarios.append({
                "channel": ch,
                "predicted_uplift": round(tau, 4),
                "uplift_percent": f"{tau*100:+.2f}%",
                "resulting_segment": seg
            })
        return scenarios

    def sweep_recency_trajectory(self, base_profile: Dict[str, Any]) -> pd.DataFrame:
        """
        Traces how customer responsiveness decays as months without purchase (recency) increases.
        """
        records = []
        for r in range(1, 13):
            temp = base_profile.copy()
            temp["recency"] = r
            tau = self._predict_tau(temp)
            records.append({
                "months_since_purchase": r,
                "predicted_uplift": round(tau, 4),
                "uplift_pct": round(tau * 100, 2)
            })
        return pd.DataFrame(records)

    def find_optimal_recourse(self, base_profile: Dict[str, Any]) -> Dict[str, Any]:
        """
        Identifies the optimal action to maximize customer elasticity.
        """
        channels = self.simulate_channel_scenarios(base_profile)
        best_channel = max(channels, key=lambda x: x["predicted_uplift"])
        base_tau = self._predict_tau(base_profile)
        
        delta = best_channel["predicted_uplift"] - base_tau
        
        return {
            "current_channel": base_profile.get("channel", "Web"),
            "current_uplift": round(base_tau, 4),
            "best_channel": best_channel["channel"],
            "best_channel_uplift": best_channel["predicted_uplift"],
            "elasticity_gain": round(delta, 4),
            "recourse_recommendation": (
                f"Transition customer to '{best_channel['channel']}' channel to unlock "
                f"+{delta*100:.2f}% incremental treatment responsiveness."
            )
        }
