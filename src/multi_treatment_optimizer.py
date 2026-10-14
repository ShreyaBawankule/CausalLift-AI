"""
Multi-Treatment Policy Optimizer & Economic Constrained Decision Engine.
Solves the optimal incentive allocation problem across multiple competing marketing offers:
    a*(X) = argmax_{a in Actions} [ tau_a(X) * Customer_Value - Cost(a) ]

Policy Simulation Framework:
While RCT datasets like Hillstrom empirically test channel themes ("Mens E-Mail" vs "Womens E-Mail" vs "No E-Mail"),
this optimizer formulates the multi-action Neyman-Rubin causal decision problem across promotional depth tiers:
- a0: "No Outreach / Organic"      (Cost: $0.00)
- a1: "10% Discount Promo"         (Cost: $0.15)
- a2: "20% Deep Discount Promo"    (Cost: $0.35)
- a3: "Free Shipping Voucher"       (Cost: $0.20)
"""

import numpy as np
from typing import Dict, Any, List


class MultiTreatmentOptimizer:
    """
    Evaluates multi-action treatment elasticity and selects the profit-maximizing offer.
    """
    def __init__(self, customer_ltv: float = 80.0):
        self.customer_ltv = customer_ltv
        self.offers = [
            {"id": "a0", "name": "No Outreach (Organic)", "cost": 0.0, "elasticity_multiplier": 0.0},
            {"id": "a1", "name": "10% Discount Promo", "cost": 0.15, "elasticity_multiplier": 0.85},
            {"id": "a2", "name": "20% Deep Discount Promo", "cost": 0.35, "elasticity_multiplier": 1.25},
            {"id": "a3", "name": "Free Shipping Voucher", "cost": 0.20, "elasticity_multiplier": 1.05}
        ]

    def optimize_offer(self, customer_profile: Dict[str, Any], base_tau: float) -> Dict[str, Any]:
        """
        Calculates Net Profit Payoff for all competing offers and picks the optimal policy action.
        """
        channel = customer_profile.get("channel", "Web")
        history = customer_profile.get("history", 150.0)
        
        evaluations = []
        for offer in self.offers:
            # Action-specific treatment effect modeling
            mult = offer["elasticity_multiplier"]
            
            # Channel synergies: Free shipping works better for Web shoppers
            if offer["id"] == "a3" and channel == "Web":
                mult *= 1.20
            # 20% deep discount cannibalizes high-spend customers
            if offer["id"] == "a2" and history > 350.0:
                mult *= 0.70

            tau_a = base_tau * mult
            gross_lift = tau_a * self.customer_ltv
            net_profit = gross_lift - offer["cost"]
            roi = (net_profit / (offer["cost"] + 1e-9)) * 100 if offer["cost"] > 0 else 0.0

            evaluations.append({
                "offer_id": offer["id"],
                "offer_name": offer["name"],
                "offer_cost": offer["cost"],
                "estimated_uplift": round(tau_a, 4),
                "gross_lift_dollars": round(gross_lift, 2),
                "net_profit_dollars": round(net_profit, 2),
                "roi_percent": round(roi, 1)
            })

        # Rank offers by net profit
        sorted_offers = sorted(evaluations, key=lambda x: x["net_profit_dollars"], reverse=True)
        best_offer = sorted_offers[0]

        # If best offer has negative net profit, fallback to No Outreach (a0)
        if best_offer["net_profit_dollars"] <= 0.0 and base_tau <= 0.005:
            optimal_action = evaluations[0]  # a0
            rationale = "All paid promotions deliver negative net profit. Preserving budget with organic outreach."
        else:
            optimal_action = best_offer
            rationale = f"Selected '{best_offer['offer_name']}' which yields highest net incremental profit (${best_offer['net_profit_dollars']:.2f})."

        return {
            "optimal_offer": optimal_action,
            "decision_rationale": rationale,
            "all_competing_offers": evaluations
        }
