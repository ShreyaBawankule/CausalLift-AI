"""
Causal Explainability & Prescriptive Policy Engine.
Provides:
1. Feature importance on heterogeneous treatment effects.
2. Customer quadrant segmentation (Persuadables, Sure Things, Lost Causes, Sleeping Dogs).
3. Interpretable Decision Rules for marketing stakeholders.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List
from sklearn.tree import DecisionTreeRegressor, export_text


def segment_customers(
    uplift_preds: np.ndarray,
    baseline_prob: np.ndarray,
    uplift_threshold: float = 0.015,
    baseline_threshold: float = 0.05
) -> np.ndarray:
    """
    Categorizes customers into four causal quadrants:
    1. Persuadable: High positive uplift (Target!)
    2. Sure Thing: High baseline probability, low uplift (Do not waste promo!)
    3. Lost Cause: Low baseline probability, low uplift (Do not waste promo!)
    4. Sleeping Dog: Negative uplift (Harmful to contact!)
    """
    segments = []
    for tau, p0 in zip(uplift_preds, baseline_prob):
        if tau < -0.005:
            segments.append("Sleeping Dog (Do Not Contact)")
        elif tau >= uplift_threshold:
            segments.append("Persuadable (Priority Target)")
        elif p0 >= baseline_threshold:
            segments.append("Sure Thing (Organic Buyer)")
        else:
            segments.append("Lost Cause (Unresponsive)")
            
    return np.array(segments)


def extract_interpretable_policy_rules(
    X: pd.DataFrame, 
    uplift_preds: np.ndarray, 
    max_depth: int = 3
) -> Dict[str, Any]:
    """
    Fits a surrogate Decision Tree on estimated treatment effects tau(X)
    to extract human-readable IF-THEN decision rules for business stakeholders.
    """
    tree = DecisionTreeRegressor(max_depth=max_depth, random_state=42)
    tree.fit(X, uplift_preds)
    
    rules_text = export_text(tree, feature_names=list(X.columns))
    
    feature_importances = pd.DataFrame({
        "feature": X.columns,
        "importance": tree.feature_importances_
    }).sort_values("importance", ascending=False)
    
    return {
        "surrogate_tree": tree,
        "policy_rules": rules_text,
        "feature_importances": feature_importances
    }
