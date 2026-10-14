"""
Uplift Evaluation & Business Decision Engine.
Implements:
1. Qini Curve & Qini Score calculation.
2. Cumulative Gain / Area Under Uplift Curve (AUUC).
3. Uplift by Deciles (Persuadables to Sleeping Dogs).
4. Business Net Profit & ROI Simulation vs. Standard ML.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple


def calculate_qini_curve(
    y_true: np.ndarray, 
    treatment: np.ndarray, 
    uplift_preds: np.ndarray, 
    n_bins: int = 100
) -> Dict[str, np.ndarray]:
    """
    Computes the Qini curve by ranking observations by predicted uplift.
    
    Qini(k) = n_{t,1}(k) - n_{c,1}(k) * (n_t(k) / n_c(k))
    """
    order = np.argsort(-uplift_preds)
    y_sorted = y_true[order]
    w_sorted = treatment[order]
    
    n_samples = len(y_true)
    bin_size = max(1, n_samples // n_bins)
    
    fractions = []
    qini_values = []
    random_values = []
    
    cum_n_t1 = 0
    cum_n_t = 0
    cum_n_c1 = 0
    cum_n_c = 0
    
    total_n_t1 = np.sum((w_sorted == 1) & (y_sorted == 1))
    total_n_t = np.sum(w_sorted == 1)
    total_n_c1 = np.sum((w_sorted == 0) & (y_sorted == 1))
    total_n_c = np.sum(w_sorted == 0)
    
    overall_qini_end = total_n_t1 - total_n_c1 * (total_n_t / (total_n_c + 1e-9))
    
    for i in range(1, n_bins + 1):
        idx = min(i * bin_size, n_samples)
        sub_y = y_sorted[:idx]
        sub_w = w_sorted[:idx]
        
        n_t1 = np.sum((sub_w == 1) & (sub_y == 1))
        n_t = np.sum(sub_w == 1)
        n_c1 = np.sum((sub_w == 0) & (sub_y == 1))
        n_c = np.sum(sub_w == 0)
        
        # Avoid division by zero
        if n_c == 0:
            qini_val = 0.0
        else:
            qini_val = n_t1 - n_c1 * (n_t / n_c)
            
        frac = idx / n_samples
        fractions.append(frac)
        qini_values.append(qini_val)
        random_values.append(frac * overall_qini_end)
        
    fractions = np.array([0.0] + fractions)
    qini_values = np.array([0.0] + qini_values)
    random_values = np.array([0.0] + random_values)
    
    # Trapezoidal integration compatible with both NumPy 1.x and 2.x
    if hasattr(np, "trapezoid"):
        trapz_func = np.trapezoid
    elif hasattr(np, "trapz"):
        trapz_func = np.trapz
    else:
        def trapz_func(y, x):
            return float(np.sum((x[1:] - x[:-1]) * (y[1:] + y[:-1]) / 2.0))

    qini_score = float(trapz_func(qini_values, fractions) - trapz_func(random_values, fractions))
    
    return {
        "fractions": fractions,
        "qini_values": qini_values,
        "random_values": random_values,
        "qini_score": float(qini_score)
    }


def compute_uplift_deciles(
    y_true: np.ndarray, 
    treatment: np.ndarray, 
    uplift_preds: np.ndarray, 
    n_deciles: int = 10
) -> pd.DataFrame:
    """
    Partitions population into 10 deciles by predicted uplift and calculates
    empirical conversion rate in treatment vs control for each decile.
    """
    df = pd.DataFrame({
        "y": y_true,
        "treatment": treatment,
        "uplift_pred": uplift_preds
    })
    
    df["decile"] = pd.qcut(df["uplift_pred"].rank(method="first"), q=n_deciles, labels=False)
    # Reverse so decile 1 is top uplift
    df["decile"] = n_deciles - df["decile"]
    
    stats = []
    for d in range(1, n_deciles + 1):
        sub = df[df["decile"] == d]
        t_sub = sub[sub["treatment"] == 1]
        c_sub = sub[sub["treatment"] == 0]
        
        conv_t = t_sub["y"].mean() if len(t_sub) > 0 else 0.0
        conv_c = c_sub["y"].mean() if len(c_sub) > 0 else 0.0
        obs_uplift = conv_t - conv_c
        
        stats.append({
            "decile": d,
            "sample_size": len(sub),
            "treatment_count": len(t_sub),
            "control_count": len(c_sub),
            "treatment_conv_rate": float(conv_t),
            "control_conv_rate": float(conv_c),
            "empirical_uplift": float(obs_uplift),
            "mean_predicted_uplift": float(sub["uplift_pred"].mean())
        })
        
    return pd.DataFrame(stats)


def simulate_business_roi(
    y_true: np.ndarray,
    treatment: np.ndarray,
    uplift_preds: np.ndarray,
    customer_value: float = 80.0,
    contact_cost: float = 0.10,
    target_fractions: list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
) -> pd.DataFrame:
    """
    Simulates Financial Net Profit and ROI comparing:
    1. Causal Uplift Targeting
    2. Random Targeting
    """
    n_samples = len(y_true)
    order = np.argsort(-uplift_preds)
    y_sorted = y_true[order]
    w_sorted = treatment[order]
    
    records = []
    
    for frac in target_fractions:
        k = int(frac * n_samples)
        if k == 0:
            continue
            
        sub_y = y_sorted[:k]
        sub_w = w_sorted[:k]
        
        n_t1 = np.sum((sub_w == 1) & (sub_y == 1))
        n_t = np.sum(sub_w == 1)
        n_c1 = np.sum((sub_w == 0) & (sub_y == 1))
        n_c = np.sum(sub_w == 0)
        
        # Incremental conversions = (Conv_Rate_T - Conv_Rate_C) * Target_Size
        p_t = n_t1 / (n_t + 1e-9)
        p_c = n_c1 / (n_c + 1e-9)
        incremental_rate = max(0.0, p_t - p_c)
        
        incremental_conversions = incremental_rate * k
        gross_revenue = incremental_conversions * customer_value
        total_cost = k * contact_cost
        net_profit = gross_revenue - total_cost
        roi = (net_profit / (total_cost + 1e-9)) * 100
        
        # Baseline comparison: Random outreach
        overall_p_t = np.sum((treatment == 1) & (y_true == 1)) / (np.sum(treatment == 1) + 1e-9)
        overall_p_c = np.sum((treatment == 0) & (y_true == 1)) / (np.sum(treatment == 0) + 1e-9)
        random_inc_rate = max(0.0, overall_p_t - overall_p_c)
        random_profit = (random_inc_rate * k * customer_value) - total_cost
        
        records.append({
            "target_fraction": frac,
            "targeted_customers": k,
            "incremental_conversions": round(incremental_conversions, 1),
            "campaign_cost": round(total_cost, 2),
            "gross_revenue": round(gross_revenue, 2),
            "uplift_net_profit": round(net_profit, 2),
            "random_net_profit": round(random_profit, 2),
            "net_lift_dollars": round(net_profit - random_profit, 2),
            "roi_percent": round(roi, 1)
        })
        
    return pd.DataFrame(records)
