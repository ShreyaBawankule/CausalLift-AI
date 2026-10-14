"""
Causal Assumptions & Diagnostic Validation Module.
Verifies:
1. Positivity / Common Support (Propensity Score Overlap).
2. Covariate Balance via Standardized Mean Differences (SMD).
3. Treatment vs. Control distribution diagnostics.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def calculate_propensity_scores(
    X: pd.DataFrame, 
    treatment: pd.Series
) -> Tuple[np.ndarray, LogisticRegression]:
    """
    Estimates the propensity score e(X) = P(W=1 | X) using regularized logistic regression.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_scaled, treatment)
    propensity_scores = clf.predict_proba(X_scaled)[:, 1]
    
    return propensity_scores, clf


def check_common_support(
    propensity_scores: np.ndarray, 
    treatment: pd.Series, 
    threshold: float = 0.01
) -> Dict[str, Any]:
    """
    Evaluates whether the positivity assumption holds:
    0 < P(W=1|X) < 1 across all observation strata.
    """
    ps_treat = propensity_scores[treatment == 1]
    ps_ctrl = propensity_scores[treatment == 0]
    
    min_treat, max_treat = np.min(ps_treat), np.max(ps_treat)
    min_ctrl, max_ctrl = np.min(ps_ctrl), np.max(ps_ctrl)
    
    # Overlap interval: [max(min_treat, min_ctrl), min(max_treat, max_ctrl)]
    overlap_min = max(min_treat, min_ctrl)
    overlap_max = min(max_treat, max_ctrl)
    
    in_support = (propensity_scores >= overlap_min) & (propensity_scores <= overlap_max)
    support_pct = np.mean(in_support) * 100
    
    violating_lower = np.mean(propensity_scores < threshold) * 100
    violating_upper = np.mean(propensity_scores > (1.0 - threshold)) * 100
    
    return {
        "treatment_ps_min": float(min_treat),
        "treatment_ps_max": float(max_treat),
        "control_ps_min": float(min_ctrl),
        "control_ps_max": float(max_ctrl),
        "common_support_range": (float(overlap_min), float(overlap_max)),
        "percent_in_common_support": float(support_pct),
        "percent_extreme_positivity_violations": float(violating_lower + violating_upper),
        "positivity_valid": bool(support_pct >= 95.0)
    }


def calculate_standardized_mean_differences(
    X: pd.DataFrame, 
    treatment: pd.Series
) -> pd.DataFrame:
    """
    Computes Standardized Mean Differences (SMD) for all covariates between Treatment and Control.
    Rule of Thumb:
      - SMD < 0.10: Excellent covariate balance (negligible selection bias)
      - SMD 0.10 - 0.20: Acceptable balance
      - SMD > 0.20: Severe selection bias requiring propensity adjustment / weighting
    """
    X_treat = X[treatment == 1]
    X_ctrl = X[treatment == 0]
    
    smd_records = []
    for col in X.columns:
        mean_treat = X_treat[col].mean()
        mean_ctrl = X_ctrl[col].mean()
        var_treat = X_treat[col].var()
        var_ctrl = X_ctrl[col].var()
        
        pooled_sd = np.sqrt((var_treat + var_ctrl) / 2.0)
        smd = (mean_treat - mean_ctrl) / (pooled_sd + 1e-9)
        
        smd_records.append({
            "covariate": col,
            "mean_treated": float(mean_treat),
            "mean_control": float(mean_ctrl),
            "smd": float(np.abs(smd)),
            "is_balanced": bool(np.abs(smd) < 0.10)
        })
        
    smd_df = pd.DataFrame(smd_records).sort_values("smd", ascending=False)
    return smd_df


def run_full_causal_diagnostics(
    X: pd.DataFrame, 
    treatment: pd.Series
) -> Dict[str, Any]:
    """
    Runs full diagnostic suite for causal inference assumptions.
    """
    ps, _ = calculate_propensity_scores(X, treatment)
    support_results = check_common_support(ps, treatment)
    smd_df = calculate_standardized_mean_differences(X, treatment)
    
    max_smd = smd_df["smd"].max()
    unbalanced_covariates = smd_df[~smd_df["is_balanced"]]["covariate"].tolist()
    
    return {
        "propensity_scores": ps,
        "common_support": support_results,
        "smd_table": smd_df,
        "max_smd": float(max_smd),
        "unbalanced_covariates": unbalanced_covariates,
        "overall_causal_assumptions_passed": bool(
            support_results["positivity_valid"] and (len(unbalanced_covariates) == 0)
        )
    }
