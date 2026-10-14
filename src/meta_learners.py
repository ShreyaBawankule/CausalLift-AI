"""
Causal Meta-Learners for Individual Treatment Effect (ITE) Estimation.
Implements:
1. S-Learner (Single Model with treatment indicator)
2. T-Learner (Two separate models for Control and Treatment)
3. X-Learner (Künzel et al. 2019 - Two-stage counterfactual imputation)
4. Model comparison wrapper
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
from sklearn.base import clone, BaseEstimator
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.linear_model import LogisticRegression


class SLearner:
    """
    S-Learner (Single Model):
    Estimates CATE using a single estimator conditioned on covariates and treatment.
    tau(X) = mu(X, 1) - mu(X, 0)
    """
    def __init__(self, base_estimator: Optional[BaseEstimator] = None):
        self.base_estimator = base_estimator or LGBMClassifier(
            n_estimators=150, learning_rate=0.05, max_depth=5, random_state=42, verbose=-1
        )
        self.model = None

    def fit(self, X: pd.DataFrame, treatment: pd.Series, outcome: pd.Series):
        X_aug = X.copy()
        X_aug["_treatment_indicator_"] = treatment.values
        self.model = clone(self.base_estimator)
        self.model.fit(X_aug, outcome)
        return self

    def predict_uplift(self, X: pd.DataFrame) -> np.ndarray:
        X_treat = X.copy()
        X_treat["_treatment_indicator_"] = 1
        X_ctrl = X.copy()
        X_ctrl["_treatment_indicator_"] = 0
        
        if hasattr(self.model, "predict_proba"):
            p1 = self.model.predict_proba(X_treat)[:, 1]
            p0 = self.model.predict_proba(X_ctrl)[:, 1]
        else:
            p1 = self.model.predict(X_treat)
            p0 = self.model.predict(X_ctrl)
            
        return p1 - p0


class TLearner:
    """
    T-Learner (Two Models):
    Trains separate models for Treatment and Control groups.
    tau(X) = mu_1(X) - mu_0(X)
    """
    def __init__(self, base_estimator: Optional[BaseEstimator] = None):
        self.base_estimator = base_estimator or LGBMClassifier(
            n_estimators=150, learning_rate=0.05, max_depth=5, random_state=42, verbose=-1
        )
        self.model_ctrl = None
        self.model_treat = None

    def fit(self, X: pd.DataFrame, treatment: pd.Series, outcome: pd.Series):
        idx_ctrl = (treatment == 0)
        idx_treat = (treatment == 1)
        
        self.model_ctrl = clone(self.base_estimator)
        self.model_ctrl.fit(X[idx_ctrl], outcome[idx_ctrl])
        
        self.model_treat = clone(self.base_estimator)
        self.model_treat.fit(X[idx_treat], outcome[idx_treat])
        return self

    def predict_uplift(self, X: pd.DataFrame) -> np.ndarray:
        if hasattr(self.model_treat, "predict_proba"):
            p1 = self.model_treat.predict_proba(X)[:, 1]
            p0 = self.model_ctrl.predict_proba(X)[:, 1]
        else:
            p1 = self.model_treat.predict(X)
            p0 = self.model_ctrl.predict(X)
            
        return p1 - p0


class XLearner:
    """
    X-Learner (Künzel et al. 2019):
    Specifically engineered for unequal treatment/control sample sizes.
    1. First stage: Estimate mu_0 and mu_1 using T-Learner.
    2. Impute counterfactual effects:
       D_1 = Y_1 - mu_0(X_1)
       D_0 = mu_1(X_0) - Y_0
    3. Second stage: Fit tau_1(X) ~ D_1 and tau_0(X) ~ D_0.
    4. Combine: tau(X) = e(X)*tau_0(X) + (1 - e(X))*tau_1(X).
    """
    def __init__(
        self, 
        stage1_estimator: Optional[BaseEstimator] = None,
        stage2_estimator: Optional[BaseEstimator] = None
    ):
        self.stage1_estimator = stage1_estimator or LGBMClassifier(
            n_estimators=100, learning_rate=0.05, max_depth=4, random_state=42, verbose=-1
        )
        self.stage2_estimator = stage2_estimator or LGBMRegressor(
            n_estimators=100, learning_rate=0.05, max_depth=4, random_state=42, verbose=-1
        )
        self.model_ctrl = None
        self.model_treat = None
        self.model_tau_ctrl = None
        self.model_tau_treat = None
        self.propensity_model = None

    def fit(self, X: pd.DataFrame, treatment: pd.Series, outcome: pd.Series):
        X_arr = X.values if isinstance(X, pd.DataFrame) else X
        w_arr = treatment.values
        y_arr = outcome.values

        idx_ctrl = (w_arr == 0)
        idx_treat = (w_arr == 1)

        # Stage 0: Propensity model e(X)
        self.propensity_model = LogisticRegression(max_iter=500, random_state=42)
        self.propensity_model.fit(X_arr, w_arr)

        # Stage 1: Fit outcome response models
        self.model_ctrl = clone(self.stage1_estimator)
        self.model_ctrl.fit(X_arr[idx_ctrl], y_arr[idx_ctrl])

        self.model_treat = clone(self.stage1_estimator)
        self.model_treat.fit(X_arr[idx_treat], y_arr[idx_treat])

        # Counterfactual Imputation
        if hasattr(self.model_ctrl, "predict_proba"):
            y0_hat_for_treat = self.model_ctrl.predict_proba(X_arr[idx_treat])[:, 1]
            y1_hat_for_ctrl = self.model_treat.predict_proba(X_arr[idx_ctrl])[:, 1]
        else:
            y0_hat_for_treat = self.model_ctrl.predict(X_arr[idx_treat])
            y1_hat_for_ctrl = self.model_treat.predict(X_arr[idx_ctrl])

        D_treat = y_arr[idx_treat] - y0_hat_for_treat
        D_ctrl = y1_hat_for_ctrl - y_arr[idx_ctrl]

        # Stage 2: Fit treatment effect models
        self.model_tau_treat = clone(self.stage2_estimator)
        self.model_tau_treat.fit(X_arr[idx_treat], D_treat)

        self.model_tau_ctrl = clone(self.stage2_estimator)
        self.model_tau_ctrl.fit(X_arr[idx_ctrl], D_ctrl)

        return self

    def predict_uplift(self, X: pd.DataFrame) -> np.ndarray:
        X_arr = X.values if isinstance(X, pd.DataFrame) else X
        
        # Propensity weights e(X)
        propensity = self.propensity_model.predict_proba(X_arr)[:, 1]
        
        tau_ctrl_pred = self.model_tau_ctrl.predict(X_arr)
        tau_treat_pred = self.model_tau_treat.predict(X_arr)
        
        # Convex combination
        tau = propensity * tau_ctrl_pred + (1.0 - propensity) * tau_treat_pred
        return tau


def train_all_learners(
    X_train: pd.DataFrame, 
    tr_train: pd.Series, 
    y_train: pd.Series
) -> Dict[str, Any]:
    """
    Fits S-Learner, T-Learner, and X-Learner on the training split.
    """
    models = {
        "S-Learner": SLearner(),
        "T-Learner": TLearner(),
        "X-Learner": XLearner()
    }
    
    fitted_models = {}
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, tr_train, y_train)
        fitted_models[name] = model
        
    return fitted_models
