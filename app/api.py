"""
FastAPI Microservice for Real-Time Causal Uplift & Prescriptive Intervention.
Endpoints:
- GET  /health              : Service health check
- GET  /model-info          : Metadata and benchmark summary
- POST /predict-uplift      : Real-time CATE estimation and policy decision
- POST /copilot-brief       : GenAI-powered executive counterfactual reasoning & ad creative
- POST /conformal-predict   : Conformal 90% uncertainty intervals & statistical risk bounds
- POST /multi-treatment-opt : Multi-offer optimization (10% vs 20% vs Free Shipping)
- POST /what-if-simulate    : Counterfactual recourse simulation across channels
"""

import os
import sys

# Ensure project root is in sys.path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import joblib
import numpy as np
import pandas as pd
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.causal_copilot import generate_causal_brief
from src.conformal_uplift import get_default_conformal_predictor
from src.multi_treatment_optimizer import MultiTreatmentOptimizer
from src.counterfactual_simulator import CounterfactualSimulator


app = FastAPI(
    title="Causal Uplift & GenAI Decision Platform",
    description="Real-time individual treatment effect estimation, conformal uncertainty, and multi-offer optimization",
    version="3.0.0"
)

MODEL_PATH = os.path.join(ROOT_DIR, "models", "best_uplift_model.pkl")
FEATURES_PATH = os.path.join(ROOT_DIR, "models", "feature_names.pkl")

# In-memory singletons
model = None
expected_features = None
conformal_engine = None
multi_optimizer = None
whatif_sim = None


@app.on_event("startup")
def load_artifacts():
    global model, expected_features, conformal_engine, multi_optimizer, whatif_sim
    if os.path.exists(MODEL_PATH) and os.path.exists(FEATURES_PATH):
        model = joblib.load(MODEL_PATH)
        expected_features = joblib.load(FEATURES_PATH)
        conformal_engine = get_default_conformal_predictor()
        multi_optimizer = MultiTreatmentOptimizer(customer_ltv=80.0)
        whatif_sim = CounterfactualSimulator(model, expected_features)
        print("Model, conformal engine, and multi-offer optimizer initialized successfully.")
    else:
        print("Warning: Model artifacts not found yet. Run pipeline first.")


class CustomerProfile(BaseModel):
    customer_id: str = Field(..., example="CUST_10492")
    recency: int = Field(..., ge=1, le=36, description="Months since last purchase", example=3)
    history: float = Field(..., ge=0.0, description="Past year spend ($)", example=245.50)
    mens: int = Field(..., ge=0, le=1, description="Purchased mens merchandise in past year", example=1)
    womens: int = Field(..., ge=0, le=1, description="Purchased womens merchandise in past year", example=0)
    zip_code: str = Field(..., description="Urban, Suburban, or Rural", example="Suburban")
    newbie: int = Field(..., ge=0, le=1, description="New customer in past 12 months", example=0)
    channel: str = Field(..., description="Web, Phone, or Multichannel", example="Web")


class PrescriptiveDecision(BaseModel):
    customer_id: str
    predicted_uplift: float
    segment: str
    recommended_action: str
    priority_level: str
    expected_incremental_value: float


class DynamicCreative(BaseModel):
    subject_line: str
    discount_tier: str
    target_category: str
    preferred_channel: str
    email_body: str


class CausalCopilotBrief(BaseModel):
    customer_id: str
    causal_quadrant: str
    estimated_uplift_tau: float
    baseline_probability: float
    treated_probability: float
    expected_incremental_lift_dollars: float
    executive_rationale: str
    recommended_policy_action: str
    urgency_tier: str
    dynamic_creative: DynamicCreative


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "features_count": len(expected_features) if expected_features else 0
    }


@app.get("/model-info")
def model_info():
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not yet loaded.")
    return {
        "model_class": model.__class__.__name__,
        "features": expected_features
    }


def _infer_tau_and_segment(customer: CustomerProfile):
    raw_dict = customer.dict()
    cid = raw_dict.pop("customer_id")
    df_input = pd.DataFrame([raw_dict])
    
    df_encoded = pd.get_dummies(df_input, drop_first=True, dtype=int)
    aligned = pd.DataFrame(0, index=[0], columns=expected_features)
    for col in df_encoded.columns:
        if col in aligned.columns:
            aligned[col] = df_encoded[col]
            
    tau = float(model.predict_uplift(aligned)[0])
    
    if tau < -0.005:
        segment = "Sleeping Dog"
        action = "DO NOT CONTACT (Outreach causes net churn/irritation)"
        priority = "Excluded"
        inc_val = 0.0
    elif tau >= 0.020:
        segment = "Persuadable"
        action = "Send 20% Retention Promo (Highest ROI opportunity)"
        priority = "Tier 1 High Priority"
        inc_val = round(tau * 80.0, 2)
    elif tau >= 0.008:
        segment = "Persuadable (Moderate)"
        action = "Send Light Engagement Email"
        priority = "Tier 2 Medium Priority"
        inc_val = round(tau * 80.0, 2)
    elif customer.history > 300:
        segment = "Sure Thing"
        action = "Organic Buyer: No discount needed (Save marketing cost)"
        priority = "Monitor Only"
        inc_val = 0.0
    else:
        segment = "Lost Cause"
        action = "Unresponsive: Do not spend marketing budget"
        priority = "Excluded"
        inc_val = 0.0
        
    return cid, tau, segment, action, priority, inc_val


@app.post("/predict-uplift", response_model=PrescriptiveDecision)
def predict_uplift(customer: CustomerProfile):
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not yet loaded.")
        
    cid, tau, segment, action, priority, inc_val = _infer_tau_and_segment(customer)
    return PrescriptiveDecision(
        customer_id=cid,
        predicted_uplift=round(tau, 4),
        segment=segment,
        recommended_action=action,
        priority_level=priority,
        expected_incremental_value=inc_val
    )


@app.post("/copilot-brief", response_model=CausalCopilotBrief)
def copilot_brief(customer: CustomerProfile):
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not yet loaded.")
        
    cid, tau, segment, _, _, _ = _infer_tau_and_segment(customer)
    brief_data = generate_causal_brief(customer.dict(), tau=tau, segment=segment)
    return CausalCopilotBrief(**brief_data)


@app.post("/conformal-predict")
def conformal_predict(customer: CustomerProfile):
    """
    Computes 90% guaranteed Conformal Prediction intervals [tau_lower, tau_upper].
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not yet loaded.")
        
    cid, tau, _, _, _, _ = _infer_tau_and_segment(customer)
    intervals = conformal_engine.predict_intervals(tau)
    intervals["customer_id"] = cid
    return intervals


@app.post("/multi-treatment-opt")
def multi_treatment_opt(customer: CustomerProfile):
    """
    Solves optimal offer assignment across [10%, 20%, Free Shipping, No Outreach].
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not yet loaded.")
        
    cid, tau, _, _, _, _ = _infer_tau_and_segment(customer)
    opt_result = multi_optimizer.optimize_offer(customer.dict(), base_tau=tau)
    opt_result["customer_id"] = cid
    return opt_result


@app.post("/what-if-simulate")
def what_if_simulate(customer: CustomerProfile):
    """
    Simulates counterfactual channel transitions and recency decay curves.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not yet loaded.")
        
    channels = whatif_sim.simulate_channel_scenarios(customer.dict())
    recourse = whatif_sim.find_optimal_recourse(customer.dict())
    return {
        "customer_id": customer.customer_id,
        "channel_scenarios": channels,
        "optimal_recourse": recourse
    }
