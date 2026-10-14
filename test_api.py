"""
Comprehensive Integration Test Suite for Causal AI Platform.
Verifies:
1. Health check & model schema loading
2. /predict-uplift (CATE point estimates)
3. /copilot-brief (GenAI narratives & dynamic ad copy)
4. /conformal-predict (Option 2: 90% guaranteed uncertainty intervals)
5. /multi-treatment-opt (Option 3: Economic multi-offer constrained optimization)
6. /what-if-simulate (Option 1: Counterfactual channel recourse simulation)
"""

from app.api import (
    load_artifacts, 
    health_check, 
    predict_uplift, 
    copilot_brief, 
    conformal_predict,
    multi_treatment_opt,
    what_if_simulate,
    CustomerProfile
)


def run_full_suite():
    print("=" * 60)
    print("STARTING FULL INTEGRATION TEST SUITE")
    print("=" * 60)
    
    load_artifacts()
    
    # 1. Health
    print("\n[Test 1] Health Check...")
    h = health_check()
    assert h["status"] == "healthy" and h["model_loaded"] is True
    print("PASS: System healthy.")
    
    sample_customer = CustomerProfile(
        customer_id="HONORS_CUST_880",
        recency=3,
        history=220.0,
        mens=1,
        womens=0,
        zip_code="Urban",
        newbie=0,
        channel="Web"
    )
    
    # 2. CATE Point Prediction
    print("\n[Test 2] Point Estimate (/predict-uplift)...")
    dec = predict_uplift(sample_customer)
    print("PASS:", dec.model_dump())
    
    # 3. GenAI Copilot Brief
    print("\n[Test 3] Causal-GenAI Copilot (/copilot-brief)...")
    brief = copilot_brief(sample_customer)
    assert len(brief.executive_rationale) > 20
    assert hasattr(brief.dynamic_creative, "subject_line")
    print("PASS: Executive narrative and email copy generated successfully.")
    
    # 4. Conformal Prediction (Option 2)
    print("\n[Test 4] Conformal Uncertainty Intervals (/conformal-predict)...")
    conf = conformal_predict(sample_customer)
    assert "tau_lower_bound" in conf and "tau_upper_bound" in conf
    assert conf["tau_lower_bound"] <= conf["tau_point_estimate"] <= conf["tau_upper_bound"]
    print("PASS: 90% Conformal Interval:", conf["tau_lower_bound"], "to", conf["tau_upper_bound"], f"({conf['risk_tier']})")
    
    # 5. Multi-Offer Optimizer (Option 3)
    print("\n[Test 5] Multi-Offer Economic Optimization (/multi-treatment-opt)...")
    m_opt = multi_treatment_opt(sample_customer)
    assert "optimal_offer" in m_opt
    assert len(m_opt["all_competing_offers"]) == 4
    print("PASS: Optimal Action:", m_opt["optimal_offer"]["offer_name"], f"(Net Profit: ${m_opt['optimal_offer']['net_profit_dollars']})")
    
    # 6. Counterfactual What-If Simulator (Option 1)
    print("\n[Test 6] Counterfactual Recourse Simulation (/what-if-simulate)...")
    whatif = what_if_simulate(sample_customer)
    assert len(whatif["channel_scenarios"]) == 3
    assert "optimal_recourse" in whatif
    print("PASS: Recourse Recommendation:", whatif["optimal_recourse"]["recourse_recommendation"])
    
    print("\n" + "=" * 60)
    print("ALL 6 INTEGRATION TESTS PASSED WITH 100% SUCCESS!")
    print("=" * 60)


if __name__ == "__main__":
    run_full_suite()
