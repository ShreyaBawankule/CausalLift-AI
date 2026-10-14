"""
End-to-End Orchestration Pipeline for Causal Uplift Modeling.
Automates:
1. Data Ingestion & Preprocessing
2. Causal Assumptions Validation (Positivity & Balance)
3. Model Training (S-Learner, T-Learner, X-Learner)
4. Evaluation (Qini Score, AUUC, Deciles, Financial ROI)
5. Decision Policy Extraction
6. Model Persistence for API & Dashboard
"""

import os
import joblib
import json
import numpy as np
import pandas as pd
from typing import Dict, Any

from src.data_loader import (
    download_or_generate_dataset,
    preprocess_data,
    prepare_train_test_split
)
from src.causal_validation import run_full_causal_diagnostics
from src.meta_learners import train_all_learners
from src.evaluation import (
    calculate_qini_curve,
    compute_uplift_deciles,
    simulate_business_roi
)
from src.explainability import (
    segment_customers,
    extract_interpretable_policy_rules
)
from src.conformal_uplift import ConformalUpliftPredictor


MODELS_DIR = "models"
REPORTS_DIR = "reports"


def run_pipeline() -> Dict[str, Any]:
    print("=" * 60)
    print(">> RUNNING CAUSAL UPLIFT MODELING PIPELINE")
    print("=" * 60)
    
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)
    
    # 1. Ingestion
    print("\n[Step 1/6] Ingesting and preprocessing dataset...")
    raw_df = download_or_generate_dataset()
    X, treatment, outcome = preprocess_data(raw_df)
    print(f"Features: {X.shape[1]} | Samples: {len(X)}")
    
    # 2. Causal Diagnostics
    print("\n[Step 2/6] Executing Causal Diagnostics (Positivity & Balance)...")
    diag = run_full_causal_diagnostics(X, treatment)
    print(f"Common Support: {diag['common_support']['percent_in_common_support']:.1f}%")
    print(f"Max Standardized Mean Difference (SMD): {diag['max_smd']:.4f}")
    if diag['overall_causal_assumptions_passed']:
        print("[OK] Causal assumptions validated (Negligible selection bias).")
    else:
        print("[WARN] Some covariates show mild imbalance. Review SMD report.")
        
    diag['smd_table'].to_csv(os.path.join(REPORTS_DIR, "smd_covariate_balance.csv"), index=False)
    
    # 3. Stratified Train / Calibration / Test Split
    print("\n[Step 3/6] Generating Stratified Train (60%), Calibration (15%), Test (25%) Splits...")
    splits = prepare_train_test_split(X, treatment, outcome, test_size=0.40, seed=42)
    X_train, tr_train, y_train = splits["X_train"], splits["treatment_train"], splits["outcome_train"]
    
    holdout_splits = prepare_train_test_split(
        splits["X_test"], splits["treatment_test"], splits["outcome_test"], 
        test_size=0.625, seed=42
    )
    X_calib, tr_calib, y_calib = holdout_splits["X_train"], holdout_splits["treatment_train"], holdout_splits["outcome_train"]
    X_test, tr_test, y_test = holdout_splits["X_test"], holdout_splits["treatment_test"], holdout_splits["outcome_test"]
    print(f"Train samples: {len(X_train)} | Calibration samples: {len(X_calib)} | Test samples: {len(X_test)}")
    
    # 4. Model Training
    print("\n[Step 4/6] Training Meta-Learners (S-Learner, T-Learner, X-Learner)...")
    trained_models = train_all_learners(X_train, tr_train, y_train)
    
    # 5. Uplift Evaluation Benchmark
    print("\n[Step 5/6] Benchmarking Models on Test Set (Qini & Financial ROI)...")
    benchmark_results = {}
    qini_curves = {}
    
    for name, model in trained_models.items():
        preds = model.predict_uplift(X_test)
        qini = calculate_qini_curve(y_test.values, tr_test.values, preds)
        deciles = compute_uplift_deciles(y_test.values, tr_test.values, preds)
        roi_sim = simulate_business_roi(y_test.values, tr_test.values, preds, contact_cost=0.10)
        
        benchmark_results[name] = {
            "qini_score": round(qini["qini_score"], 2),
            "top_decile_empirical_uplift": round(float(deciles.iloc[0]["empirical_uplift"]) * 100, 2),
            "max_net_profit": round(float(roi_sim["uplift_net_profit"].max()), 2)
        }
        qini_curves[name] = qini
        
        deciles.to_csv(os.path.join(REPORTS_DIR, f"{name.lower().replace('-', '_')}_deciles.csv"), index=False)
        roi_sim.to_csv(os.path.join(REPORTS_DIR, f"{name.lower().replace('-', '_')}_roi_simulation.csv"), index=False)
        
    benchmark_summary = pd.DataFrame(benchmark_results).T.sort_values("qini_score", ascending=False)
    print("\n[RESULTS] Uplift Model Benchmark Leaderboard:")
    print(benchmark_summary.to_string())
    
    # Persist benchmark leaderboard for dashboard View 3
    benchmark_summary.to_csv(os.path.join(REPORTS_DIR, "benchmark_leaderboard.csv"))
    
    best_model_name = benchmark_summary.index[0]
    best_model = trained_models[best_model_name]
    print(f"\n[BEST] Selected model: {best_model_name}")
    
    # Save best model and artifacts
    joblib.dump(best_model, os.path.join(MODELS_DIR, "best_uplift_model.pkl"))
    joblib.dump(list(X.columns), os.path.join(MODELS_DIR, "feature_names.pkl"))
    
    # Calibrate and persist Conformal Prediction Engine
    print("\n[Conformal Calibration] Calibrating Conformal Uncertainty Predictor on Calibration Set...")
    conformal_predictor = ConformalUpliftPredictor().calibrate(X_calib, tr_calib.values, y_calib.values, best_model)
    joblib.dump(conformal_predictor, os.path.join(MODELS_DIR, "conformal_predictor.pkl"))
    print(f"[OK] Conformal predictor calibrated (q_ctrl={conformal_predictor.q_ctrl:.4f}, q_treat={conformal_predictor.q_treat:.4f}).")
    
    # 6. Explainability & Policy Extraction
    print("\n[Step 6/6] Extracting Prescriptive Decision Policy...")
    best_preds_test = best_model.predict_uplift(X_test)
    policy_info = extract_interpretable_policy_rules(X_test, best_preds_test, max_depth=3)
    
    with open(os.path.join(REPORTS_DIR, "policy_decision_rules.txt"), "w") as f:
        f.write(policy_info["policy_rules"])
        
    print("\nHuman-Readable Policy Rules Extracted (Snippet):")
    print("\n".join(policy_info["policy_rules"].splitlines()[:15]))
    
    # Save test dataset sample with segments for dashboard
    test_sample = X_test.copy()
    test_sample["treatment_actual"] = tr_test
    test_sample["conversion_actual"] = y_test
    test_sample["predicted_uplift"] = best_preds_test
    test_sample["segment"] = segment_customers(best_preds_test, np.full(len(best_preds_test), 0.03))
    test_sample.to_csv(os.path.join(REPORTS_DIR, "test_predictions_segmented.csv"), index=False)
    
    print("\n" + "=" * 60)
    print("[SUCCESS] PIPELINE EXECUTION COMPLETE! Artifacts stored in 'models/' and 'reports/'")
    print("=" * 60)
    
    return {
        "best_model_name": best_model_name,
        "leaderboard": benchmark_summary,
        "diagnostics": diag
    }


if __name__ == "__main__":
    run_pipeline()
