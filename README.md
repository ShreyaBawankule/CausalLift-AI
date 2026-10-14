# Causal Uplift Modeling & Prescriptive Retention Platform
## Final Year Honors Project in Data Science & Decision Intelligence

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-FF4B4B.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📌 Executive Summary

Traditional customer retention and marketing workflows rely on **predictive machine learning** (e.g., standard churn classification $P(Y=1 \mid X)$). This approach has a fundamental flaw: **high risk does not imply high responsiveness**. Companies routinely waste millions of dollars offering discounts to customers who would have stayed anyway (*Sure Things*) or provoking inactive users into cancelling (*Sleeping Dogs*).

This project implements an end-to-end **Causal Uplift Modeling System** that estimates the **Conditional Average Treatment Effect (CATE)**:

$$\tau(X) = \mathbb{E}[Y(1) - Y(0) \mid X]$$

By isolating true incremental elasticity, this system prescribes optimal interventions for individual users, maximizing return-on-investment (ROI) while mathematically protecting the marketing budget.

---

## 🎯 The Four Causal Quadrants

```
                    Treatment Sensitivity (Uplift tau)
                                    ▲
                                    │
               Persuadables         │         Sure Things
          (TARGET IMMEDIATELY)      │    (DO NOT WASTE PROMO)
      High Incremental Benefit      │    Organic conversions
                                    │
────────────────────────────────────┼────────────────────────────────► Baseline Prob P(Y|0)
                                    │
               Lost Causes          │        Sleeping Dogs
             (DO NOT TARGET)        │     (AVOID CONTACTING)
       Unresponsive to outreach     │   Contact causes negative churn
                                    │
```

1. **Persuadables ($\tau > 0$):** Convert *only* when given an offer. Primary target.
2. **Sure Things ($\tau \approx 0, Y=1$):** Will convert organically without discounts.
3. **Lost Causes ($\tau \approx 0, Y=0$):** Will not convert regardless of outreach.
4. **Sleeping Dogs ($\tau < 0$):** Outreach triggers adverse reactions (e.g., subscription cancellation).

---

## 🔬 Scientific Methodology & Architecture

### 1. Causal Diagnostics
* **Positivity / Common Support:** Verifies that $0 < P(W=1 \mid X) < 1$ across all feature strata.
* **Covariate Balance (SMD):** Standardized Mean Differences computed across treated and control populations to verify unconfoundedness ($\text{SMD} < 0.10$).

### 2. Meta-Learner Implementations
* **S-Learner (Single Model):** Trains a unified LightGBM model on augmented feature matrix $[X, W]$.
* **T-Learner (Two Models):** Trains independent models for Treatment and Control groups.
* **X-Learner (Künzel et al. 2019):** Two-stage counterfactual imputation designed for imbalanced treatment ratios.

### 3. Uplift Evaluation Framework
Because individual counterfactuals are unobservable in real life:
* **Qini Curve & Qini Score:** Measures cumulative incremental conversions over a random targeting baseline.
* **Uplift Deciles:** Empirical lift verification across predicted deciles.
### 4. The Causal-GenAI Copilot (GenAI + Causal Fusion)
* **Prescriptive Counterfactual Reasoner:** Translates numerical treatment effects $\tau(X)$ into executive narratives explaining potential outcomes $Y(1)$ vs. $Y(0)$.
* **Dynamic Ad Creative Generator:** Uses customer-level category preferences and elasticity to generate personalized email subject lines, body copy, and discount codes in real time.

### 5. Conformal Prediction (Uncertainty Quantification)
* **Finite-Sample Coverage Guarantees:** Split conformal prediction computes distribution-free 90% confidence intervals $[\tau_{\text{lower}}, \tau_{\text{upper}}]$.
* **Statistical Risk Tiers:** Flags *Guaranteed Persuadables* ($\tau_{\text{lower}} > 0$) versus *Volatile Customers* whose confidence bands cross zero.

### 6. Multi-Offer Policy Optimizer
* **Constrained Economic Allocation:** Solves $\arg\max_a [\tau_a(X) \times \text{LTV} - \text{Cost}(a)]$ across 4 competing actions: 10% Off, 20% Off, Free Shipping, or Organic Outreach.

### 7. Counterfactual 'What-If' Recourse Simulator
* **Actionable Algorithmic Recourse:** Simulates how transitions across shopping channels (Phone $\to$ Web $\to$ Multichannel) and purchase recency decay curves shift customer elasticity.

---

## 🛠️ Repository Structure

```text
├── data/
│   ├── raw/                      # Downloaded MineThatData experiment dataset (64k rows)
│   └── processed/                # Preprocessed feature sets
├── src/
│   ├── data_loader.py            # Dataset acquisition & stratified splitting
│   ├── causal_validation.py      # Positivity & SMD balance diagnostics
│   ├── meta_learners.py          # S-Learner, T-Learner, X-Learner implementations
│   ├── evaluation.py             # Qini curves, AUUC, deciles, ROI simulations
│   ├── explainability.py         # Quadrant segmentation & Policy Tree rules
│   ├── causal_copilot.py         # GenAI Counterfactual Reasoner & Dynamic Copy Generator
│   ├── conformal_uplift.py       # 90% Conformal Prediction uncertainty intervals
│   ├── multi_treatment_optimizer.py # Multi-offer profit maximization optimizer
│   ├── counterfactual_simulator.py  # 'What-If' channel and recency recourse engine
│   └── pipeline.py               # Automated end-to-end orchestration
├── app/
│   ├── api.py                    # FastAPI microservice (6 REST endpoints)
│   └── dashboard.py              # Interactive Streamlit executive dashboard (8 Tabs)
├── models/                       # Serialized best model & feature schema
├── reports/                      # Benchmark CSVs, Qini curves, decision rules
├── test_api.py                   # Automated integration test suite (6/6 tests passing)
├── requirements.txt              # Production dependency specifications
└── README.md
```

---

## 🚀 Quickstart Guide

### 1. Environment Setup
```bash
# Clone the repository
git clone <repo-url>
cd HONORS_PROJECT_NEW

# Activate virtual environment
.\.venv\Scripts\activate   # Windows
# source .venv/bin/activate # Linux/macOS

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Full Pipeline
Executes data ingestion, causal validation, model training, benchmarking, and rule extraction:
```bash
python -m src.pipeline
```

### 3. Launch Interactive Dashboard
```bash
streamlit run app/dashboard.py
```

### 4. Launch FastAPI REST Service
```bash
uvicorn app.api:app --reload --port 8000
```
Interactive Swagger documentation available at: `http://localhost:8000/docs`.

---

## 💼 Recruiter & Defense Interview Highlights

* **Why not standard classification?** Standard classification predicts correlation, not causation. A high-probability churner might still churn after receiving an incentive (Lost Cause). Uplift models directly estimate the treatment effect $\frac{\partial Y}{\partial W}$.
* **How are counterfactuals evaluated?** Via Qini curves and Area Under Uplift Curve (AUUC) on holdout randomized test sets.
* **What is the business value?** Directly quantifiable in net margin lift: reducing wasted incentive spend on Sure Things and eliminating churn caused by Sleeping Dogs.
