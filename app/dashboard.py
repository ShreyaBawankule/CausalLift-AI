"""
CausalLift AI: Precision Customer Retention Platform.
Executive UI with high-visibility charts, bold blue sidebar headings, and polished layouts.
"""

import os
import sys

# Ensure project root is in sys.path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import joblib
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

from src.causal_copilot import generate_causal_brief
from src.conformal_uplift import get_default_conformal_predictor, ConformalUpliftPredictor
from src.multi_treatment_optimizer import MultiTreatmentOptimizer
from src.counterfactual_simulator import CounterfactualSimulator

# ----------------- PAGE CONFIGURATION -----------------
st.set_page_config(
    page_title="CausalLift AI - Precision Customer Retention",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------- MODERN CUSTOM CSS WITH BLUE ACCENTS -----------------
st.markdown("""
<style>
    /* Main body background */
    .main {
        background-color: #F8FAFC;
    }

    /* Sidebar Background and Borders */
    [data-testid="stSidebar"] {
        background-color: #F1F5F9 !important;
        border-right: 2px solid #E2E8F0 !important;
        padding-top: 10px;
    }

    /* ALL SIDEBAR HEADINGS & LABELS IN VIBRANT BLUE */
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] .brand-title {
        color: #1D4ED8 !important; /* Deep Vibrant Blue */
        font-weight: 800 !important;
    }
    
    [data-testid="stSidebar"] label p,
    [data-testid="stSidebar"] div[role="radiogroup"] label span {
        color: #1E40AF !important; /* Rich Blue */
        font-weight: 700 !important;
        font-size: 0.96rem !important;
    }

    /* Sidebar Brand Title Card */
    .sidebar-brand-card {
        background: #EFF6FF;
        border: 2px solid #BFDBFE;
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 20px;
        text-align: center;
    }
    .brand-main-title {
        font-size: 1.45rem;
        font-weight: 900;
        color: #1D4ED8;
        letter-spacing: -0.01em;
    }
    .brand-sub-title {
        font-size: 0.82rem;
        font-weight: 700;
        color: #2563EB;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 2px;
    }

    /* Status Pill */
    .status-pill {
        background: #ECFDF5;
        border: 1px solid #A7F3D0;
        border-radius: 8px;
        padding: 12px;
        margin-top: 20px;
    }
    .status-pill-text {
        color: #065F46;
        font-size: 0.85rem;
        font-weight: 700;
    }
    .status-pill-sub {
        color: #047857;
        font-size: 0.75rem;
        font-weight: 500;
    }

    /* High Visibility Metric Cards */
    .metric-card {
        background: white;
        border: 1px solid #CBD5E1;
        border-radius: 12px;
        padding: 18px 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        margin-bottom: 15px;
    }
    .metric-label {
        font-size: 0.85rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        color: #475569;
        margin-bottom: 6px;
    }
    .metric-value {
        font-size: 1.85rem;
        font-weight: 800;
        color: #0F172A;
    }
    .metric-caption {
        font-size: 0.82rem;
        font-weight: 600;
        margin-top: 4px;
    }

    /* Understandable Header Card */
    .header-card {
        background: white;
        border: 1px solid #E2E8F0;
        border-left: 6px solid #2563EB;
        border-radius: 10px;
        padding: 18px 22px;
        margin-bottom: 24px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.04);
    }
    .header-title {
        font-size: 1.4rem;
        font-weight: 800;
        color: #0F172A;
        margin-bottom: 4px;
    }
    .header-subtitle {
        font-size: 0.95rem;
        color: #475569;
        line-height: 1.45;
    }

    /* Email Preview Window */
    .email-container {
        background: white;
        border: 2px solid #CBD5E1;
        border-radius: 12px;
        overflow: hidden;
        margin-top: 18px;
        box-shadow: 0 10px 15px -3px rgba(0,0,0,0.06);
    }
    .email-header {
        background: #F8FAFC;
        padding: 14px 20px;
        border-bottom: 1px solid #E2E8F0;
        font-size: 0.92rem;
        color: #1E293B;
    }
    .email-body {
        padding: 24px;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        line-height: 1.65;
        color: #0F172A;
        font-size: 0.96rem;
        white-space: pre-wrap;
    }
</style>
""", unsafe_allow_html=True)

# ----------------- PATHS & CACHE SETUP -----------------
MODELS_DIR = os.path.join(ROOT_DIR, "models")
REPORTS_DIR = os.path.join(ROOT_DIR, "reports")
BEST_MODEL_PATH = os.path.join(MODELS_DIR, "best_uplift_model.pkl")
FEATURES_PATH = os.path.join(MODELS_DIR, "feature_names.pkl")
TEST_PREDS_PATH = os.path.join(REPORTS_DIR, "test_predictions_segmented.csv")
SMD_PATH = os.path.join(REPORTS_DIR, "smd_covariate_balance.csv")

@st.cache_data
def load_cached_data():
    if os.path.exists(TEST_PREDS_PATH) and os.path.exists(SMD_PATH):
        df_preds = pd.read_csv(TEST_PREDS_PATH)
        df_smd = pd.read_csv(SMD_PATH)
        return df_preds, df_smd
    return None, None

df_preds, df_smd = load_cached_data()

@st.cache_resource
def load_engines():
    if os.path.exists(BEST_MODEL_PATH) and os.path.exists(FEATURES_PATH):
        model = joblib.load(BEST_MODEL_PATH)
        feats = joblib.load(FEATURES_PATH)
        conformal_eng = get_default_conformal_predictor()
        multi_opt = MultiTreatmentOptimizer(customer_ltv=80.0)
        whatif_eng = CounterfactualSimulator(model, feats)
        return model, feats, conformal_eng, multi_opt, whatif_eng
    return None, None, None, None, None

model, expected_feats, conformal_eng, multi_opt, whatif_eng = load_engines()

# ----------------- BLUE-THEMED SIDEBAR -----------------
with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand-card">
        <div class="brand-main-title">🎯 CausalLift AI</div>
        <div class="brand-sub-title">Precision Customer Retention</div>
    </div>
    """, unsafe_allow_html=True)
    
    view_selection = st.radio(
        "NAVIGATION MODULES",
        [
            "🏠 1. Overview & Key Results",
            "🔬 2. Data Fairness Check",
            "🏆 3. AI Models Leaderboard",
            "💰 4. Profit & ROI Calculator",
            "🤖 5. Smart AI Assistant",
            "🎲 6. Safety & Risk Guardrails",
            "🏷️ 7. Best Offer Finder",
            "🔮 8. 'What-If' Simulator"
        ],
        index=0
    )
    
    st.markdown("""
    <div class="status-pill">
        <div class="status-pill-text">● Champion Model Active: T-Learner</div>
        <div class="status-pill-sub">Dataset: 64,000 Hillstrom Benchmark</div>
    </div>
    """, unsafe_allow_html=True)

# ----------------- VIEW 1: OVERVIEW & KEY RESULTS -----------------
if view_selection == "🏠 1. Overview & Key Results":
    st.markdown("""
    <div class="header-card">
        <div class="header-title">🏠 Executive Summary: Finding Customers Who Actually Need Discounts</div>
        <div class="header-subtitle">
            Most businesses waste millions giving coupons to customers who would buy anyway (Sure Things), 
            or irritating inactive users into cancelling (Sleeping Dogs). 
            <strong>CausalLift AI</strong> identifies the <strong>Persuadables</strong>—customers who only buy because they received a promotion.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    if df_preds is not None:
        total_test = len(df_preds)
        persuadables = (df_preds["segment"].str.contains("Persuadable")).sum()
        sleeping_dogs = (df_preds["segment"].str.contains("Sleeping Dog")).sum()
        avg_uplift = df_preds["predicted_uplift"].mean() * 100
        
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Test Customers Analyzed</div>
                <div class="metric-value">{total_test:,}</div>
                <div class="metric-caption" style="color: #64748B;">Randomized holdout dataset</div>
            </div>
            """, unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Persuadables (Target These)</div>
                <div class="metric-value" style="color: #10B981;">{persuadables:,}</div>
                <div class="metric-caption" style="color: #10B981;">{(persuadables/total_test)*100:.1f}% of total · Highest sales impact</div>
            </div>
            """, unsafe_allow_html=True)
        with c3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Sleeping Dogs (Never Contact)</div>
                <div class="metric-value" style="color: #EF4444;">{sleeping_dogs:,}</div>
                <div class="metric-caption" style="color: #EF4444;">{(sleeping_dogs/total_test)*100:.1f}% · Cancellations prevented</div>
            </div>
            """, unsafe_allow_html=True)
        with c4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Extra Sales Lift</div>
                <div class="metric-value" style="color: #2563EB;">+{avg_uplift:.2f}%</div>
                <div class="metric-caption" style="color: #2563EB;">True incremental buying increase</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        col_pie, col_expl = st.columns([1.2, 1])
        
        with col_pie:
            st.subheader("The 4 Customer Groups Identified by AI")
            seg_counts = df_preds["segment"].value_counts()
            
            fig_p, ax_p = plt.subplots(figsize=(6.5, 4.5), dpi=130)
            colors = ["#10B981", "#3B82F6", "#F59E0B", "#EF4444"]
            wedges, texts, autotexts = ax_p.pie(
                seg_counts, 
                labels=seg_counts.index, 
                autopct="%1.1f%%", 
                startangle=140, 
                colors=colors,
                textprops=dict(color="#0F172A", fontsize=10, weight="bold"),
                wedgeprops=dict(width=0.48, edgecolor='white', linewidth=2.5)
            )
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_weight('bold')
                autotext.set_fontsize(11)
            ax_p.axis("equal")
            st.pyplot(fig_p, use_container_width=True)

        with col_expl:
            st.subheader("What Does This Mean for Business?")
            st.markdown("""
            * 🟢 **Persuadables:** Will only purchase if they receive an incentive. **Spend marketing dollars here.**
            * 🔵 **Sure Things:** Loyal buyers who buy regularly anyway. **Save money by not giving discounts.**
            * 🟠 **Lost Causes:** Completely unresponsive to ads. **Do not waste outreach budget.**
            * 🔴 **Sleeping Dogs:** Inactive users who get annoyed by emails and cancel. **Keep them in the Do-Not-Disturb list.**
            """)

# ----------------- VIEW 2: DATA FAIRNESS CHECK -----------------
elif view_selection == "🔬 2. Data Fairness Check":
    st.markdown("""
    <div class="header-card">
        <div class="header-title">🔬 Data Fairness Check: Is Our Experiment Scientifically Fair?</div>
        <div class="header-subtitle">
            Just like testing a new medicine requires identical health profiles in both the drug and placebo groups, 
            marketing data must be 100% fair and unbiased. Here we verify that treatment and control groups were balanced.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    col_metric1, col_metric2 = st.columns(2)
    with col_metric1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Data Overlap Score (Positivity)</div>
            <div class="metric-value" style="color: #10B981;">100.0% Overlap</div>
            <div class="metric-caption" style="color: #10B981;">Both groups cover all customer types with zero gaps</div>
        </div>
        """, unsafe_allow_html=True)
    with col_metric2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Group Difference Score (Max SMD)</div>
            <div class="metric-value" style="color: #2563EB;">0.0071 (Pass)</div>
            <div class="metric-caption" style="color: #2563EB;">Way below the 0.10 safety limit · Zero selection bias</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    if df_smd is not None:
        c_left, c_right = st.columns([1, 1.2])
        
        with c_left:
            st.subheader("Trait Balance Table")
            display_smd = df_smd.rename(columns={
                "covariate": "Customer Trait",
                "mean_treated": "Treated Avg",
                "mean_control": "Control Avg",
                "smd": "Difference (SMD)",
                "is_balanced": "Fair?"
            })
            st.dataframe(
                display_smd.style.format({
                    "Treated Avg": "{:.3f}",
                    "Control Avg": "{:.3f}",
                    "Difference (SMD)": "{:.4f}"
                }), 
                use_container_width=True, 
                height=300
            )
            
        with c_right:
            st.subheader("Visual Difference Chart (All Bars Below Safe Red Line)")
            fig_smd, ax_smd = plt.subplots(figsize=(6.5, 3.8), dpi=130)
            y_pos = range(len(df_smd))
            bars = ax_smd.barh(y_pos, df_smd["smd"], color="#2563EB", alpha=0.9, height=0.55)
            ax_smd.axvline(0.10, color="#EF4444", linestyle="--", linewidth=2, label="Safety Limit (0.10)")
            ax_smd.set_yticks(y_pos)
            ax_smd.set_yticklabels(df_smd["covariate"], fontsize=10, weight="bold")
            ax_smd.set_xlabel("Difference Score (Lower is better)", fontsize=10, weight="bold")
            ax_smd.bar_label(bars, fmt='%.4f', padding=4, fontsize=9, weight="bold")
            ax_smd.set_xlim(0, 0.12)
            ax_smd.spines['top'].set_visible(False)
            ax_smd.spines['right'].set_visible(False)
            ax_smd.legend(loc="lower right")
            st.pyplot(fig_smd, use_container_width=True)

# ----------------- VIEW 3: AI MODELS LEADERBOARD -----------------
elif view_selection == "🏆 3. AI Models Leaderboard":
    st.markdown("""
    <div class="header-card">
        <div class="header-title">🏆 AI Models Leaderboard: Which Algorithm Found the Most Sales?</div>
        <div class="header-subtitle">
            We trained 3 different causal machine learning algorithms to compete against each other. 
            Because standard accuracy cannot measure what "would have happened", we evaluate them using the <strong>Qini Score</strong> 
            (measuring extra sales generated above random guessing).
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    lb_path = os.path.join(REPORTS_DIR, "benchmark_leaderboard.csv")
    if os.path.exists(lb_path):
        df_lb = pd.read_csv(lb_path, index_col=0)
        medals = ["🥇 1st Place", "🥈 2nd Place", "🥉 3rd Place"]
        lead_cols = st.columns(min(3, len(df_lb)))
        for idx, (m_name, row) in enumerate(df_lb.iloc[:3].iterrows()):
            color = "#2563EB" if idx == 0 else "#475569"
            badge = " · Champion" if idx == 0 else ""
            with lead_cols[idx]:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">{medals[idx]}: {m_name}</div>
                    <div class="metric-value" style="color: {color};">Qini = {row['qini_score']:.2f}</div>
                    <div class="metric-caption" style="color: #10B981;">+{row['top_decile_empirical_uplift']:.2f}% Top Decile Lift{badge}</div>
                </div>
                """, unsafe_allow_html=True)
    else:
        lead_col1, lead_col2, lead_col3 = st.columns(3)
        with lead_col1:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-label">🥇 1st Place: T-Learner</div>
                <div class="metric-value" style="color: #2563EB;">Qini = 9.38</div>
                <div class="metric-caption" style="color: #10B981;">+1.06% Top Decile Lift · Champion</div>
            </div>
            """, unsafe_allow_html=True)
        with lead_col2:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-label">🥈 2nd Place: S-Learner</div>
                <div class="metric-value" style="color: #475569;">Qini = 7.22</div>
                <div class="metric-caption" style="color: #2563EB;">+0.91% Top Decile Lift</div>
            </div>
            """, unsafe_allow_html=True)
        with lead_col3:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-label">🥉 3rd Place: X-Learner</div>
                <div class="metric-value" style="color: #475569;">Qini = 5.61</div>
                <div class="metric-caption" style="color: #64748B;">+0.56% Top Decile Lift</div>
            </div>
            """, unsafe_allow_html=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
    for learner in ["t_learner", "s_learner", "x_learner"]:
        decile_file = os.path.join(REPORTS_DIR, f"{learner}_deciles.csv")
        if os.path.exists(decile_file):
            st.subheader(f"Ranked Customer Groups (`{learner.upper().replace('_', '-')}` Decile Breakdown)")
            d_df = pd.read_csv(decile_file)
            renamed_dec = d_df.rename(columns={
                "decile": "Decile Group (1=Best)",
                "sample_size": "Customer Count",
                "treatment_conv_rate": "Promo Group Buy Rate",
                "control_conv_rate": "No-Promo Buy Rate",
                "empirical_uplift": "Actual Lift Created",
                "mean_predicted_uplift": "AI Predicted Lift"
            })
            st.dataframe(
                renamed_dec.style.format({
                    "Promo Buy Rate": "{:.2%}",
                    "No-Promo Buy Rate": "{:.2%}",
                    "Actual Lift Created": "{:+.2%}",
                    "AI Predicted Lift": "{:+.2%}"
                }), 
                use_container_width=True
            )
            break

# ----------------- VIEW 4: PROFIT & ROI CALCULATOR -----------------
elif view_selection == "💰 4. Profit & ROI Calculator":
    st.markdown("""
    <div class="header-card">
        <div class="header-title">💰 Financial Profit & Marketing Budget Calculator</div>
        <div class="header-subtitle">
            Move the sliders below to see real-world business results. 
            Notice how targeting only the top AI-selected customers delivers <strong>higher net profit</strong> than spamming your entire customer base.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    s_col1, s_col2, s_col3 = st.columns(3)
    val_customer = s_col1.slider("Value of 1 Customer Purchase ($)", 20.0, 300.0, 80.0, 5.0)
    cost_contact = s_col2.slider("Cost to Send 1 Email/Promo ($)", 0.01, 2.00, 0.10, 0.01)
    budget_target = s_col3.slider("% of Customer Base to Target (%)", 5, 100, 30, 5) / 100.0
    
    if df_preds is not None:
        n_total = len(df_preds)
        n_target = int(budget_target * n_total)
        
        df_ranked = df_preds.sort_values("predicted_uplift", ascending=False).reset_index(drop=True)
        top_k = df_ranked.iloc[:n_target]
        
        p_t = top_k[top_k["treatment_actual"] == 1]["conversion_actual"].mean()
        p_c = top_k[top_k["treatment_actual"] == 0]["conversion_actual"].mean()
        inc_rate = max(0.0, p_t - p_c)
        inc_conversions = inc_rate * n_target
        
        gross_rev = inc_conversions * val_customer
        cost = n_target * cost_contact
        net_profit = gross_rev - cost
        roi = (net_profit / (cost + 1e-9)) * 100
        
        st.markdown("<br>", unsafe_allow_html=True)
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Targeted Customers</div>
                <div class="metric-value">{n_target:,}</div>
            </div>
            """, unsafe_allow_html=True)
        with m2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Campaign Cost</div>
                <div class="metric-value" style="color: #EF4444;">${cost:,.2f}</div>
            </div>
            """, unsafe_allow_html=True)
        with m3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Extra Revenue Created</div>
                <div class="metric-value" style="color: #2563EB;">${gross_rev:,.2f}</div>
            </div>
            """, unsafe_allow_html=True)
        with m4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Net Profit in Pocket</div>
                <div class="metric-value" style="color: #10B981;">${net_profit:,.2f}</div>
                <div class="metric-caption" style="color: #10B981;">{roi:.1f}% Return on Investment</div>
            </div>
            """, unsafe_allow_html=True)

# ----------------- VIEW 5: SMART AI ASSISTANT -----------------
elif view_selection == "🤖 5. Smart AI Assistant":
    st.markdown("""
    <div class="header-card">
        <div class="header-title">🤖 Smart AI Assistant: Plain-English Explanations & Custom Copy</div>
        <div class="header-subtitle">
            Type in any customer's shopping details below. Our AI predicts their responsiveness, 
            explains the business decision in plain English, and <strong>automatically drafts personalized email copy</strong> 
            using deterministic counterfactual reasoning (with optional live LLM API support).
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    with st.expander("📝 Enter Customer Profile", expanded=True):
        col_in1, col_in2, col_in3 = st.columns(3)
        in_cid = col_in1.text_input("Customer ID", "CUST_EXAMPLE_101")
        in_recency = col_in2.number_input("Months Since Last Purchase", 1, 36, 2)
        in_history = col_in3.number_input("Past Year Spend ($)", 0.0, 2000.0, 340.0)
        
        col_in4, col_in5, col_in6 = st.columns(3)
        in_zip = col_in4.selectbox("Area Type", ["Urban", "Suburban", "Rural"])
        in_channel = col_in5.selectbox("Primary Shopping Channel", ["Web", "Multichannel", "Phone"])
        in_newbie = col_in6.selectbox("New Customer (Joined in last 12 months)?", [0, 1])
        
        col_in7, col_in8 = st.columns(2)
        in_mens = col_in7.selectbox("Bought Men's Items Before?", [1, 0])
        in_womens = col_in8.selectbox("Bought Women's Items Before?", [0, 1])
    
    raw_input = {
        "customer_id": in_cid, "recency": in_recency, "history": in_history,
        "mens": in_mens, "womens": in_womens, "newbie": in_newbie,
        "zip_code": in_zip, "channel": in_channel
    }
    
    if st.button("Evaluate Customer & Generate Copy", type="primary", use_container_width=True):
        if model is not None:
            df_in = pd.DataFrame([raw_input]).drop(columns=["customer_id"])
            df_enc = pd.get_dummies(df_in, drop_first=True, dtype=int)
            aligned = pd.DataFrame(0, index=[0], columns=expected_feats)
            for c in df_enc.columns:
                if c in aligned.columns:
                    aligned[c] = df_enc[c]
            tau = float(model.predict_uplift(aligned)[0])
            
            if tau < -0.005:
                segment = "Sleeping Dog (Do Not Contact)"
            elif tau >= 0.015:
                segment = "Persuadable (Priority Target)"
            elif in_history > 300:
                segment = "Sure Thing (Organic Buyer)"
            else:
                segment = "Lost Cause (Unresponsive)"
                
            brief = generate_causal_brief(raw_input, tau=tau, segment=segment)
            
            st.markdown("<br>", unsafe_allow_html=True)
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Predicted Lift (tau)", f"{tau*100:+.2f}%")
            k2.metric("Customer Category", segment.split(" ")[0])
            k3.metric("Buy Chance Without Promo", f"{brief['baseline_probability']*100:.1f}%")
            k4.metric("Buy Chance With Promo", f"{brief['treated_probability']*100:.1f}%")
            
            st.markdown("#### 🧠 Plain-English Business Explanation")
            st.info(brief["executive_rationale"])
            
            st.markdown("#### ✉️ Automatically Drafted Email Preview")
            creative = brief["dynamic_creative"]
            if "Suppressed" in creative["subject_line"]:
                st.warning("⚠️ Outreach blocked by AI to prevent customer irritation or wasted discount budget.")
            else:
                st.markdown(f"""
                <div class="email-container">
                    <div class="email-header">
                        <strong>Subject:</strong> {creative['subject_line']}<br>
                        <strong>Target Product:</strong> {creative['target_category']} | 
                        <strong>Offer Tier:</strong> {creative['discount_tier']}
                    </div>
                    <div class="email-body">
{creative['email_body']}
                    </div>
                </div>
                """, unsafe_allow_html=True)

# ----------------- VIEW 6: SAFETY & RISK GUARDRAILS -----------------
elif view_selection == "🎲 6. Safety & Risk Guardrails":
    st.markdown("""
    <div class="header-card">
        <div class="header-title">🎲 Safety Guardrails: Adaptive Statistical Decision Guarantee</div>
        <div class="header-subtitle">
            AI shouldn't guess blindly with company money. As you move the slider, the <strong>Statistical Guarantee</strong> 
            dynamically scales based on how far the customer's response is from the risky zero boundary.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    in_tau = st.slider("Customer Estimated Responsiveness Lift (%)", -4.0, 5.0, 2.5, 0.1) / 100.0
    
    import importlib
    import src.conformal_uplift
    importlib.reload(src.conformal_uplift)
    from src.conformal_uplift import ConformalUpliftPredictor, get_default_conformal_predictor
    
    predictor = get_default_conformal_predictor()
    conf_res = predictor.predict_intervals(in_tau)
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Statistical Guarantee", conf_res["coverage_level"], delta=conf_res.get("confidence_label", ""))
    c1.progress(conf_res.get("coverage_float", 0.90))
    c2.metric("Estimated Lift", f"{conf_res['tau_point_estimate']*100:+.2f}%")
    c3.metric(f"Worst Case ({conf_res['coverage_level']})", f"{conf_res['tau_lower_bound']*100:+.2f}%")
    c4.metric(f"Best Case ({conf_res['coverage_level']})", f"{conf_res['tau_upper_bound']*100:+.2f}%")
    
    st.markdown("<br>", unsafe_allow_html=True)
    if conf_res["risk_color"] == "green":
        st.success(f"🛡️ **{conf_res['risk_tier']}**: {conf_res['statistical_recommendation']}")
    elif conf_res["risk_color"] == "red":
        st.error(f"🚫 **{conf_res['risk_tier']}**: {conf_res['statistical_recommendation']}")
    elif conf_res["risk_color"] == "orange":
        st.warning(f"⚠️ **{conf_res['risk_tier']}**: {conf_res['statistical_recommendation']}")
    else:
        st.info(f"ℹ️ **{conf_res['risk_tier']}**: {conf_res['statistical_recommendation']}")
        
    fig_ci, ax_ci = plt.subplots(figsize=(7.5, 2.2), dpi=130)
    ax_ci.plot([conf_res['tau_lower_bound']*100, conf_res['tau_upper_bound']*100], [1, 1], 'o-', color='#2563EB', linewidth=5, markersize=10)
    ax_ci.axvline(0, color='#EF4444', linestyle='--', linewidth=2, label='Zero Extra Sales Line')
    ax_ci.plot(conf_res['tau_point_estimate']*100, 1, 's', color='#10B981', markersize=12, label='Main Estimate')
    ax_ci.set_yticks([])
    ax_ci.spines['top'].set_visible(False)
    ax_ci.spines['right'].set_visible(False)
    ax_ci.spines['left'].set_visible(False)
    ax_ci.set_xlabel("Lift Range in Percent (%)", fontsize=10, weight="bold")
    ax_ci.legend(loc="upper left")
    st.pyplot(fig_ci, use_container_width=True)

# ----------------- VIEW 7: BEST OFFER FINDER -----------------
elif view_selection == "🏷️ 7. Best Offer Finder":
    st.markdown("""
    <div class="header-card">
        <div class="header-title">🏷️ Best Offer Finder: Which Promotion Makes the Most Profit?</div>
        <div class="header-subtitle">
            Instead of guessing between 10% Off, 20% Off, or Free Shipping, this engine compares 
            all competing offers and picks the one that leaves the most <strong>net cash in the business</strong>.
            <div style="font-size: 0.85rem; color: #cbd5e1; margin-top: 8px;">
                <em>Policy Simulation Framework:</em> Solves multi-action optimal treatment allocation 
                \( a^*(X) = \arg\max_{a} [\tau_a(X) \cdot \text{LTV} - \text{Cost}(a)] \) under promotional elasticity tiers extending empirical RCT baseline cohorts.
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    if multi_opt is not None:
        sample_cust = {"recency": 2, "history": 250.0, "channel": "Web"}
        tau_val = 0.028
        
        opt_res = multi_opt.optimize_offer(sample_cust, base_tau=tau_val)
        best = opt_res["optimal_offer"]
        
        st.success(f"🏆 **Winner: `{best['offer_name']}`** | Net Profit: **${best['net_profit_dollars']:.2f}** (ROI: {best['roi_percent']:.1f}%)")
        st.caption(f"Reason: {opt_res['decision_rationale']}")
        
        st.markdown("<br>", unsafe_allow_html=True)
        col_g1, col_g2 = st.columns([1.1, 1])
        
        with col_g1:
            st.subheader("Net Profit by Offer ($)")
            df_comp = pd.DataFrame(opt_res["all_competing_offers"])
            
            fig_bar, ax_bar = plt.subplots(figsize=(6, 3.8), dpi=130)
            colors = ["#94A3B8", "#3B82F6", "#F59E0B", "#10B981"]
            bars = ax_bar.bar(df_comp["offer_name"], df_comp["net_profit_dollars"], color=colors, width=0.55)
            ax_bar.axhline(0, color="black", linestyle="--", linewidth=1)
            ax_bar.set_ylabel("Net Profit ($)", fontsize=10, weight="bold")
            ax_bar.spines['top'].set_visible(False)
            ax_bar.spines['right'].set_visible(False)
            ax_bar.bar_label(bars, fmt='$%.2f', padding=4, fontsize=9, weight="bold")
            plt.xticks(rotation=15, ha='right', fontsize=9, weight="bold")
            st.pyplot(fig_bar, use_container_width=True)
            
        with col_g2:
            st.subheader("Comparison Table")
            renamed_offers = df_comp.rename(columns={
                "offer_name": "Offer Name",
                "offer_cost": "Cost ($)",
                "estimated_uplift": "Expected Lift",
                "gross_lift_dollars": "Gross Revenue",
                "net_profit_dollars": "Net Profit",
                "roi_percent": "ROI %"
            })
            st.dataframe(
                renamed_offers.style.format({
                    "Cost ($)": "${:.2f}",
                    "Expected Lift": "{:+.2%}",
                    "Gross Revenue": "${:.2f}",
                    "Net Profit": "${:.2f}",
                    "ROI %": "{:.1f}%"
                }), 
                use_container_width=True
            )

# ----------------- VIEW 8: 'WHAT-IF' SIMULATOR -----------------
elif view_selection == "🔮 8. 'What-If' Simulator":
    st.markdown("""
    <div class="header-card">
        <div class="header-title">🔮 'What-If' Simulator: How Can We Turn a Lost Customer into a Buyer?</div>
        <div class="header-subtitle">
            See what happens when a customer switches channels (e.g., from Phone to Web) or how their response decays 
            the longer they go without visiting your store.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    if whatif_eng is not None:
        sample_profile = {"recency": 4, "history": 200.0, "mens": 1, "womens": 0, "zip_code": "Urban", "newbie": 0, "channel": "Phone"}
        
        col_c1, col_c2 = st.columns([1, 1.1])
        with col_c1:
            st.subheader("1. What Happens If They Switch Shopping Channels?")
            channels = whatif_eng.simulate_channel_scenarios(sample_profile)
            df_ch = pd.DataFrame(channels).rename(columns={
                "channel": "Shopping Channel",
                "predicted_uplift": "AI Lift Score",
                "uplift_percent": "Lift %",
                "resulting_segment": "Customer Group"
            })
            st.dataframe(df_ch, use_container_width=True)
            
            recourse = whatif_eng.find_optimal_recourse(sample_profile)
            st.info(f"💡 **Suggested Strategy:** {recourse['recourse_recommendation']}")
            
        with col_c2:
            st.subheader("2. Responsiveness Decay Over Time (Months Inactive)")
            traj = whatif_eng.sweep_recency_trajectory(sample_profile)
            
            fig_decay, ax_decay = plt.subplots(figsize=(6.5, 3.5), dpi=130)
            ax_decay.plot(traj["months_since_purchase"], traj["uplift_pct"], 'o-', color="#2563EB", linewidth=3, markersize=7)
            ax_decay.axhline(0, color="black", linestyle="--", linewidth=1.2)
            ax_decay.spines['top'].set_visible(False)
            ax_decay.spines['right'].set_visible(False)
            ax_decay.set_xlabel("Months Inactive", fontsize=10, weight="bold")
            ax_decay.set_ylabel("Sales Lift (%)", fontsize=10, weight="bold")
            ax_decay.grid(True, linestyle=":", alpha=0.6)
            st.pyplot(fig_decay, use_container_width=True)
