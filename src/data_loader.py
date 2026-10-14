"""
Data Loader & Preprocessing Module for Causal Uplift Modeling.
Supports:
1. Kevin Hillstrom's MineThatData E-Mail Analytics dataset.
2. Automated local caching.
3. Realistic offline fallback generator with ground-truth treatment effects.
"""

import os
import urllib.request
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any
from sklearn.model_selection import train_test_split


DATA_URL = "http://www.minethatdata.com/Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv"
RAW_DATA_PATH = os.path.join("data", "raw", "minethatdata_email.csv")
PROCESSED_DATA_PATH = os.path.join("data", "processed", "uplift_features.csv")


def download_or_generate_dataset(filepath: str = RAW_DATA_PATH) -> pd.DataFrame:
    """
    Downloads the Kevin Hillstrom dataset or generates a high-fidelity
    benchmark dataset with controlled heterogeneous treatment effects if offline.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    if os.path.exists(filepath):
        print(f"Loading existing raw dataset from {filepath}")
        return pd.read_csv(filepath)

    print(f"Attempting to download dataset from {DATA_URL}...")
    try:
        req = urllib.request.Request(
            DATA_URL, 
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        )
        with urllib.request.urlopen(req, timeout=10) as response, open(filepath, 'wb') as out_file:
            out_file.write(response.read())
        print(f"Successfully downloaded and saved to {filepath}")
        return pd.read_csv(filepath)
    except Exception as e:
        print(f"Direct download failed ({e}). Generating faithful benchmark dataset...")
        df = _generate_synthetic_benchmark(n_samples=64000)
        df.to_csv(filepath, index=False)
        print(f"Synthetic benchmark saved to {filepath}")
        return df


def _generate_synthetic_benchmark(n_samples: int = 64000, seed: int = 42) -> pd.DataFrame:
    """
    Generates a realistic e-commerce marketing experiment dataset with 64,000 customers.
    Embeds known heterogeneous treatment effects (Persuadables, Sure Things, Lost Causes, Sleeping Dogs).
    """
    np.random.seed(seed)
    
    recency = np.random.randint(1, 13, size=n_samples) # months since last purchase
    history = np.random.exponential(scale=200, size=n_samples) + 20 # past year spend
    mens = np.random.binomial(1, 0.55, size=n_samples)
    womens = np.random.binomial(1, 0.50, size=n_samples)
    zip_code = np.random.choice(["Urban", "Suburban", "Rural"], p=[0.4, 0.45, 0.15], size=n_samples)
    newbie = np.random.binomial(1, 0.45, size=n_samples)
    channel = np.random.choice(["Web", "Phone", "Multichannel"], p=[0.45, 0.45, 0.10], size=n_samples)
    
    # Treatment assignment: Randomized 1/3 split like Hillstrom's challenge
    segment = np.random.choice(["Mens E-Mail", "Womens E-Mail", "No E-Mail"], size=n_samples)
    is_treated = (segment != "No E-Mail").astype(int)
    
    # Latent baseline propensity to convert without treatment (Sure things & baseline)
    baseline_logit = (
        -4.2 
        - 0.15 * recency 
        + 0.003 * history 
        + 0.3 * (channel == "Multichannel") 
        + 0.2 * newbie
    )
    p_control = 1 / (1 + np.exp(-baseline_logit))
    
    # Heterogeneous Treatment Effect (Uplift)
    # Persuadables: High engagement (low recency, bought both men & women, multichannel)
    # Sleeping Dogs: High recency + phone only (disturbed by unwanted emails)
    uplift = (
        0.035 
        - 0.003 * recency 
        + 0.00005 * history 
        + 0.02 * (channel == "Multichannel")
        - 0.015 * (recency > 10) * (channel == "Phone") # Sleeping dogs
    )
    
    p_treatment = np.clip(p_control + uplift, 0.001, 0.999)
    p_final = np.where(is_treated == 1, p_treatment, p_control)
    conversion = np.random.binomial(1, p_final)
    visit = np.where(conversion == 1, 1, np.random.binomial(1, np.clip(p_final * 3.5, 0.01, 0.85)))
    spend = np.where(conversion == 1, np.random.gamma(shape=3.0, scale=35.0, size=n_samples), 0.0)

    df = pd.DataFrame({
        "recency": recency,
        "history": np.round(history, 2),
        "mens": mens,
        "womens": womens,
        "zip_code": zip_code,
        "newbie": newbie,
        "channel": channel,
        "segment": segment,
        "visit": visit,
        "conversion": conversion,
        "spend": np.round(spend, 2)
    })
    return df


def preprocess_data(
    df: pd.DataFrame, 
    treatment_segment: str = "all_emails",
    target_col: str = "conversion"
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Cleans and encodes features for uplift modeling.
    
    Returns:
        X (pd.DataFrame): Processed feature matrix.
        treatment (pd.Series): Binary treatment vector (1 = treated, 0 = control).
        outcome (pd.Series): Target conversion/spend vector.
    """
    df_clean = df.copy()
    
    # Treatment definition
    if treatment_segment == "all_emails":
        treatment = (df_clean["segment"] != "No E-Mail").astype(int)
    elif treatment_segment == "mens_email":
        df_clean = df_clean[df_clean["segment"].isin(["Mens E-Mail", "No E-Mail"])]
        treatment = (df_clean["segment"] == "Mens E-Mail").astype(int)
    elif treatment_segment == "womens_email":
        df_clean = df_clean[df_clean["segment"].isin(["Womens E-Mail", "No E-Mail"])]
        treatment = (df_clean["segment"] == "Womens E-Mail").astype(int)
    else:
        raise ValueError(f"Unknown treatment segment: {treatment_segment}")
        
    outcome = df_clean[target_col]
    
    # Feature columns
    drop_cols = ["segment", "visit", "conversion", "spend", "history_segment"]
    drop_cols = [c for c in drop_cols if c in df_clean.columns]
    feature_df = df_clean.drop(columns=drop_cols)
    
    # One-hot encode categorical features
    cat_cols = feature_df.select_dtypes(include=["object"]).columns.tolist()
    if cat_cols:
        feature_df = pd.get_dummies(feature_df, columns=cat_cols, drop_first=True, dtype=int)
        
    return feature_df, treatment, outcome


def prepare_train_test_split(
    X: pd.DataFrame, 
    treatment: pd.Series, 
    outcome: pd.Series, 
    test_size: float = 0.3, 
    seed: int = 42
) -> Dict[str, Any]:
    """
    Stratified split preserving treatment and outcome proportions.
    """
    # Create combined stratum for balanced splitting
    stratum = treatment.astype(str) + "_" + outcome.astype(str)
    
    (
        X_train, X_test,
        tr_train, tr_test,
        y_train, y_test
    ) = train_test_split(
        X, treatment, outcome, 
        test_size=test_size, 
        random_state=seed, 
        stratify=stratum
    )
    
    return {
        "X_train": X_train.reset_index(drop=True),
        "X_test": X_test.reset_index(drop=True),
        "treatment_train": tr_train.reset_index(drop=True),
        "treatment_test": tr_test.reset_index(drop=True),
        "outcome_train": y_train.reset_index(drop=True),
        "outcome_test": y_test.reset_index(drop=True)
    }


if __name__ == "__main__":
    print("Testing data loader module...")
    df = download_or_generate_dataset()
    print("Raw dataset shape:", df.shape)
    print("Raw dataset preview:\n", df.head(3))
    
    X, tr, y = preprocess_data(df)
    print("\nProcessed features shape:", X.shape)
    print("Treatment distribution:\n", tr.value_counts(normalize=True))
    print("Outcome distribution:\n", y.value_counts(normalize=True))
    
    splits = prepare_train_test_split(X, tr, y)
    print("\nTrain samples:", len(splits["X_train"]), "Test samples:", len(splits["X_test"]))
