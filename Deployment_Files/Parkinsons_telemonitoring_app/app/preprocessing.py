import pandas as pd
import numpy as np
from sklearn.preprocessing import FunctionTransformer, RobustScaler

# ---------- Feature Engineering ----------
# ---------- Feature Engineering ----------
def engineer_features(df):
    df = df.copy()

    # Derived features
    df["log_PPE"] = np.log1p(df["PPE"])
    df["Inv_JitterAbs"] = 1 / (df["Jitter(Abs)"] + 1e-6)
    df["Shimmer_Slope"] = df["Shimmer:APQ11"] - (df["Shimmer:APQ3"] + 1e-6)
    df["Age_Sex_Interaction"] = df["age"] * df["sex"]

    return df

engineer_transformer = FunctionTransformer(engineer_features, validate=False)

# ---------- Feature Selection ----------
def select_features(df):
    return df[[
        "age", "sex", "Shimmer_Slope", "Inv_JitterAbs",
        "log_PPE", "RPDE", "DFA", "Age_Sex_Interaction"
    ]]

select_features_transformer = FunctionTransformer(select_features, validate=False)

# ---------- Scaling ----------
def get_scaler():
    return RobustScaler()

# ---------- Severity Label Encoding ----------
def encode_severity_labels(updrs_series):
    """
    Convert UPDRS values into binary severity labels.
    0 = Mild (UPDRS < 21), 1 = Moderate+ (UPDRS >= 21)
    """
    #return (updrs_series >= 20).astype(int)
    return np.where(updrs_series < 21, 0, 1)
