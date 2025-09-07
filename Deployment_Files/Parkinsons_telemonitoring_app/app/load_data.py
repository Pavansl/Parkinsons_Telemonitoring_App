import os
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit

REQUIRED_COLUMNS_BASE = [
    "age", "sex", "Jitter:RAP", "Shimmer:APQ3", "Shimmer:APQ11",
    "RPDE", "DFA", "PPE", "subject#"
]

# For training or evaluation
REQUIRED_COLUMNS_WITH_LABEL = REQUIRED_COLUMNS_BASE + ["motor_UPDRS"]


def load_and_validate_data(file_path, require_label=True):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} not found.")
    
    df = pd.read_csv(file_path)
    print(f"Loaded: {file_path} | Rows: {df.shape[0]}, Columns: {df.shape[1]}")
    
    required = REQUIRED_COLUMNS_WITH_LABEL if require_label else REQUIRED_COLUMNS_BASE
    missing_cols = [col for col in required if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    assert df["age"].between(30, 100).all(), "Invalid 'age' values"
    assert df["sex"].isin([0, 1]).all(), "Invalid 'sex' values"

    num_cols = df.select_dtypes(include=np.number).columns
    neg_mask = (df[num_cols] < 0)
    total_neg = neg_mask.sum().sum()
    if total_neg > 0:
        share = total_neg / df.shape[0]
        if share < 0.01:
            df = df[~neg_mask.any(axis=1)]
            print(f"Dropped {total_neg} rows with negative values (<1%)")
        else:
            print(f"Warning: High number of negatives ({total_neg}) — data may be noisy.")

    print("Validation complete.")
    return df


def split_and_save(df, test_size=0.2, random_state=42):
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    X = df.drop(columns=["motor_UPDRS"])
    y = df["motor_UPDRS"]
    groups = df["subject#"]
    train_idx, test_idx = next(gss.split(X, y, groups))
    df_train = df.iloc[train_idx].copy()
    df_test = df.iloc[test_idx].copy()
    os.makedirs("data", exist_ok=True)
    df_train.to_csv("data/train.csv", index=False)
    df_test.to_csv("data/test.csv", index=False)
    print("Split complete:")
    print(f" - Train: {df_train.shape}")
    print(f" - Test: {df_test.shape}")
    return df_train, df_test

def load_and_prepare_data(mode="split", file_path="data/full.csv"):
    df = load_and_validate_data(file_path)
    if mode == "split":
        return split_and_save(df)
    elif mode == "predict":
        return df
    else:
        raise ValueError("Invalid mode. Use 'split' or 'predict'.")
