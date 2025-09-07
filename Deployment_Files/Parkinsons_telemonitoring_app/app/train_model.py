import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, GroupShuffleSplit
from sklearn.metrics import classification_report
from app.preprocessing import engineer_transformer, select_features_transformer, get_scaler, encode_severity_labels
from app.load_data import load_and_prepare_data
import os
os.makedirs("/app/model", exist_ok=True)


# Validate and split the dataset using external function
df_train, df_test = load_and_prepare_data(mode="split", file_path="data/parkinsons_updrs.data")

# Encode labels AFTER split
y_train = encode_severity_labels(df_train["motor_UPDRS"])
y_test = encode_severity_labels(df_test["motor_UPDRS"])

# Build pipeline
pipe = Pipeline([
    ("engineer", engineer_transformer),
    ("select", select_features_transformer),
    ("scale", get_scaler()),
    ("clf", LogisticRegression(max_iter=1000))
])

# Hyperparameter tuning
grid = GridSearchCV(pipe, param_grid={"clf__C": [0.01, 0.1, 1]}, cv=5)
grid.fit(df_train, y_train)
print("Best Params:", grid.best_params_)

# Save best model
joblib.dump(grid.best_estimator_, "model/logistic_model.pkl")
print("Model saved to app/model/logistic_model.pkl")
