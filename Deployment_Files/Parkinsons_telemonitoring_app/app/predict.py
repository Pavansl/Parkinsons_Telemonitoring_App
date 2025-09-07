import argparse
import pandas as pd
import joblib
import os
from sklearn.metrics import classification_report
from app.preprocessing import encode_severity_labels
from app.load_data import load_and_validate_data
from app.visual_outputs import save_shap_summary, save_lime_explanation

def run_explainability(model, df_raw):
    import os
    import pandas as pd
    from app.visual_outputs import (
        save_shap_summary,
        save_shap_beeswarm,
        save_shap_misclassified_plot,
        save_lime_explanation,
    )
    from app.preprocessing import encode_severity_labels

    print("\n Running SHAP and LIME explainability...")

    # Transform data using full pipeline except final classifier
    X_proc = model[:-1].transform(df_raw.copy())

    # Recover column names after transformation
    X_named = model.named_steps["select"].transform(
        model.named_steps["engineer"].transform(df_raw.copy())
    )
    X_proc = pd.DataFrame(X_proc, columns=X_named.columns)

    # Run SHAP visualizations
    print("Generating SHAP summary plot...")
    save_shap_summary(model.named_steps["clf"], X_proc, "output/shap_summary.png")

    print("Generating SHAP beeswarm plot...")
    save_shap_beeswarm(model.named_steps["clf"], X_proc, "output/shap_beeswarm.png")

    # If UPDRS scores exist, compare misclassified
    if "motor_UPDRS" in df_raw.columns:
        y_true = encode_severity_labels(df_raw["motor_UPDRS"])
        y_pred = model.named_steps["clf"].predict(X_proc)
        print("Generating SHAP plot for misclassified test samples...")
        save_shap_misclassified_plot(model.named_steps["clf"], X_proc, y_true, y_pred, "output/shap_misclassified.png")

    # Run LIME for 3 samples
    print("Generating LIME explanation for 3 test samples...")
    for i, idx in enumerate([3653, 3482, 2775]):
        save_lime_explanation(
            model.named_steps["clf"],
            X_proc,
            index=idx % len(X_proc),
            output_path=f"output/lime_explanation_{i+1}.html"
        )




def predict_on_file(file_path):
    print(" Loading model...")
    model = joblib.load("model/logistic_model.pkl")
    print(" Model loaded.")

    # Load and validate test data
    df_raw = load_and_validate_data(file_path, require_label=False)
    print(f"Loaded: {file_path} | Rows: {df_raw.shape[0]}, Columns: {df_raw.shape[1]}")
    print("Validation complete.")

    # Predict using full pipeline
    y_pred = model.predict(df_raw)
    df_raw["Predicted_Severity"] = y_pred

    # Optional evaluation if ground truth exists
    if "motor_UPDRS" in df_raw.columns:
        y_true = encode_severity_labels(df_raw["motor_UPDRS"])
        print("\n Classification Report:")
        print(classification_report(y_true, y_pred))

    os.makedirs("output", exist_ok=True)
    df_raw.to_csv("output/predictions.csv", index=False)
    print(" Predictions saved to output/predictions.csv")

    # SHAP & LIME
    run_explainability(model, df_raw)
    
      
def load_model_and_transform(file_path):
    from app.load_data import load_and_validate_data
    model = joblib.load("model/logistic_model.pkl")
    df = load_and_validate_data(file_path)

    X_proc = model[:-1].transform(df.copy())
    X_named = model.named_steps["select"].transform(
        model.named_steps["engineer"].transform(df.copy())
    )
    X_proc = pd.DataFrame(X_proc, columns=X_named.columns)
    return model, X_proc



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, required=True, help="Path to test CSV file")
    args = parser.parse_args()
    predict_on_file(args.file)
