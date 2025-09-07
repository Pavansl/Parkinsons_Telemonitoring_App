
import os
import shap
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def _safe_shap_slice(shap_values):
    if len(shap_values.shape) == 3:
        return shap_values[:, :, 1]
    return shap_values

def save_shap_summary(model, X, output_path):
    explainer = shap.Explainer(model, X, feature_names=X.columns)
    shap_values = explainer(X)
    shap_to_plot = _safe_shap_slice(shap_values)
    shap.plots.bar(shap_to_plot, show=False)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()

def save_shap_beeswarm(model, X, output_path):
    explainer = shap.Explainer(model, X, feature_names=X.columns)
    shap_values = explainer(X)
    shap_to_plot = _safe_shap_slice(shap_values)
    shap.plots.beeswarm(shap_to_plot, show=False)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()

def save_shap_misclassified_plot(model, X, y_true, y_pred, output_path):
    misclassified_indices = np.where(y_true != y_pred)[0]
    if len(misclassified_indices) == 0:
        print("No misclassified samples to plot.")
        return
    explainer = shap.Explainer(model, X, feature_names=X.columns)
    shap_values = explainer(X.iloc[misclassified_indices])
    shap_to_plot = _safe_shap_slice(shap_values)
    shap.plots.beeswarm(shap_to_plot, show=False)
    plt.title("SHAP Summary for Misclassified Test Samples")
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()

def save_lime_explanation(model, X, index=0, output_path="output/lime_explanation.html"):
    from lime.lime_tabular import LimeTabularExplainer

    print(f"Generating LIME explanation for Test Sample {index}...")
    explainer = LimeTabularExplainer(
        training_data=np.array(X),
        feature_names=X.columns.tolist(),
        class_names=["Mild", "Moderate+"],
        mode="classification"
    )

    exp = explainer.explain_instance(
        data_row=X.iloc[index],
        predict_fn=model.predict_proba
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    exp.save_to_file(output_path)
    print(f"LIME explanation saved to {output_path}")
