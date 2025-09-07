
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd

st.set_page_config(page_title="Parkinson's Model Explanation", layout="centered")

st.markdown("##  Parkinson's Model Explanation Dashboard")
st.markdown("This dashboard visualizes SHAP and LIME explanations for the trained severity classification model.")

# SHAP
st.markdown("###  SHAP Summary (Feature Importance)")
st.image("output/shap_summary.png")

st.markdown("###  SHAP Beeswarm Plot")
st.image("output/shap_beeswarm.png")

st.markdown("###  SHAP Misclassified Samples")
st.image("output/shap_misclassified.png")

# LIME Fixed Samples
st.markdown("###  LIME Explanations for 3 Fixed Samples")
for i in range(1, 4):
    st.markdown(f"#### LIME Explanation for Test Sample {i}")
    with open(f"output/lime_explanation_{i}.html", "r", encoding="utf-8") as f:
        components.html(f.read(), height=450, scrolling=True)

# LIME Interactive via subject ID + test_time
st.markdown("###  LIME Explanation by Subject ID and Test Time")

df_test = pd.read_csv("data/test.csv")

subject_options = df_test["subject#"].unique()
selected_subject = st.selectbox("Select Subject ID", sorted(subject_options))

test_times = df_test[df_test["subject#"] == selected_subject]["test_time"].unique()
selected_time = st.selectbox("Select Test Time", sorted(test_times))

if st.button("Generate LIME for Selected Record"):
    # Locate the correct row index
    selected_index = df_test[
        (df_test["subject#"] == selected_subject) &
        (df_test["test_time"] == selected_time)
    ].index[0]

    from predict import load_model_and_transform
    from visual_outputs import save_lime_explanation

    model, X_proc = load_model_and_transform("data/test.csv")
    output_path = "output/lime_user_selected.html"
    save_lime_explanation(model.named_steps["clf"], X_proc, index=selected_index, output_path=output_path)

    with open(output_path, "r", encoding="utf-8") as f:
        components.html(f.read(), height=600, scrolling=True)
