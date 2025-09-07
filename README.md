Parkinson’s Disease Telemonitoring Project – README
====================================================

Overview
--------
This project involves both Jupyter-based exploratory development and a fully containerized deployment system for predicting Parkinson’s Disease severity using voice features. It integrates interpretable machine learning models (Logistic Regression, Random Forest, XGBoost) with SHAP and LIME for transparency. The system is built to support both academic experimentation and real-world usability.

-----------------------------
Project Structure
-----------------------------

Sakleshpur_Lingaraju_00746907/
├── Parkinsons_Disease_Notebook.ipynb      ← Full Jupyter analysis
├── Parkinsons_Disease_Notebook.py         ← Exported script version
├── Parkinsons_Disease_Notebook.html       ← Rendered HTML version
├── Deployment_Files/
│   ├── Parkinsons_telemonitoring_app/     ← Docker app and code base
│   │   ├── Dockerfile
│   │   ├── requirements.txt
│   │   ├── app/
│   │   │   ├── train_model.py
│   │   │   ├── predict.py
│   │   │   ├── preprocessing.py
│   │   │   ├── load_data.py
│   │   │   ├── visual_outputs.py
│   │   │   └── interpretation_dashboard.py
│   │   ├── data/
│   │   │   ├── parkinsons_updrs.data
│   │   │   └── new_test.csv (optional user-supplied)
│   │   ├── output/						  # Output files created after prediction
│   │		├── predictions.csv           # Final predictions + probabilities
│   │		├── shap_summary.png          # SHAP global feature importance
│   │		├── shap_beeswarm.png         # SHAP instance-level visualization
│   │		├── shap_misclassified.png    # SHAP for misclassified test samples
│   │		├── lime_explanation_1.html   # LIME for sample 1
│   │		├── lime_explanation_2.html   # LIME for sample 2
│   │		├── lime_explanation_3.html   # LIME for sample 3
├── pull_image.bat
├── train.bat
├── prediction.bat




-----------------------------
Jupyter Notebook Development
-----------------------------
The file `Parkinsons_Disease_Notebook.ipynb` includes:
- Exploratory Data Analysis (EDA)
- Feature Engineering (e.g., shimmer slope, age × jitter)
- Model Training (Logistic Regression, XGBoost)
- Evaluation with accuracy, F1-score, confusion matrix
- Interpretation with SHAP and LIME
- Progression tracking and patient severity classification

It forms the core experimental and analytical foundation for the Docker app.




-----------------------------
Docker Deployment
-----------------------------
This project has been dockerized for platform-independent use.

Docker Hub Image:
`pavansakleshpurlingaraju/parkinsons-app:latest`

Run these batch scripts from the root:

1. **pull_image.bat**
   Pulls the Docker image.

2. **train.bat**
   Trains the model using `parkinsons_updrs.data`.

3. **prediction.bat**
   Runs prediction, accepts optional test file, and launches Streamlit dashboard.

Access: [http://localhost:8501](http://localhost:8501)



------------------------------
Important Notes for deployment
------------------------------
- For now prediction uses test.csv file present on the image itself.
- If user want to test with new test file that is present in thier pc then,
	`data/new_test.csv` must be present and full path must be specified while running prediction.
- Results will be saved to `output/predictions.csv`
- SHAP and LIME outputs will be displayed in the Streamlit dashboard.

-----------------------------
Credits
-----------------------------

Developed as part of MSc Dissertation – University of Salford (2025).
