Parkinson’s Disease Telemonitoring Project – README
====================================================

Overview
--------
This project involves both Jupyter-based exploratory development and a fully containerized deployment system for predicting Parkinson’s Disease severity using voice features. It integrates interpretable machine learning models (Logistic Regression, Random Forest, XGBoost) with SHAP and LIME for transparency. The system is built to support both academic experimentation and real-world usability.

-----------------------------
Project Structure
-----------------------------

## 📂 Project Structure
├── Parkinsons_Disease_Notebook.ipynb   # Full Jupyter analysis
├── Parkinsons_Disease_Notebook.py      # Exported script version
├── Parkinsons_Disease_Notebook.html    # Rendered HTML version
├── Deployment_Files/
│   ├── Parkinsons_telemonitoring_app/
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
│   │   │   └── new_test.csv (optional)
│   │   └── output/   # Generated after prediction
│   │       ├── predictions.csv
│   │       ├── shap_summary.png
│   │       ├── shap_beeswarm.png
│   │       ├── shap_misclassified.png
│   │       ├── lime_explanation_1.html
│   │       ├── lime_explanation_2.html
│   │       └── lime_explanation_3.html
│   ├── pull_image.bat
│   ├── train.bat
│   └── prediction.bat
└── README.md





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
