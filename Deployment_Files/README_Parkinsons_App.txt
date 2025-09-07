Parkinson’s Disease Telemonitoring App
======================================

This project is submitted as a containerized telemonitoring system for Parkinson’s Disease severity prediction and interpretability. The actual `Parkinsons_telemonitoring_app/` folder contains the code and configuration files used to build the Docker image.

End users **do not need to rebuild this image locally**. Instead, the system automatically pulls a prebuilt image from Docker Hub when run using the batch files provided.

Docker should be up and running

-----------------------------
Folder Structure
-----------------------------

Deployment_Files/
├── Parkinsons_telemonitoring_app/        <- Code and Docker image build context
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── app/
│   │   ├── train_model.py
│   │   ├── predict.py
│   │   ├── preprocessing.py
│   │   ├── load_data.py
│   │   ├── visual_outputs.py
│   │   └── interpretation_dashboard.py
│   ├── data/
│   │   ├── parkinsons_updrs.data
│   │   └── new_test.csv
│   └── output/						  # Output files created after prediction
│       ├── predictions.csv
│   	├── shap_summary.png          # SHAP global feature importance
│   	├── shap_beeswarm.png         # SHAP instance-level visualization
│   	├── shap_misclassified.png    # SHAP for misclassified test samples
│   	├── lime_explanation_1.html   # LIME for sample 1
│   	├── lime_explanation_2.html   # LIME for sample 2
│   	├── lime_explanation_3.html   # LIME for sample 3
├── pull_image.bat
├── train.bat
├── prediction.bat
├── README_Parkinsons_App.txt

-----------------------------
Docker Deployment Note
-----------------------------

This app does **not require local Docker image builds**. It uses a **public Docker Hub image**:

Image: `docker.io/pavansakleshpurlingaraju/parkinsons-app:latest`

The batch scripts provided automatically pull this image if not already present.

-----------------------------
Step-by-Step Instructions
-----------------------------

1. **Pull the Docker Image**  
   Run:
   ```
   pull_image.bat
   ```

2. **Train the Model **  
   This reads from `data/parkinsons_updrs.data` and saves model to `app/model/`
   ```
   train.bat
   ```

3. **Run Prediction and Launch Dashboard**  
   Run:
   ```
   prediction.bat
   ```

   Access the dashboard at:
   http://localhost:8501

-----------------------------
Notes
-----------------------------

- For now prediction uses test.csv file present on the image itself.
- If user want to test with new test file that is present in thier pc then,
	`data/new_test.csv` must be present and full path must be specified while running prediction.
- Results will be saved to `output/predictions.csv`
- SHAP and LIME outputs will be displayed in the Streamlit dashboard.

-----------------------------
Credits
-----------------------------

Developed as part of MSc Dissertation – University of Salford (2025)
