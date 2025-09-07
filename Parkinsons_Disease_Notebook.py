#!/usr/bin/env python
# coding: utf-8

# # Predictive Modelling of Parkinson’s Disease Severity Using Telemonitoring Voice Data: A Machine Learning Approach

# ### Installation Needed 

# In[1]:


#!pip install xgboost
#!pip install shap
#!pip install lime


# ### Imports and Initial Setup

# In[2]:


import os
import warnings

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.offline as pyo
import plotly.io as pio
pio.renderers.default = "notebook_connected"

from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

import statsmodels.formula.api as smf
from scipy.interpolate import interp1d
from scipy.stats import ttest_ind, pearsonr
from sklearn.preprocessing import FunctionTransformer

from sklearn.model_selection import train_test_split, GroupShuffleSplit, cross_val_score, GridSearchCV, StratifiedGroupKFold
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import (r2_score, mean_squared_error, accuracy_score, precision_score,
                             f1_score, classification_report, ConfusionMatrixDisplay, confusion_matrix)

import shap
from shap import summary_plot
import lime
import lime.lime_tabular


#%matplotlib inline
warnings.filterwarnings("ignore")


# ## Data Inspection and Splitting

# ### Data Loading and Validation

# This section ensures the dataset is correctly loaded and clean before moving to analysis.  
# We perform the following checks:
# 
# - Confirm the file exists before loading
# - Load the dataset into memory
# - Display the first and last few rows to inspect structure
# - Show missing value counts for each column
# - Review summary statistics using 'describe()'
# - Identify and display any duplicate rows
# - Validate that:
#   - 'age' values are between 30 and 100
#   - 'sex' values are either 0 or 1
# - Check for negative values in numeric columns
# 
# Negative values are flagged because features like 'Jitter', 'Shimmer', and clinical scores should not be negative. These values may result from sensor noise, data entry issues, or corruption.  
# If the number of negative rows is very small (less than 1% of total), we remove them to improve data integrity without affecting dataset balance.

# In[3]:


def load_and_validate_data(file_path):
    # 1. Check if file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} not found.")

    # 2. Load data
    df = pd.read_csv(file_path)
    print("=== File loaded successfully ===")
    
    # 3. Display head
    print("\n=== First 5 rows ===")
    display(df.head())  

    # 4. Check for missing values
    print("\n=== Missing values ===")
    display(df.isnull().sum())

    # 5. Descriptive statistics
    print("\n=== Descriptive statistics ===")
    display(df.describe())

    # 6. Check for duplicate rows
    print("\n=== Checking for duplicate rows ===")
    duplicate_rows = df[df.duplicated()]
    print(f"Total duplicate rows: {len(duplicate_rows)}")
    if not duplicate_rows.empty:
        display(duplicate_rows.head())

    # 8. Value range checks
    assert df['age'].between(30, 100).all(), "Unexpected values in 'age' column"
    assert df['sex'].isin([0, 1]).all(), "Invalid values in 'sex' column"
    
    # 7. Checking for multiple signal readings from one telemonitoring session
    print("\n=== Multiple measures check ===\n")
    dup_check = df.groupby(['subject#', 'test_time']).size()
    true_duplicates = dup_check[dup_check > 1]
    print(f"Multiple measures at the same time: {len(true_duplicates)}")

    # 8. Check and remove negative values (if negligible)
    numeric_cols = df.select_dtypes(include=np.number).columns
    negative_mask = (df[numeric_cols] < 0)
    negative_counts = negative_mask.sum()
    total_negatives = negative_mask.values.sum()

    print("\n=== Negative value counts ===")
    display(negative_counts[negative_counts > 0])

    if total_negatives / df.shape[0] < 0.01:
        df = df[~negative_mask.any(axis=1)]
        print(f"\nDropped {total_negatives} rows with negative values (negligible share).")

    print("\n=== Validation complete ===\n\n")
    return df


# In[4]:


# === Load and validate data ===
DATA_FILE = "parkinsons_updrs.data"  # Change to your actual filename
df = load_and_validate_data(DATA_FILE)


# ### Group-Aware Train-Test Split
# We split the dataset into training and test sets using 'GroupShuffleSplit' to ensure no subject appears in both sets.

# In[5]:


X = df.drop(columns=['motor_UPDRS', 'total_UPDRS'])
y = df["motor_UPDRS"]  # or "total_UPDRS"
groups = df["subject#"]

gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups=groups))

X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]


# ## Exploratory data Analysis

# #### Plot distributions of numeric features
# 
# Histograms are used to observe the distribution and spread of numerical features in the dataset.

# In[6]:


print("\n Plotting feature distributions...")
X_train.drop(columns=['subject#']).hist(figsize=(15, 12), bins=30, color='skyblue', edgecolor='black')
plt.suptitle("Feature Distributions", fontsize=16)
plt.tight_layout()
plt.show()


# #### Correlation Heatmap 
# 
# Correlation heatmap is used to identify relationships between features and detect potential multicollinearity.

# In[7]:


# Combine features and target for correlation
train_data = X_train.copy()
train_data["motor_UPDRS"] = df["motor_UPDRS"]

# Compute correlation matrix
corr_matrix = train_data.drop(columns=['subject#']).corr()

# Plot correlation heatmap
plt.figure(figsize=(14, 10))
ax = sns.heatmap(
    corr_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0,
    square=True,
    cbar_kws={
        'label': 'Correlation Strength',
        'shrink': 0.9,
        'ticks': [-1, -0.5, 0, 0.5, 1]
    }
)

# Label colorbar meaning
colorbar = ax.collections[0].colorbar
colorbar.ax.set_yticklabels([
    '-1: Strong Negative',
    '-0.5: Moderate Negative',
    '0: No Correlation',
    '+0.5: Moderate Positive',
    '+1: Strong Positive'
])

plt.title("Correlation Heatmap", fontsize=16)
plt.tight_layout()
plt.show()


# #### Violin plot
# Violin plot is used to visualize the distribution and density of a feature across severity classes.  
# It helps reveal differences in value spread, central tendency, and potential class-separating patterns.

# In[8]:


# Create Age Groups
train_data["age_group"] = pd.cut(train_data["age"], bins=[0, 50, 60, 70, 80, 100], labels=["<50", "50-60", "60-70", "70-80", "80+"])

# Motor UPDRS by Sex
plt.figure(figsize=(8, 5))
sns.violinplot(data=train_data, x="sex", y="motor_UPDRS")
plt.title("Motor UPDRS by Sex")
plt.xticks([0, 1], ['Male', 'Female'])
plt.tight_layout()
plt.show()

# Motor UPDRS by Age Group
plt.figure(figsize=(10, 5))
sns.violinplot(data=train_data, x="age_group", y="motor_UPDRS", palette="Set2")
plt.title("Motor UPDRS by Age Group")
plt.tight_layout()
plt.show()


# #### 4. Trend Heatmap 
# Avg UPDRS over Time per Age Group (with labeled colorbar)
# 
# While overall distributions give a general view, class-wise plots (e.g., violin/histogram by severity) help identify features that distinguish between classes for classification.

# In[9]:


# Bin test_time into 10 equal-width bins
train_data["time_bin"] = pd.cut(train_data["test_time"], bins=10)

# Group and compute mean motor_UPDRS
heatmap_data = train_data.groupby(["age_group", "time_bin"])["motor_UPDRS"].mean().reset_index()

# Pivot for heatmap
heatmap_pivot = heatmap_data.pivot(index="age_group", columns="time_bin", values="motor_UPDRS")

# Plot heatmap
plt.figure(figsize=(10, 6))
ax = sns.heatmap(heatmap_pivot, cmap="YlOrRd", annot=True, fmt=".1f", cbar_kws={
    'label': 'Average Motor UPDRS',
    'shrink': 0.9
})

# Customize colorbar tick labels
colorbar = ax.collections[0].colorbar
colorbar.ax.set_yticklabels([f"{v:.0f}" for v in colorbar.get_ticks()])  # Rounded values

plt.title("Mean Motor UPDRS Over Time by Age Group ")
plt.xlabel("Test Time Bin")
plt.ylabel("Age Group")
plt.tight_layout()
plt.show()


# ### Rapid Progression Analysis – Overview
# 
# - This section identifies patients who show a steep increase in motor symptom severity ('motor_UPDRS') over time.  Detecting rapid progression helps profile high-risk individuals, and could improve early intervention or personalized treatment models.  
# - We track how 'motor_UPDRS' changes per subject and flag cases with steep trends.
# - We first sort the dataset by 'subject#' and 'test_time' to ensure the progression is tracked chronologically for each patient. This is essential before applying rolling or difference-based analysis.
# 
# 
# 

# #### Step 1: Filter out recently improving subjects
# - We define a 30-day progression threshold based on a yearly rate of 4.6 motor_UPDRS points.  
# - Subjects who have recently improved (i.e., decreasing trend in the last 30 days) are filtered out using is_recently_decreasing().  
# - Only those still progressing are retained for further rapid progression analysis.
# 

# In[10]:


# 5. RAPID PROGRESSION DETECTION – 30-days WINDOW-BASED

train_subjects = train_data["subject#"].unique()

# Define progression threshold
fast_motor_progression_per_year = 4.6 
window_days = 30
motor_threshold = fast_motor_progression_per_year / 365 * window_days
top_n = 10

def is_recently_decreasing(group, col, tail_days=30):
    recent = group[group["test_time"] >= (group["test_time"].max() - tail_days)]
    return recent[col].iloc[-1] < recent[col].iloc[0]

filtered_subjects = []
for subject in train_subjects:
    group = train_data[train_data["subject#"] == subject].sort_values("test_time")
    if not is_recently_decreasing(group, "motor_UPDRS"):
        filtered_subjects.append(subject)


# #### Step 2: Detect 30-day rapid progression spikes (motor only)
# - We define has_rapid_progression to check if a subject shows a sharp increase in motor_UPDRS within any 30-day window (±2 days).  
# - For each subject, we compare all test point pairs and flag those with a change exceeding the threshold.  
# - Subjects meeting this condition are added to rapid_motor_progressors.

# In[11]:


def has_rapid_progression(group, col, threshold):
    times = group["test_time"].values
    values = group[col].values
    for i in range(len(times)):
        for j in range(i + 1, len(times)):
            delta_days = times[j] - times[i]
            if window_days - 2 < delta_days <= window_days + 2:
                delta_val = values[j] - values[i]
                if delta_val > threshold:
                    return True
    return False

rapid_motor_progressors = []
for subject in filtered_subjects:
    group = train_data[train_data["subject#"] == subject].sort_values("test_time")
    if has_rapid_progression(group, "motor_UPDRS", motor_threshold):
        rapid_motor_progressors.append(subject)


# #### Step 3: Compute Slopes and Rank Top Subjects
# 
# - We define compute_slope using linear regression to calculate the rate of change in motor_UPDRS over time.  
# - We apply it to each rapid progressor to get their progression slope and rank them.  
# - The top N subjects with the steepest slopes are selected as the fastest progressors.

# In[12]:


def compute_slope(x, y):
    x = np.array(x).reshape(-1, 1)
    y = np.array(y)
    if len(x) < 2:
        return 0
    return LinearRegression().fit(x, y).coef_[0]

motor_slopes = {
    subject: compute_slope(train_data[train_data["subject#"] == subject]["test_time"],
                           train_data[train_data["subject#"] == subject]["motor_UPDRS"])
    for subject in rapid_motor_progressors
}

top_motor_subjects = sorted(motor_slopes, key=motor_slopes.get, reverse=True)[:top_n]

time_range = train_data.groupby("subject#")["test_time"].apply(lambda x: x.max() - x.min())
print(f"\nTime range (min–max): {time_range.min():.2f} – {time_range.max():.2f} days")
print(f"30-day Motor UPDRS Threshold: {motor_threshold:.2f}")
print(f"Top {top_n} Rapid Motor Progressors: {top_motor_subjects}\n")


# #### Step 4: Plot the Progression
# 
# We use Plotly to visualize the `motor_UPDRS` progression over time for the top N fastest progressors.  
# Each line represents a subject, showing how quickly their scores are rising across test days.  
# This interactive plot helps validate the rapid progression detection and provides clear insight into patient deterioration trends.
# 

# In[13]:


fig = go.Figure()

for subject in top_motor_subjects:
    patient_data = train_data[train_data["subject#"] == subject].sort_values("test_time")
    fig.add_trace(go.Scatter(
        x=patient_data["test_time"],
        y=patient_data["motor_UPDRS"],
        mode='lines+markers',
        name=f"Subject {subject}",
        hoverinfo="text",
        text=[f"Subject: {subject}<br>Time: {t:.1f}<br>Motor UPDRS: {u:.2f}" for t, u in zip(patient_data["test_time"], patient_data["motor_UPDRS"])]
    ))

fig.update_layout(
    title=f"Top {top_n} Motor UPDRS Progressors (30-Day Spike, Training Set)",
    xaxis_title="Test Time (Days)",
    yaxis_title="Motor UPDRS",
    hovermode="closest",
    template="plotly_white",
    width=1000, height=600
)
fig.show()
# Save interactive plot as HTML file (fully self-contained)
#pyo.plot(fig, filename='top_10_updrs_progressors.html', auto_open=True)


# ### Rolling average per subject
# 
# - Calculateed a 3-point rolling average of motor_UPDRS for each subject using rolling(window=3).  
# - This smooths out short-term fluctuations in symptom scores and helps highlight the underlying progression trend more clearly.
# 
# - Randomly selected 5 subjects from the training set and plot their rolling average motor_UPDRS over time.  
# - This provides a visual understanding of how symptoms evolve and how rolling averages reduce noise in progression patterns.

# In[14]:


# Step 1: Compute rolling average per subject 
rolling_df = []

for subject, group in train_data.groupby("subject#"):
    group_sorted = group.sort_values("test_time").copy()
    group_sorted["motor_UPDRS_rollmean"] = group_sorted["motor_UPDRS"].rolling(window=3, min_periods=1).mean()
    rolling_df.append(group_sorted)

df_rolling = pd.concat(rolling_df)

# Step 2: Plot rolling average for 5 random subjects in training set
sample_subjects = train_data["subject#"].drop_duplicates().sample(5, random_state=42)

plt.figure(figsize=(12, 6))
for subj in sample_subjects:
    temp = df_rolling[df_rolling["subject#"] == subj]
    plt.plot(temp["test_time"], temp["motor_UPDRS_rollmean"], label=f"Subject {subj}")

plt.title("Rolling Average of Motor UPDRS (Window=3)")
plt.xlabel("Test Time (Days)")
plt.ylabel("Motor UPDRS (Rolling Mean)")
plt.legend()
plt.tight_layout()
plt.show()


# ### 30-Day Delta Changes in Motor UPDRS
# - Here iterated through each subject and compute the change in motor_UPDRS across all available 30-day windows (±2 days).  
# - Only the first matching pair for each starting point is used to avoid multiple overlapping windows.  
# - This quantifies how much each patient’s symptoms changed in typical 1-month periods.
# - plot a histogram of the 30-day delta values to observe the distribution of symptom changes.  
# - This shows how frequently patients improve, worsen, or remain stable over 1-month intervals, giving insights into short-term progression patterns.

# In[15]:


delta_motor = []

# Loop through each subject in training data
for subject, group in train_data.groupby("subject#"):
    group = group.sort_values("test_time").reset_index(drop=True)
    for i in range(len(group)):
        for j in range(i + 1, len(group)):
            delta_t = group.loc[j, "test_time"] - group.loc[i, "test_time"]
            if 28 <= delta_t <= 32:
                delta_motor.append(group.loc[j, "motor_UPDRS"] - group.loc[i, "motor_UPDRS"])
                break  # Only take the first 30-day window delta for each i

# Plot histogram of delta changes
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.hist(delta_motor, bins=30, color='teal', edgecolor='black')
plt.title("Δ Motor UPDRS Over 30-Day Windows (Training Set)")
plt.xlabel("Change in Motor UPDRS")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()


# ### Severity Distribution Analysis
# - Clinical severity levels were defined based on motor_UPDRS score ranges: 0–20 for Mild and 21–40 for Moderate+.
# - Subjects were filtered into severity categories, and counts were computed using conditional indexing within a loop.
# 
# 
# ##### Severity Label Encoding for Classification and distribution
# The continuous motor_UPDRS scores were converted into binary severity labels:
# - "Mild" for scores ≤ 20
# - "Moderate+" for scores > 20
# 
# These labels were then encoded into integers using LabelEncoder to prepare for classification modeling.
# Class distribution in y_train_encoded was checked to verify label balance after encoding.

# In[16]:


# 1. Drop duplicates to count per subject
train_unique = df.iloc[train_idx].drop_duplicates("subject#")

# 2. Define severity bins and count per group
severity_df = pd.DataFrame([
    {
        "Severity Level": label,
        "Motor UPDRS Count": train_unique.query(f"motor_UPDRS >= {low} and motor_UPDRS <= {high}").shape[0]
    }
    for label, (low, high) in {"Mild": (0, 20), "Moderate+": (21, 40)}.items()
])

# 3. Plot bar chart
plt.figure(figsize=(4, 5))
ax = sns.barplot(data=severity_df, x="Severity Level", y="Motor UPDRS Count", palette="pastel")
for p in ax.patches:
    ax.text(p.get_x() + p.get_width() / 2, p.get_height(), f"{int(p.get_height())}", 
            ha="center", va="bottom", fontweight="bold")
plt.title("Motor UPDRS Severity Level Distribution (Train Set)")
plt.ylabel("Number of Subjects")
plt.xlabel("Severity Level")
plt.tight_layout()
plt.show()

# 4. Convert to binary labels + encode
le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train.apply(lambda x: "Mild" if x <= 20 else "Moderate+"))
y_test_encoded = le.transform(y_test.apply(lambda x: "Mild" if x <= 20 else "Moderate+"))

# 5. Show class counts
print("Class Distribution in y_train (Binary):")
print(pd.Series(y_train_encoded).value_counts())


# ## Feature Engineering and Scaling

# ### Domain based feature engineering and Scaling
# - Applies domain-driven transformations to input data.
# - log, sqrt, interaction, and shimmer slope features used in modeling.

# In[17]:


base_features = [
    "age", "sex", "Jitter(%)", "Jitter(Abs)", "Jitter:RAP", "Jitter:PPQ5", "Jitter:DDP",
    "Shimmer", "Shimmer(dB)", "Shimmer:APQ3", "Shimmer:APQ5", "Shimmer:APQ11", "Shimmer:DDA",
    "NHR", "HNR", "RPDE", "DFA", "PPE"]

def engineer_features(df):
    df = df.copy()

    # Derived features
    df["log_PPE"] = np.log1p(df["PPE"])
    df["Inv_JitterAbs"] = 1 / (df["Jitter(Abs)"] + 1e-6)
    df["Shimmer_Slope"] = df["Shimmer:APQ11"] - (df["Shimmer:APQ3"] + 1e-6)
    df["Age_Sex_Interaction"] = df["age"] * df["sex"]


    return df


# In[18]:


# Transformer to use in pipeline
engineer_transformer = FunctionTransformer(engineer_features, validate=False)

# Based on correlation and distribution EDA and Avoiding redundancy 
final_features = [
    "age", "sex", "Shimmer_Slope", "Inv_JitterAbs",
    "log_PPE", "RPDE", "DFA", "Age_Sex_Interaction"
]

select_features_transformer = FunctionTransformer(lambda df: df[final_features], validate=False)


# #### Selected Feature Justification
# The following features were selected based on clinical relevance, exploratory analysis, and engineered signal strength:
# - **age**: A key demographic factor often correlated with progression severity.
# - **sex**: Included to account for differences in vocal traits across genders.
# - **Shimmer_Slope**: Reflects difference between short- and long-term shimmer measures (APQ11 - APQ3), indicating vocal instability.
# - **Inv_JitterAbs**: The inverse of jitter (1 / Jitter:Abs) emphasizes vocal stability, with higher values indicating more regular vocal fold vibrations, thus serving as a clinically interpretable marker of healthier voice function.
# - **RPDE**: Lower RPDE values indicate less periodicity and higher noise — a hallmark of dysphonia in Parkinson’s speech, especially as severity increases.
# - **log_PPE**: Log-transformed Pitch Period Entropy; stabilizes skew and highlights pitch irregularity.
# - **DFA**: High DFA values reflect more stable and self-similar patterns in vocal signals — typically preserved in early stages but lost as motor control worsens.
# 
# These features balance interpretability and predictive value, making them suitable for modeling motor symptom severity.

# ### Feature Scaling 
# - We apply RobustScaler to voice features to normalize them without being skewed by extreme outliers.
# - Unlike standard scaling, it uses the median and interquartile range (IQR), making it ideal for jitter, shimmer, and PPE which can have sharp spikes in Parkinson’s speech.

# In[19]:


def get_scaler():
    return RobustScaler()


# ## Modelling - Severity Level Prediction (Mild and Moderate+)
# ### Cross-Validation Evaluation of Three Classifiers Using Pipelines
# 
# - Three classification models (Logistic Regression, Random Forest, and XGBoost) were wrapped in Pipeline objects, each with 'RobustScaler' applied in pipeline step.  
# This ensures consistent feature scaling before model fitting.
# 
# - 'StratifiedGroupKFold' was used to perform 5-fold cross-validation, preserving both class balance and group independence across folds.  
# For each model, predictions were collected across all validation folds and combined to evaluate performance.
# 
# - The final classification report for each model summarizes overall precision, recall, and F1-score across all folds using the encoded severity labels.

# In[20]:


# Define group labels for cross-validation
groups_train = df.iloc[train_idx]["subject#"]

from sklearn.decomposition import PCA

# Define model pipelines with PCA step added
clf_pipelines = {
    "Logistic": Pipeline([
        ("engineer", engineer_transformer),
        ("select", select_features_transformer),
        ("scale", get_scaler()),
        ("clf", LogisticRegression(max_iter=1000, class_weight='balanced'))
    ]),
    "RandomForest": Pipeline([
        ("engineer", engineer_transformer),
        ("select", select_features_transformer),
        ("scale", get_scaler()),
        ("clf", RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42))
    ]),
    "XGBoost": Pipeline([
        ("engineer", engineer_transformer),
        ("select", select_features_transformer),
        ("scale", get_scaler()),
        ("clf", XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric='logloss', random_state=42))
    ])
}


# Stratified group-aware cross-validation
cv = StratifiedGroupKFold(n_splits=5)
cv_clf_results_all = {}

# Evaluate each classifier
for model_name, pipeline in clf_pipelines.items():
    clf_preds = []

    for tr_idx, val_idx in cv.split(X_train, y_train_encoded, groups=groups_train):
        X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train_encoded[tr_idx], y_train_encoded[val_idx]

        pipeline.fit(X_tr, y_tr)
        y_pred = pipeline.predict(X_val)

        fold_df = pd.DataFrame({
            "Actual_Severity": y_val,
            "Predicted_Severity": y_pred
        }, index=X_val.index)

        clf_preds.append(fold_df)

    # Store aggregated results for this model
    cv_clf_results_all[model_name] = pd.concat(clf_preds, ignore_index=True) 
    
# Print out performance metrics
for name, result_df in cv_clf_results_all.items():
    print(f"\n{name} Classification Performance:")
    print(classification_report(result_df["Actual_Severity"], result_df["Predicted_Severity"], target_names=le.classes_))


# ### Model Comparison Summary (Cross-Validation)
# 
# - Macro-averaged accuracy, precision, recall, Sensitivity(TPR), Specificity(TNR) and F1-scores were computed for each model using cross-validation results.  
# - This summary table supports model selection based on consistent performance across folds.

# In[21]:


from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix

summary_metrics = []

# Loop over the collected CV results
for model_name, result_df in cv_clf_results_all.items():
    y_true = result_df["Actual_Severity"]
    y_pred = result_df["Predicted_Severity"]

    # Confusion matrix components
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    summary_metrics.append({
        "Model": model_name,
        "Accuracy": round(accuracy_score(y_true, y_pred), 3),
        "Precision (Macro)": round(precision_score(y_true, y_pred, average="macro"), 3),
        "Recall (Macro)": round(recall_score(y_true, y_pred, average="macro"), 3),
        "F1-Score (Macro)": round(f1_score(y_true, y_pred, average="macro"), 3),
        "Sensitivity (TPR)": round(tp / (tp + fn), 3),
        "Specificity (TNR)": round(tn / (tn + fp), 3)
    })

# Create and display summary DataFrame
comparison_df = pd.DataFrame(summary_metrics).sort_values(by="F1-Score (Macro)", ascending=False)

print("\nModel Comparison Summary (Cross-Validation):")
display(comparison_df)


# ### Hyperparameter Tuning Using GridSearchCV with Pipelines
# 
# To optimize model performance, hyperparameter tuning was performed for Logistic Regression, Random Forest, and XGBoost using 'GridSearchCV'.  
# Each model was wrapped in a 'Pipeline' with a 'RobustScaler' to ensure consistent preprocessing.
# 
# The 'tune_model' function takes a model pipeline, parameter grid, training data, and group-aware cross-validation strategy ('StratifiedGroupKFold').  
# It evaluates each parameter combination using 5-fold cross-validation, scoring by macro-averaged F1-score to handle class imbalance.
# 
# Custom parameter grids were defined for each model, and the best estimator (i.e., best set of parameters) was selected and stored for future evaluation.
# 

# In[22]:


# === 1. Function to run GridSearchCV on a pipeline ===
def tune_model(pipeline, param_grid, X, y, groups, cv):
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=cv,
        scoring='f1_macro',
        n_jobs=-1,
        verbose=1
    )
    grid_search.fit(X, y, groups=groups)
    return grid_search

# === 2. Define parameter grids for each model ===
param_grids = {
    "Logistic": {
        "clf__C": [0.01, 0.1, 1, 10]
    },
    "RandomForest": {
        "clf__n_estimators": [100, 200],
        "clf__max_depth": [None, 5, 10],
        "clf__min_samples_split": [2, 5]
    },
    "XGBoost": {
        "clf__n_estimators": [100, 200],
        "clf__max_depth": [3, 5],
        "clf__learning_rate": [0.01, 0.1]
    }
}

# === 3. Perform tuning for each model ===
best_estimators = {}
cv_scores = {}

for model_name, pipeline in clf_pipelines.items():
    print(f"\nTuning {model_name}...")
    grid = tune_model(pipeline, param_grids[model_name], X_train, y_train_encoded, groups_train, cv)
    best_estimators[model_name] = grid.best_estimator_
    cv_scores[model_name] = round(grid.best_score_, 4)
    print(f"Best Params for {model_name}: {grid.best_params_}")
    print(f"Best F1 Score: {cv_scores[model_name]}")

# === 4. Create combined model dictionary for training evaluation ===
tuned_models = {
    "Logistic Regression": (best_estimators["Logistic"], cv_scores["Logistic"]),
    "Random Forest": (best_estimators["RandomForest"], cv_scores["RandomForest"]),
    "XGBoost": (best_estimators["XGBoost"], cv_scores["XGBoost"])
}


# ### Training Set Evaluation of Tuned Models
# 
# Each best estimator (from hyperparameter tuning) was retrained on the full training set and evaluated to assess model fit.
# 
# The evaluate_on_train function computes:
# - **Classification report** including precision, recall, and F1-scores
# - **Macro F1-score and accuracy** on the training set
# - Comparison with the **cross-validation (CV) F1-score** to assess overfitting risk
# - **Confusion matrix** to visualize prediction distribution
# 
# A small gap between training and CV performance indicates good generalization, while a large gap may suggest overfitting.

# In[23]:


def evaluate_on_train(model_name, model, X_train, y_train, label_encoder, cv_f1_score=None):
    print(f"\n\n=== {model_name} Performance on Training Set ===")
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_train)
    
    print(classification_report(y_train, y_pred, target_names=label_encoder.classes_))
    print("Accuracy:", round(accuracy_score(y_train, y_pred), 3))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_train, y_pred))
    
    if cv_f1_score:
        print("CV F1-Score (Macro):", round(cv_f1_score, 3))

# Loop through models with F1-score
for model_name, (model, cv_f1) in tuned_models.items():
    evaluate_on_train(model_name, model, X_train, y_train_encoded, le, cv_f1_score=cv_f1)


# #### Tuned Model Performance Summary
# 
# This table summarizes the performance of all three tuned models on the training set using key classification metrics.  
# It includes macro-averaged scores, sensitivity (recall for Moderate+), specificity (recall for Mild), and cross-validated F1-scores for comparison.
# 

# In[24]:


train_summary_metrics = []

for model_name, (model, cv_f1_score_val) in tuned_models.items():
    model.fit(X_train, y_train_encoded)
    y_pred = model.predict(X_train)

    acc = accuracy_score(y_train_encoded, y_pred)
    precision = precision_score(y_train_encoded, y_pred, average="macro")
    recall = recall_score(y_train_encoded, y_pred, average="macro")
    f1 = f1_score(y_train_encoded, y_pred, average="macro")

    tn, fp, fn, tp = confusion_matrix(y_train_encoded, y_pred).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)

    train_summary_metrics.append({
        "Model": model_name,
        "Accuracy": round(acc, 3),
        "Precision (Macro)": round(precision, 3),
        "Recall (Macro)": round(recall, 3),
        "F1-Score (Macro)": round(f1, 3),
        "Sensitivity (TPR)": round(sensitivity, 3),
        "Specificity (TNR)": round(specificity, 3),
        "CV F1-Score": round(cv_f1_score_val, 3)
    })

# === 3. Display comparison table ===
comparison_df = pd.DataFrame(train_summary_metrics).sort_values(by="F1-Score (Macro)", ascending=False)

print("\n=== Tuned Models: Training Performance Summary ===")
display(comparison_df)


# ### Final Evaluation on Held-Out Test Set (Logistic Regression)
# 
# The best-performing model from tuning (Logistic Regression) was used to make predictions on the held-out test set.
# 
# The following metrics were computed:
# - **Classification Report**: Shows precision, recall, and F1-score per class
# - **Accuracy**: Overall classification accuracy
# - **Confusion Matrix**: Displays the distribution of predicted vs actual labels
# 
# This evaluation provides an unbiased estimate of real-world model performance, as the test set was never used during training or tuning.

# In[25]:


def evaluate_model_on_test(model, X_test, y_test, label_encoder, model_name="Model"):
    # Predict on test set
    y_pred = model.predict(X_test)

    # Compute evaluation metrics
    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="macro")
    recall = recall_score(y_test, y_pred, average="macro")
    f1 = f1_score(y_test, y_pred, average="macro")

    # Handle confusion matrix safely
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    # Display detailed performance report
    print(f"=== {model_name} – Held-Out Test Set Performance ===")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
    print(f"Accuracy: {round(acc, 3)}")
    print(f"Precision (Macro): {round(precision, 3)}")
    print(f"Recall (Macro): {round(recall, 3)}")
    print(f"F1-Score (Macro): {round(f1, 3)}")
    print(f"Sensitivity (TPR): {round(sensitivity, 3)}")
    print(f"Specificity (TNR): {round(specificity, 3)}")
    print("\nConfusion Matrix:")
    print(cm)

    return y_pred


# Evaluating on the Logistic Regression model on the test set
evaluate_model_on_test(
    model=best_estimators["Logistic"],
    X_test=X_test,  # raw test set (will go through pipeline inside)
    y_test=y_test_encoded,
    label_encoder=le,
    model_name="Logistic Regression"
)


# #### Interpretation 
# 
# - The model achieved an accuracy of 60% and a macro F1-score of 0.59 on the held-out test set.  
# - This is close to the CV F1-score (0.56), which is a good sign — it suggests the model generalized well and is not overfitting.  - Performance is balanced, though slightly better on the 'Mild' class, indicating room for improvement in distinguishing 'Moderate+' cases.
# 

# ## Logistic Regression Feature Importance
# 
# To interpret how each feature contributes to the prediction, the model coefficients from the best Logistic Regression estimator were extracted.  
# These coefficients indicate the direction and strength of influence each feature has on the predicted severity class.
# 

# In[26]:


# === Access logistic model from pipeline ===
# Ensure you're using the correct best estimator (not redefined)
logistic_model = best_estimators["Logistic"].named_steps["clf"]

# === Extract coefficients for each feature ===
logit_coefs = logistic_model.coef_[0]

# === Plotting feature importance based on coefficients ===
plt.figure(figsize=(7, 4))
plt.barh(final_features, logit_coefs)
plt.xlabel("Coefficient Value")
plt.ylabel("Feature")
plt.title("Logistic Regression Feature Importance (Raw Coefficients)")
plt.grid(True)
plt.tight_layout()
plt.show()


# #### Interpretation
# 
# Positive coefficients indicate features associated with higher severity (Moderate+), while negative values are linked to the Mild class.  
# This helps identify which vocal and clinical factors are most influential in the model's decision-making.
# 

# ###  X_train_final and X_test_final from pipeline

# In[27]:


# Get preprocessing pipeline from best estimator
preprocessing = best_estimators["Logistic"].named_steps

# Transform training and test data manually 
X_train_final = preprocessing["scale"].transform(
    preprocessing["select"].transform(
        preprocessing["engineer"].transform(X_train.copy())
    )
)

X_test_final = preprocessing["scale"].transform(
    preprocessing["select"].transform(
        preprocessing["engineer"].transform(X_test.copy())
    )
)


# ### SHAP Interpretation

# ####  SHAP Analysis: Interpreting Feature Influence on Parkinson's Severity Prediction
# 
# SHAP (SHapley Additive exPlanations) is used to interpret how each feature contributes to the model's prediction. This method is based on cooperative game theory and provides both **global** and **individual-level** explanations.
# 
# In this section, we:
# 1. Initialize a SHAP explainer for the trained logistic regression model.
# 2. Compute SHAP values for the test set to understand feature contributions.
# 3. Generate visualizations:
#    - **Bar Plot**: Global feature importance (mean absolute SHAP values)
#    - **Beeswarm Plot**: Distribution of individual SHAP values per feature across all test samples
# 
# These plots help identify which features consistently drive predictions toward the "Mild" or "Moderate+" class.
# 

# In[28]:


# 1. Create SHAP explainer for linear model
explainer = shap.Explainer(logistic_model, X_train_final, feature_names=final_features)

# 2. Compute SHAP values for the held-out test set
shap_values = explainer(X_test_final)

# 3. Global Feature Importance (Bar plot)
shap.summary_plot(shap_values, X_test_final, plot_type="bar", show=True)

# 4. Individual-Level Impact (Beeswarm plot)
shap.summary_plot(shap_values, X_test_final, show=True)


# #### SHAP Interpretation Summary
# - **Age**
# Highest global impact on severity classification.
# Older individuals more likely predicted as Moderate+.
# Consistent with disease progression trends.
# 
# - **DFA (Detrended Fluctuation Analysis)**
# High values reduce severity prediction.
# Reflects more stable vocal dynamics, typical of milder symptoms.
# 
# - **Shimmer_Slope**
# Strong indicator of amplitude instability in voice.
# High values push prediction toward Moderate+.
# 
# - **Age_Sex_Interaction**
# Moderate influence.
# Suggests age and sex jointly affect vocal degradation patterns.
# 
# - **log_PPE**
# Higher pitch entropy = more vocal irregularity.
# Indicates impaired neuromuscular control seen in PD.
# 
# - **RPDE**
# Subtle but meaningful.
# Captures voice periodicity, helps differentiate between healthy and impaired speech.
# 
# - **Sex**
# Plays a role, but secondary to core acoustic markers.
# 
# - **Inv_JitterAbs**
# New inverse jitter-based feature.
# Lower jitter (higher Inv_JitterAbs) contributes to Mild prediction.
# Acts as a stability marker in speech.

# ###  SHAP Summary – Misclassified Test Samples
# 
# To better understand the model's limitations, we analyzed SHAP values specifically for **misclassified test cases** (i.e., where the predicted severity class didn't match the actual one). This helps identify which features were most misleading or contributed to confusion.

# In[29]:


# 1. Predict on already-transformed X_test_final (no need to transform again)
logistic_model = best_estimators["Logistic"].named_steps["clf"]
y_test_pred = logistic_model.predict(X_test_final)

# 2. Identify misclassified samples
misclassified_mask = (y_test_pred != y_test_encoded)

# 3. Create SHAP explainer on training set
explainer = shap.Explainer(logistic_model, X_train_final, feature_names=final_features)

# 4. Compute SHAP values on the test set
shap_values = explainer(X_test_final)

# 5. Filter SHAP values for misclassified samples
shap_misclassified = shap.Explanation(
    values=shap_values.values[misclassified_mask],
    base_values=shap_values.base_values[misclassified_mask],
    data=X_test_final[misclassified_mask],
    feature_names=final_features
)

# 6. Plot beeswarm for misclassified
shap.summary_plot(shap_misclassified, X_test_final[misclassified_mask], plot_type="dot", show=True)


# #### Observations from the Beeswarm Plot (Misclassified Samples)
# 
# - **DFA**
# Still dominant among errors.
# High values sometimes lead to incorrect Mild classification.
# 
# - **Shimmer_Slope**
# High variability persists.
# Contributes both positively and negatively — inconsistent across individuals.
# 
# - **Age_Sex_Interaction**
# Moderate role.
# Interactions seem to add ambiguity in borderline predictions.
# 
# - **log_PPE and RPDE**
# Overlapping contributions across classes.
# May lack discriminative power alone in difficult cases.
# 
# - **Sex**
# Lower influence overall, but shows direction flips.
# Potentially interacts with acoustic features in subtle ways.
# 
# - **Inv_JitterAbs**
# Low inverse jitter still drives some Moderate+ predictions.
# Impact less clear in misclassifications.

# ### LIME Interpretability – Logistic Regression
# 
# To improve interpretability at the individual prediction level, LIME (Local Interpretable Model-Agnostic Explanations) was applied to the best Logistic Regression model.  
# LIME explains a single prediction by approximating the model locally with an interpretable surrogate (e.g., linear model) and identifies the most influential features for each case.
# 

# In[30]:


#  Initialize the explainer
lime_explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=X_train_final,                # NumPy array
    feature_names=final_features,               # Feature names after transform
    class_names=le.classes_,                    # Class labels ("Mild", "Moderate+")
    mode='classification',
    discretize_continuous=True
)

#  Pick a few indices to explain
sample_indices = np.random.choice(X_test_final.shape[0], 3, replace=False)

#  Explain and display each
for i, idx in enumerate(sample_indices):
    print(f"\nLIME Explanation for Test Sample Index: {idx}")
    
    exp = lime_explainer.explain_instance(
        data_row=X_test_final[idx],
        predict_fn=logistic_model.predict_proba,
        num_features=8
    )
    
    exp.show_in_notebook(show_table=True)  # Or save as HTML
    # exp.save_to_file(f"lime_explanation_{i+1}.html")


# #### LIME Interpretation (Logistic Model – Sample Cases)
# **Sample 155**
# - Prediction: Moderate+ (0.56)
# - Moderate+ Influencers: Higher age (>0.54), low DFA, and low Inv_JitterAbs
# - Mild Influencers: Lower log_PPE, neutral Age_Sex_Interaction, and RPDE > 0.50
# - Interpretation: Slightly above-threshold age and lowered pitch dynamics (DFA) led the model toward Moderate+, despite signs of acoustic stability.
# 
# **Sample 1227**
# - Prediction: Moderate+ (0.88)
# - Moderate+ Influencers: Very high age (1.54), low DFA, high Age_Sex_Interaction
# - Mild Influencers: Low log_PPE and mildly elevated Shimmer_Slope
# - Interpretation: Clear prediction for Moderate+ due to multiple strong vocal and demographic indicators outweighing mild acoustic compensation.
# 
# **Sample 335**
# - Prediction: Mild (0.58)
# - Mild Influencers: Low age, very low Shimmer_Slope, and high Inv_JitterAbs
# - Moderate+ Factors: High RPDE, male sex, and elevated Age_Sex_Interaction
# - Interpretation: Despite structural voice issues, younger age and vocal regularity drove the model toward a Mild outcome.

# In[ ]:




