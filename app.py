import streamlit as st
import pandas as pd
import numpy as np
import joblib
from catboost import CatBoostClassifier, Pool

# =========================
# Load Models, Metrics, and Data
# =========================

@st.cache_resource
def load_model(file_path):
    """Loads the saved model."""
    if file_path.endswith('.cbm'):
        model = CatBoostClassifier()
        model.load_model(file_path, format="cbm")
    else:
        model = joblib.load(file_path)
    return model

@st.cache
def load_metrics(file_path):
    """Loads the saved metrics."""
    return np.load(file_path, allow_pickle=True).item()

@st.cache
def load_data(file_path):
    """Loads the customer dataset."""
    return pd.read_excel(file_path)

# =========================
# Prediction Function
# =========================

def make_prediction(model, features, is_catboost=False, categorical_columns=None):
    """Generates the prediction and probability."""
    if is_catboost:
        pool = Pool(features, cat_features=categorical_columns)
        proba = model.predict_proba(pool)[:, 1]
    else:
        proba = model.predict_proba(features)[:, 1]
    prediction = proba >= 0.5
    return prediction, proba

# =========================
# Streamlit App
# =========================

# App Title
st.title("Customer Purchase Prediction App")

# Load Dataset
data_file = "customer_data_with_cltv.xlsx"
customer_data = load_data(data_file)

# Display Dataset Preview
st.sidebar.header("Dataset Preview")
st.sidebar.dataframe(customer_data.head())

# Model Selection
st.sidebar.header("Select Machine Learning Model")
model_options = {
    "Lasso Logistic Regression": {
        "model_path": "best_logistic_pipeline.pkl",
        "metrics_path": "logistic_pipeline_results.npy",
        "is_catboost": False
    },
    "Logistic Regression (L2)": {
        "model_path": "best_lr_model.pkl",
        "metrics_path": "lr_model_results.npy",
        "is_catboost": False
    },
    "Calibrated SVC": {
        "model_path": "best_svc_pipeline.pkl",
        "metrics_path": "svc_pipeline_results.npy",
        "is_catboost": False
    },
    "Decision Tree": {
        "model_path": "best_decision_tree_pipeline.pkl",
        "metrics_path": "decision_tree_pipeline_results.npy",
        "is_catboost": False
    },
    "CatBoost": {
        "model_path": "best_catboost_model.cbm",
        "metrics_path": "catboost_model_results.npy",
        "is_catboost": True
    }
}

selected_model_name = st.sidebar.selectbox("Select Model", list(model_options.keys()))
selected_model_details = model_options[selected_model_name]

# Load the Selected Model and Metrics
model = load_model(selected_model_details["model_path"])
metrics = load_metrics(selected_model_details["metrics_path"])

# Display Model Metrics
st.sidebar.header("Model Metrics")
st.sidebar.write(f"**Best Threshold:** {metrics['best_threshold']:.2f}")
st.sidebar.write(f"**Precision-Recall AUC (Validation):** {metrics['pr_auc_val']:.2f}")
st.sidebar.write(f"**Precision-Recall AUC (Test):** {metrics['pr_auc_test']:.2f}")
st.sidebar.write(f"**ROC AUC (Test):** {metrics['roc_auc_test']:.2f}")

# Select Customer
st.header("Predict Customer Purchase Likelihood")
customer_id = st.selectbox("Select Customer", customer_data["CUSTOMERNAME"].unique())
customer_row = customer_data[customer_data["CUSTOMERNAME"] == customer_id]

# Drop columns not needed for prediction
drop_columns = ["CUSTOMERNAME", "Purchase Probability"]
categorical_columns = ["Most Frequent Item_Group", "Most Frequent Trend_Classification", "Customer_Lifetime_Category"]
features = customer_row.drop(columns=drop_columns, errors="ignore")

# Prediction
if st.button("Predict"):
    is_catboost = selected_model_details["is_catboost"]
    prediction, proba = make_prediction(model, features, is_catboost, categorical_columns)
    prediction_result = "Yes" if prediction[0] else "No"
    st.write(f"**Prediction Outcome:** {prediction_result}")
    st.write(f"**Likelihood of Purchase:** {proba[0] * 100:.2f}%")
