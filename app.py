import streamlit as st
import pandas as pd
import numpy as np
import joblib

# =========================
# Load Models, Metrics, and Data
# =========================

@st.cache_resource
def load_model(file_path):
    """Loads the saved model."""
    return joblib.load(file_path)

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

def make_prediction(model, features):
    """Generates the prediction and probability."""
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
    },
    "Logistic Regression (L2)": {
        "model_path": "best_lr_model.pkl",
        "metrics_path": "lr_model_results.npy",
    },
    "Calibrated SVC": {
        "model_path": "best_svc_pipeline.pkl",
        "metrics_path": "svc_pipeline_results.npy",
    },
    "Decision Tree": {
        "model_path": "best_decision_tree_pipeline.pkl",
        "metrics_path": "decision_tree_pipeline_results.npy",
    }
}

selected_model_name = st.sidebar.selectbox("Select Model", list(model_options.keys()))
selected_model_details = model_options[selected_model_name]

# Load the Selected Model and Metrics
model = load_model(selected_model_details["model_path"])
metrics = load_metrics(selected_model_details["metrics_path"])

# Display Model Metrics
st.sidebar.header("Model Metrics")
st.sidebar.write(f"**Best Threshold:** {metrics.get('best_threshold', 'N/A')}")
st.sidebar.write(f"**Precision-Recall AUC (Test):** {metrics.get('pr_auc_test', 'N/A')}")
st.sidebar.write(f"**ROC AUC (Test):** {metrics.get('roc_auc_test', 'N/A')}")

# Select Customer
st.header("Predict Customer Purchase Likelihood")
customer_id = st.selectbox("Select Customer", customer_data["CUSTOMERNAME"].unique())
customer_row = customer_data[customer_data["CUSTOMERNAME"] == customer_id]

# Drop columns not needed for prediction
drop_columns = ["CUSTOMERNAME", "Purchase Probability"]
features = customer_row.drop(columns=drop_columns, errors="ignore")

# Prediction
if st.button("Predict"):
    prediction, proba = make_prediction(model, features)
    prediction_result = "Yes" if prediction[0] else "No"
    st.write(f"**Prediction Outcome:** {prediction_result}")
    st.write(f"**Likelihood of Purchase:** {proba[0] * 100:.2f}%")
