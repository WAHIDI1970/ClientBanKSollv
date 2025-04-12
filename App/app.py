# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import joblib
from sklearn.neighbors import KNeighborsClassifier

# =============================================
# FIX FOR CUSTOM KNN CLASS
# =============================================
class ModeleKNNOptimise(KNeighborsClassifier):
    """Wrapper class to match your notebook's custom KNN model"""
    pass

# =============================================
# PAGE CONFIGURATION
# =============================================
st.set_page_config(
    page_title="Client Solvency Predictor",
    layout="wide",
    page_icon="🏦"
)

st.title("🏦 Client Solvency Prediction Dashboard")
st.markdown("Compare predictions from Logistic Regression and KNN models")

# =============================================
# MODEL LOADING
# =============================================
@st.cache_resource
def load_models():
    try:
        logistic_model = joblib.load("models/logistic_model.pkl")
        knn_model = joblib.load("models/KNN (1).pkl")
        scaler = joblib.load("models/scaler (1).pkl")
        return {"logistic": logistic_model, "knn": knn_model, "scaler": scaler}
    except Exception as e:
        st.error(f"❌ Model loading failed: {str(e)}")
        st.stop()

models = load_models()

# =============================================
# USER INPUT FORM
# =============================================
st.sidebar.header("📋 Client Information")
col1, col2 = st.sidebar.columns(2)

# Input fields for features
with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    marital = st.selectbox(
        "Marital Status",
        options=[1, 2, 3],
        format_func=lambda x: {1: "Single", 2: "Married", 3: "Divorced"}[x]
    )
    expenses = st.number_input("Monthly Expenses (€)", min_value=0.0, value=200.0, step=50.0)

with col2:
    income = st.number_input("Monthly Income (€)", min_value=0.0, value=800.0, step=50.0)
    amount = st.number_input("Loan Amount (€)", min_value=0.0, value=1000.0, step=100.0)
    price = st.number_input("Purchase Value (€)", min_value=0.0, value=1200.0, step=100.0)

# Prepare input data
client_data = pd.DataFrame({
    "Age": [age],
    "Marital": [marital],
    "Expenses": [expenses],
    "Income": [income],
    "Amount": [amount],
    "Price": [price]
})

# =============================================
# PREDICTION FUNCTION
# =============================================
def make_predictions(data):
    try:
        # Scale features for logistic regression
        scaled_data = models['scaler'].transform(data)
        
        results = {
            "logistic": {
                "prediction": models['logistic'].predict(scaled_data)[0],
                "probability": models['logistic'].predict_proba(scaled_data)[0][1]
            },
            "knn": {
                "prediction": models['knn'].predict(data)[0],
                "probability": models['knn'].predict_proba(data)[0][1]
            }
        }
        return results
    except Exception as e:
        st.error(f"❌ Prediction error: {str(e)}")
        st.stop()

# =============================================
# RESULTS DISPLAY
# =============================================
if st.sidebar.button("🔮 Predict Solvency", type="primary"):
    with st.spinner("Analyzing client data..."):
        results = make_predictions(client_data)
        
        col1, col2 = st.columns(2)
        
        # Logistic Regression Results
        with col1:
            st.subheader("📈 Logistic Regression")
            if results['logistic']['prediction'] == 1:
                st.error(f"🚨 Non-Solvent (Confidence: {results['logistic']['probability']:.1%})")
            else:
                st.success(f"✅ Solvent (Confidence: {1-results['logistic']['probability']:.1%})")

        # KNN Results
        with col2:
            st.subheader("🔢 KNN Classifier")
            if results['knn']['prediction'] == 1:
                st.error(f"🚨 Non-Solvent (Confidence: {results['knn']['probability']:.1%})")
            else:
                st.success(f"✅ Solvent (Confidence: {1-results['knn']['probability']:.1%})")

        st.divider()
        if results['logistic']['prediction'] == results['knn']['prediction']:
            st.success("🎯 Both models agree on the prediction")
        else:
            st.warning("⚠️ Models disagree - consider manual review")
