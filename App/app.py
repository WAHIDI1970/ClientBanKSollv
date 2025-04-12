# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Page Configuration
st.set_page_config(
    page_title="Client Solvency Predictor",
    layout="centered",
    page_icon="🏦"
)

# Title and Description
st.title("🏦 Client Solvency Prediction")
st.markdown("""
Compare predictions from both Logistic Regression and KNN models
""")

# Custom Class Definition (Critical for your KNN model)
class ModeleKNNOptimise:
    pass  # This matches your notebook's custom class

# Load Models with Error Handling
@st.cache_resource
def load_models():
    try:
        # Register custom class before loading
        joblib.register('ModeleKNNOptimise', ModeleKNNOptimise)
        
        return {
            'logistic': joblib.load("models/logistic_model.pkl"),
            'knn': joblib.load("models/KNN (1).pkl"),
            'scaler': joblib.load("models/scaler (1).pkl")
        }
    except Exception as e:
        st.error(f"❌ Model loading failed: {str(e)}")
        st.stop()

models = load_models()

# Client Input Form
with st.form("client_input"):
    st.header("🧾 Client Information")
    
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=30)
        marital = st.selectbox(
            "Marital Status",
            options=[1, 2, 3],
            format_func=lambda x: {1: "Single", 2: "Married", 3: "Divorced"}[x]
        )
        expenses = st.number_input("Monthly Expenses (€)", min_value=0.0, value=200.0)
    
    with col2:
        income = st.number_input("Monthly Income (€)", min_value=0.0, value=800.0)
        amount = st.number_input("Loan Amount (€)", min_value=0.0, value=1000.0)
        price = st.number_input("Purchase Value (€)", min_value=0.0, value=1200.0)
    
    submitted = st.form_submit_button("Predict Solvency")

# Prediction Function
def predict(client_data):
    try:
        # Scale data for logistic regression
        scaled_data = models['scaler'].transform(client_data)
        
        # Get predictions
        logistic_pred = models['logistic'].predict(scaled_data)[0]
        logistic_proba = models['logistic'].predict_proba(scaled_data)[0][1]
        
        knn_pred = models['knn'].predict(client_data)[0]
        knn_proba = models['knn'].predict_proba(client_data)[0][1]
        
        return {
            'logistic': {
                'pred': logistic_pred,
                'proba': logistic_proba,
                'label': "Non-Solvent" if logistic_pred == 1 else "Solvent"
            },
            'knn': {
                'pred': knn_pred,
                'proba': knn_proba,
                'label': "Non-Solvent" if knn_pred == 1 else "Solvent"
            }
        }
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        st.stop()

# Display Results
if submitted:
    client_data = pd.DataFrame({
        "Age": [age],
        "Marital": [marital],
        "Expenses": [expenses],
        "Income": [income],
        "Amount": [amount],
        "Price": [price]
    })
    
    results = predict(client_data)
    
    st.header("📊 Prediction Results")
    
    # Logistic Regression Results
    with st.expander("Logistic Regression", expanded=True):
        col1, col2 = st.columns([1, 3])
        with col1:
            if results['logistic']['pred'] == 1:
                st.error("🔴 Non-Solvent")
            else:
                st.success("🟢 Solvent")
        
        with col2:
            st.progress(results['logistic']['proba'])
            st.caption(f"Confidence: {results['logistic']['proba']:.1%}")
    
    # KNN Results
    with st.expander("KNN Model", expanded=True):
        col1, col2 = st.columns([1, 3])
        with col1:
            if results['knn']['pred'] == 1:
                st.error("🔴 Non-Solvent")
            else:
                st.success("🟢 Solvent")
        
        with col2:
            st.progress(results['knn']['proba'])
            st.caption(f"Confidence: {results['knn']['proba']:.1%}")
    
    # Comparison Summary
    st.divider()
    st.subheader("🔍 Model Comparison")
    
    if results['logistic']['pred'] == results['knn']['pred']:
        st.success("✅ Both models agree on the prediction")
    else:
        st.warning("⚠️ Models disagree on the prediction")
    
    # Client Data Display
    st.subheader("📋 Client Data Summary")
    st.dataframe(client_data.style.format({
        "Expenses": "€{:.2f}",
        "Income": "€{:.2f}",
        "Amount": "€{:.2f}",
        "Price": "€{:.2f}"
    }))

# Footer
st.caption("Note: Predictions are based on machine learning models and should be used as advisory only.")
