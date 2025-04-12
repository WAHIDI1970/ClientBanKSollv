# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Page config
st.set_page_config(
    page_title="Client Solvency Predictor", 
    layout="wide",
    page_icon="🏦"
)

# Title
st.title("🏦 Client Solvency Prediction Dashboard")
st.markdown("Compare predictions from Logistic Regression and KNN models")

# Sidebar - User Inputs
st.sidebar.header("📋 Client Information")

# Input fields with improved UX
col1, col2 = st.sidebar.columns(2)
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

# Model Loading with error handling
@st.cache_resource
def load_models():
    try:
        models = {
            "logistic": joblib.load("models/logistic_model.pkl"),
            "knn": joblib.load("models/KNN (1).pkl"),
            "scaler": joblib.load("models/scaler (1).pkl")
        }
        return models
    except Exception as e:
        st.error(f"❌ Model loading failed: {str(e)}")
        st.stop()

models = load_models()

# Prediction function
def make_predictions(data):
    try:
        # Scale features for logistic regression
        scaled_data = models['scaler'].transform(data)
        
        # Get predictions
        results = {}
        for name in ['logistic', 'knn']:
            if name == 'logistic':
                pred = models[name].predict(scaled_data)
                proba = models[name].predict_proba(scaled_data)
            else:
                pred = models[name].predict(data)
                proba = models[name].predict_proba(data)
                
            results[name] = {
                'prediction': pred[0],
                'probability': proba[0][1]  # Probability of class 1 (non-solvent)
            }
        return results
    except Exception as e:
        st.error(f"❌ Prediction error: {str(e)}")
        st.stop()

# Display results
if st.sidebar.button("🔮 Predict Solvency", type="primary"):
    with st.spinner("Analyzing client data..."):
        results = make_predictions(client_data)
        
        # Results columns
        col1, col2 = st.columns(2)
        
        # Logistic Regression Results
        with col1:
            st.subheader("Logistic Regression")
            if results['logistic']['prediction'] == 1:
                st.error(f"🚨 Non-Solvent (Probability: {results['logistic']['probability']:.1%})")
            else:
                st.success(f"✅ Solvent (Probability: {1 - results['logistic']['probability']:.1%})")
            
            # Confusion matrix visualization
            st.markdown("**Model Performance**")
            fig, ax = plt.subplots()
            ConfusionMatrixDisplay.from_estimator(
                models['logistic'], 
                models['scaler'].transform(client_data), 
                display_labels=['Solvent', 'Non-Solvent'],
                ax=ax
            )
            st.pyplot(fig)
        
        # KNN Results
        with col2:
            st.subheader("KNN Classifier")
            if results['knn']['prediction'] == 1:
                st.error(f"🚨 Non-Solvent (Probability: {results['knn']['probability']:.1%})")
            else:
                st.success(f"✅ Solvent (Probability: {1 - results['knn']['probability']:.1%})")
            
            # Feature importance placeholder
            st.markdown("**Feature Importance**")
            st.info("Note: KNN doesn't provide inherent feature importance")
        
        # Client data display
        st.divider()
        st.subheader("📊 Client Data Summary")
        st.dataframe(client_data.style.format({
            "Expenses": "€{:.2f}",
            "Income": "€{:.2f}", 
            "Amount": "€{:.2f}",
            "Price": "€{:.2f}"
        }))

        # Download results
        result_df = client_data.copy()
        for name in ['logistic', 'knn']:
            result_df[f'{name}_prediction'] = ['Non-Solvent' if results[name]['prediction'] == 1 else 'Solvent']
            result_df[f'{name}_probability'] = [f"{results[name]['probability']:.1%}"]
        
        csv = result_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Prediction Results",
            data=csv,
            file_name="client_solvency_prediction.csv",
            mime="text/csv"
        )

# Model comparison section
st.sidebar.divider()
st.sidebar.markdown("""
**ℹ️ Model Comparison**  
- Logistic Regression: Better for interpretability  
- KNN: Better for non-linear patterns  
""")
