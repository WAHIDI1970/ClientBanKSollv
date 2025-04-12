# app.py
import streamlit as st
import pandas as pd
import joblib
import os

# App Configuration
st.set_page_config(page_title="🏦 Credit Solvency Predictor", layout="wide")
st.title("🏦 Credit Solvency Prediction")
st.markdown("Predict client solvency using logistic regression model")

# Load Model Function
@st.cache_resource
def load_models():
    try:
        # Load with exact file names from your structure
        model = joblib.load('models/logistic_model.pkl')
        scaler = joblib.load('models/scaler (1).pkl')  # Your exact scaler filename
        return model, scaler
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        st.error("Required files in 'models' folder:")
        st.error("- logistic_model.pkl")
        st.error("- scaler (1).pkl")
        return None, None

model, scaler = load_models()
if model is None:
    st.stop()

# Input Form with Your Exact Feature Names
with st.form("credit_form"):
    st.header("Client Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        Age = st.number_input("Age", min_value=18, max_value=100, value=30)
        Marital_Status = st.selectbox("Marital Status", 
                                    ["Single", "Married", "Divorced"])
        Expenses = st.number_input("Monthly Expenses (€)", min_value=0, value=200)
    
    with col2:
        Income = st.number_input("Monthly Income (€)", min_value=0, value=800)
        Amount = st.number_input("Loan Amount (€)", min_value=0, value=1000)
        Price = st.number_input("Purchase Value (€)", min_value=0, value=1200)
    
    submitted = st.form_submit_button("Predict Solvency")

# Prediction Logic
if submitted:
    # Prepare input with correct variable names
    input_data = {
        'Age': Age,
        'Marital': 1 if Marital_Status == "Single" else 2 if Marital_Status == "Married" else 3,
        'Expenses': Expenses,
        'Income': Income,
        'Amount': Amount,
        'Price': Price
    }
    
    # Create DataFrame and scale
    input_df = pd.DataFrame([input_data])
    scaled_input = scaler.transform(input_df)
    
    # Make prediction
    prediction = model.predict(scaled_input)[0]
    probabilities = model.predict_proba(scaled_input)[0]
    
    # Display Results
    st.subheader("Prediction Result")
    
    if prediction == 1:
        st.error("Non-Solvent (High Risk)")
    else:
        st.success("Solvent (Low Risk)")
    
    # Probability visualization
    prob_df = pd.DataFrame({
        'Probability': probabilities,
        'Status': ['Solvent', 'Non-Solvent']
    }).set_index('Status')
    
    st.bar_chart(prob_df)
    st.metric("Confidence Score", 
             f"{max(probabilities)*100:.1f}%", 
             f"Difference: {abs(probabilities[0]-probabilities[1])*100:.1f}%")

# Debug section
with st.expander("Debug Info"):
    st.write("Model loaded:", type(model))
    st.write("Scaler loaded:", type(scaler))
    st.write("Expected features:", list(input_data.keys()))








