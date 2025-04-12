# app.py
import streamlit as st
import pandas as pd
import joblib
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Set page configuration
st.set_page_config(
    page_title="🏦 Credit Scoring App",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🏦 Credit Scoring App")
st.info("This app predicts client solvency using logistic regression and KNN models")

# File paths - adjust these if your files are named differently
MODEL_FILES = {
    'logistic': 'models/logistic_model.pkl',
    'knn': 'models/knn_model.pkl',  # Renamed from KNN (1).pkl
    'scaler': 'models/scaler.pkl'   # Renamed from scaler (1).pkl
}

# Load models with error handling
@st.cache_resource
def load_models():
    try:
        # First, try loading with standard classes
        models = {
            'logistic': joblib.load(MODEL_FILES['logistic']),
            'knn': joblib.load(MODEL_FILES['knn']),
            'scaler': joblib.load(MODEL_FILES['scaler'])
        }
        return models
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        st.error("Please ensure:")
        st.error("1. Model files exist in the 'models' folder")
        st.error("2. Files are named correctly (knn_model.pkl, logistic_model.pkl, scaler.pkl)")
        st.error("3. No custom class requirements exist")
        return None

# Load models
models = load_models()
if models is None:
    st.stop()

# Feature configuration
FEATURES = {
    'Age': {'type': 'number', 'min': 18, 'max': 100, 'default': 30},
    'Marital Status': {
        'type': 'select',
        'options': ['Single', 'Married', 'Divorced'],
        'mapping': {'Single': 1, 'Married': 2, 'Divorced': 3}
    },
    'Monthly Expenses (€)': {'type': 'number', 'min': 0, 'default': 200},
    'Monthly Income (€)': {'type': 'number', 'min': 0, 'default': 800},
    'Loan Amount (€)': {'type': 'number', 'min': 0, 'default': 1000},
    'Purchase Value (€)': {'type': 'number', 'min': 0, 'default': 1200}
}

# Input form in sidebar
with st.sidebar:
    st.header("Client Information")
    input_data = {}
    
    for feature, config in FEATURES.items():
        if config['type'] == 'number':
            input_data[feature] = st.number_input(
                feature,
                min_value=config['min'],
                max_value=config.get('max', None),
                value=config['default']
            )
        elif config['type'] == 'select':
            selected = st.selectbox(feature, options=config['options'])
            input_data[feature] = config['mapping'][selected]

# Prepare input data
def prepare_input(form_data):
    features = {
        'Age': form_data['Age'],
        'Marital': form_data['Marital Status'],
        'Expenses': form_data['Monthly Expenses (€)'],
        'Income': form_data['Monthly Income (€)'],
        'Amount': form_data['Loan Amount (€)'],
        'Price': form_data['Purchase Value (€)']
    }
    return pd.DataFrame([features])

# Make predictions
def predict(input_df):
    try:
        # Scale features
        scaled_input = models['scaler'].transform(input_df)
        
        # Get predictions
        log_pred = models['logistic'].predict(scaled_input)[0]
        log_proba = models['logistic'].predict_proba(scaled_input)[0]
        
        knn_pred = models['knn'].predict(input_df)[0]
        knn_proba = models['knn'].predict_proba(input_df)[0]
        
        return {
            'logistic': {
                'prediction': log_pred,
                'probability': log_proba[1] if log_pred == 1 else log_proba[0],
                'proba': log_proba
            },
            'knn': {
                'prediction': knn_pred,
                'probability': knn_proba[1] if knn_pred == 1 else knn_proba[0],
                'proba': knn_proba
            }
        }
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None

# Display results
def display_results(results):
    st.subheader("Prediction Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Logistic Regression**")
        pred = "Non-Solvent" if results['logistic']['prediction'] == 1 else "Solvent"
        st.metric("Status", pred, f"{results['logistic']['probability']:.1%}")
        st.bar_chart(pd.DataFrame({
            'Probability': results['logistic']['proba'],
            'Class': ['Solvent', 'Non-Solvent']
        }).set_index('Class'))
    
    with col2:
        st.markdown("**KNN Model**")
        pred = "Non-Solvent" if results['knn']['prediction'] == 1 else "Solvent"
        st.metric("Status", pred, f"{results['knn']['probability']:.1%}")
        st.bar_chart(pd.DataFrame({
            'Probability': results['knn']['proba'],
            'Class': ['Solvent', 'Non-Solvent']
        }).set_index('Class'))

# Main app flow
if st.sidebar.button("Predict Solvency"):
    with st.spinner("Analyzing..."):
        input_df = prepare_input(input_data)
        results = predict(input_df)
        
        if results:
            display_results(results)
            with st.expander("View Input Data"):
                st.dataframe(input_df)
else:
    st.info("Please enter client details and click 'Predict Solvency'")

# Debug section
with st.expander("Technical Details"):
    st.write("**Loaded Models:**")
    st.write(f"- Logistic Regression: {type(models['logistic'])}")
    st.write(f"- KNN: {type(models['knn'])}")
    st.write(f"- Scaler: {type(models['scaler'])}")
    
    st.write("**Expected Features:**")
    st.write(list(FEATURES.keys()))
