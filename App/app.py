# app.py
import streamlit as st
import pandas as pd
import joblib
import os
from sklearn.neighbors import KNeighborsClassifier

# Custom KNN class to match your original model
class ModelekNNOptimise(KNeighborsClassifier):
    pass  # This preserves your original model structure

# App configuration
st.set_page_config(page_title="🏦 Credit Scoring App", layout="wide")
st.title("🏦 Credit Scoring Prediction")
st.markdown("Predict client solvency using machine learning models")

# Model paths - EXACT match to your repository
MODEL_PATHS = {
    'logistic': 'models/logistic_model.pkl',
    'knn': 'models/ModeleKNNOptimise.pkl',  # Your custom KNN model
    'knn_standard': 'models/knn_model.pkl',  # Your standard KNN model
    'scaler': 'models/scaler.pkl'
}

# Load models with error handling
@st.cache_resource
def load_models():
    try:
        # Verify all files exist
        missing = [name for name, path in MODEL_PATHS.items() if not os.path.exists(path)]
        if missing:
            raise FileNotFoundError(f"Missing files: {missing}")
        
        # Load models
        models = {
            'logistic': joblib.load(MODEL_PATHS['logistic']),
            'knn': joblib.load(MODEL_PATHS['knn']),  # Custom KNN
            'scaler': joblib.load(MODEL_PATHS['scaler'])
        }
        
        # Try loading standard KNN if exists
        if os.path.exists(MODEL_PATHS['knn_standard']):
            models['knn_standard'] = joblib.load(MODEL_PATHS['knn_standard'])
        
        return models
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        st.error("Required files in 'models' folder:")
        st.error("- ModeleKNNOptimise.pkl (custom KNN)")
        st.error("- knn_model.pkl (standard KNN)")
        st.error("- logistic_model.pkl")
        st.error("- scaler.pkl")
        return None

models = load_models()
if models is None:
    st.stop()

# Input form with your exact feature names
with st.form("client_form"):
    st.header("Client Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        Age = st.number_input("Age", min_value=18, max_value=100, value=35)
        Marital_Status = st.selectbox("Marital Status", 
                                    ["Single", "Married", "Divorced"])
        Expenses = st.number_input("Monthly Expenses (€)", min_value=0, value=500)
    
    with col2:
        Income = st.number_input("Monthly Income (€)", min_value=0, value=2000)
        Amount = st.number_input("Loan Amount (€)", min_value=0, value=10000)
        Price = st.number_input("Purchase Value (€)", min_value=0, value=12000)
    
    submitted = st.form_submit_button("Predict Solvency")

# Prediction function
def predict_solvency(input_df, model_type='logistic'):
    try:
        scaled_input = models['scaler'].transform(input_df)
        
        if model_type == 'logistic':
            model = models['logistic']
        elif model_type == 'knn_custom':
            model = models['knn']
        else:
            model = models.get('knn_standard', models['knn'])
        
        prediction = model.predict(scaled_input)[0]
        proba = model.predict_proba(scaled_input)[0]
        return prediction, proba
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None, None

if submitted:
    # Prepare input data
    input_data = {
        'Age': Age,
        'Marital': 1 if Marital_Status == "Single" else 2 if Marital_Status == "Married" else 3,
        'Expenses': Expenses,
        'Income': Income,
        'Amount': Amount,
        'Price': Price
    }
    input_df = pd.DataFrame([input_data])
    
    # Make predictions
    log_pred, log_proba = predict_solvency(input_df, 'logistic')
    knn_pred, knn_proba = predict_solvency(input_df, 'knn_custom')
    
    # Display results
    st.subheader("Prediction Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Logistic Regression**")
        if log_pred == 1:
            st.error("Non-Solvent 🚨")
        else:
            st.success("Solvent ✅")
        st.metric("Confidence", f"{max(log_proba)*100:.1f}%")
        st.bar_chart(pd.DataFrame({'Probability': log_proba, 
                                 'Status': ['Solvent', 'Non-Solvent']}).set_index('Status'))
    
    with col2:
        st.markdown("**Custom KNN Model**")
        if knn_pred == 1:
            st.error("Non-Solvent 🚨")
        else:
            st.success("Solvent ✅")
        st.metric("Confidence", f"{max(knn_proba)*100:.1f}%")
        st.bar_chart(pd.DataFrame({'Probability': knn_proba, 
                                 'Status': ['Solvent', 'Non-Solvent']}).set_index('Status'))

# Debug section
with st.expander("Technical Details"):
    st.write("**Loaded Models:**")
    st.write(f"- Logistic Regression: {type(models['logistic'])}")
    st.write(f"- Custom KNN: {type(models['knn'])}")
    if 'knn_standard' in models:
        st.write(f"- Standard KNN: {type(models['knn_standard'])}")
    
    st.write("**Current Directory:**", os.listdir('.'))
    if os.path.exists('models'):
        st.write("**Models Folder Contents:**", os.listdir('models'))








