# app.py
import streamlit as st
import pandas as pd
import numpy as np
import pyreadstat
import joblib
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Set page config
st.set_page_config(
    page_title="Credit Scoring App",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def load_data_and_models():
    """Load the original .sav file and trained models"""
    try:
        # 1. Load the original scoring data to get exact feature names
        df, meta = pyreadstat.read_sav('scoring.sav')
        
        # 2. Load trained models
        log_model = joblib.load('logistic_model.joblib')
        best_knn = joblib.load('KNN (1).joblib')
        scaler = joblib.load('scaler (1).joblib')
        
        return df, log_model, best_knn, scaler
    except Exception as e:
        st.error(f"Loading error: {str(e)}")
        return None, None, None, None

# Load data and models
df, log_model, best_knn, scaler = load_data_and_models()

if df is not None:
    # Get exact feature names used in training (excluding target)
    TRAINING_FEATURES = [col for col in df.columns if col != 'Statut1']
    st.session_state['training_features'] = TRAINING_FEATURES

def prepare_input(form_data):
    """Prepare input data matching original training structure"""
    # Create a DataFrame with all original features initialized properly
    input_df = pd.DataFrame(columns=TRAINING_FEATURES)
    
    # Map form inputs to model features
    mapping = {
        'Age': 'Age',
        'Monthly Income (€)': 'Income',
        'Marital Status': 'Marital',  # Will convert to numeric
        'Loan Amount (€)': 'Amount',
        'Monthly Expenses (€)': 'Expenses',
        'Purchase Value (€)': 'Price'
    }
    
    # Convert marital status to original encoding
    marital_map = {'Single': 1, 'Married': 2, 'Divorced': 3}
    
    # Set values for known features
    for form_field, model_feature in mapping.items():
        if form_field == 'Marital Status':
            input_df[model_feature] = [marital_map[form_data[form_field]]]
        else:
            input_df[model_feature] = [float(form_data[form_field])]
    
    # Initialize outlier columns to False (adjust if your model needs different defaults)
    outlier_cols = [col for col in TRAINING_FEATURES if col.startswith('outlier_')]
    for col in outlier_cols:
        input_df[col] = False
    
    return input_df[TRAINING_FEATURES]  # Ensure correct column order

def make_predictions(input_df):
    """Make predictions using both models"""
    try:
        # Logistic Regression (requires scaling)
        scaled_input = scaler.transform(input_df)
        log_pred = log_model.predict(scaled_input)[0]
        log_proba = log_model.predict_proba(scaled_input)[0][1]  # P(Non-Solvent)
        
        # KNN (uses raw features)
        knn_pred = best_knn.predict(input_df)[0]
        knn_proba = best_knn.predict_proba(input_df)[0][1]  # P(Non-Solvent)
        
        return {
            'logistic': {
                'prediction': 'Non-Solvent' if log_pred == 1 else 'Solvent',
                'probability': log_proba
            },
            'knn': {
                'prediction': 'Non-Solvent' if knn_pred == 1 else 'Solvent',
                'probability': knn_proba
            }
        }
    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")
        return None

# Main app
def main():
    st.title("📊 Credit Scoring Dashboard")
    
    # Display original feature names for debugging
    if st.sidebar.checkbox("Show technical details"):
        st.sidebar.write("**Training Features:**", TRAINING_FEATURES)
        st.sidebar.write("**Example Data Row:**", df.iloc[0].to_dict())
    
    # Client form
    with st.form("client_info"):
        st.header("Client Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.number_input("Age", min_value=18, max_value=100, value=30)
            marital = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
            expenses = st.number_input("Monthly Expenses (€)", min_value=0.0, value=200.0, step=10.0)
        
        with col2:
            income = st.number_input("Monthly Income (€)", min_value=0.0, value=800.0, step=100.0)
            amount = st.number_input("Loan Amount (€)", min_value=0.0, value=1000.0, step=100.0)
            price = st.number_input("Purchase Value (€)", min_value=0.0, value=1200.0, step=100.0)
        
        submitted = st.form_submit_button("Predict Solvency")
    
    if submitted:
        form_data = {
            'Age': age,
            'Marital Status': marital,
            'Monthly Expenses (€)': expenses,
            'Monthly Income (€)': income,
            'Loan Amount (€)': amount,
            'Purchase Value (€)': price
        }
        
        with st.spinner("Analyzing client data..."):
            # 1. Prepare input data
            input_df = prepare_input(form_data)
            
            # 2. Make predictions
            results = make_predictions(input_df)
        
        if results:
            st.header("Prediction Results")
            
            # Display results
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Logistic Regression")
                st.metric(
                    "Status",
                    results['logistic']['prediction'],
                    f"{results['logistic']['probability']:.1%} confidence"
                )
                st.progress(results['logistic']['probability'])
            
            with col2:
                st.subheader("KNN Classifier")
                st.metric(
                    "Status",
                    results['knn']['prediction'],
                    f"{results['knn']['probability']:.1%} confidence"
                )
                st.progress(results['knn']['probability'])
            
            # Visual comparison
            fig, ax = plt.subplots(figsize=(8, 4))
            models = ['Logistic', 'KNN']
            probas = [results['logistic']['probability'], results['knn']['probability']]
            ax.bar(models, probas, color=['#1f77b4', '#ff7f0e'])
            ax.set_ylim(0, 1)
            ax.set_ylabel('Probability of Non-Solvency')
            ax.set_title('Model Comparison')
            st.pyplot(fig)

if __name__ == "__main__":
    main()

