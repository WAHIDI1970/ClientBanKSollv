# app.py
import streamlit as st
import pandas as pd
import joblib
import os
from sklearn.preprocessing import StandardScaler

# Define custom KNN class if needed
class ModelekNNOptimise:
    pass  # This should match your original KNN model class

# Set page configuration
st.title('🏦 Credit Scoring App')
st.info('This app predicts client solvency using logistic regression and KNN models')

# File paths
MODEL_DIR = "models"
MODEL_FILES = {
    'logistic': 'logistic_model.pkl',
    'knn': 'KNN (1).pkl',
    'scaler': 'scaler (1).pkl'
}

# Load models
@st.cache_resource
def load_models():
    try:
        models = {}
        for model_name, filename in MODEL_FILES.items():
            path = os.path.join(MODEL_DIR, filename)
            models[model_name] = joblib.load(path)
        return models
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None

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

OUTLIER_COLS = [
    'outlier_amount', 'outlier_expenses',
    'outlier_income', 'outlier_price'
]

# Input form in sidebar
with st.sidebar:
    st.header('Client Information')
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
    # Create DataFrame
    df = pd.DataFrame([{
        'Age': form_data['Age'],
        'Marital': form_data['Marital Status'],
        'Expenses': form_data['Monthly Expenses (€)'],
        'Income': form_data['Monthly Income (€)'],
        'Amount': form_data['Loan Amount (€)'],
        'Price': form_data['Purchase Value (€)'],
        **{col: False for col in OUTLIER_COLS}
    }])
    return df

# Make predictions
def predict(input_df):
    try:
        # Scale features
        scaled_input = models['scaler'].transform(input_df)
        
        # Logistic Regression prediction
        log_pred = models['logistic'].predict(scaled_input)[0]
        log_proba = models['logistic'].predict_proba(scaled_input)[0]
        
        # KNN prediction
        knn_pred = models['knn'].predict(input_df)[0]
        knn_proba = models['knn'].predict_proba(input_df)[0]
        
        return {
            'logistic': {
                'prediction': log_pred,
                'probability': log_proba[1] if log_pred == 1 else log_proba[0],
                'proba_array': log_proba
            },
            'knn': {
                'prediction': knn_pred,
                'probability': knn_proba[1] if knn_pred == 1 else knn_proba[0],
                'proba_array': knn_proba
            }
        }
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None

# Display results
def display_results(results):
    st.subheader('Prediction Results')
    
    # Create tabs for each model
    tab1, tab2 = st.tabs(["Logistic Regression", "KNN Model"])
    
    with tab1:
        st.write("**Logistic Regression Prediction**")
        pred_text = "Non-Solvent" if results['logistic']['prediction'] == 1 else "Solvent"
        st.success(f"Predicted Status: {pred_text}")
        
        # Probability bars
        st.write("**Probability**")
        proba_df = pd.DataFrame({
            'Status': ['Solvent', 'Non-Solvent'],
            'Probability': results['logistic']['proba_array']
        })
        st.bar_chart(proba_df.set_index('Status'))
        
    with tab2:
        st.write("**KNN Model Prediction**")
        pred_text = "Non-Solvent" if results['knn']['prediction'] == 1 else "Solvent"
        st.success(f"Predicted Status: {pred_text}")
        
        # Probability bars
        st.write("**Probability**")
        proba_df = pd.DataFrame({
            'Status': ['Solvent', 'Non-Solvent'],
            'Probability': results['knn']['proba_array']
        })
        st.bar_chart(proba_df.set_index('Status'))
    
    # Comparison
    st.subheader("Model Comparison")
    compare_df = pd.DataFrame({
        'Model': ['Logistic Regression', 'KNN'],
        'Non-Solvent Probability': [
            results['logistic']['proba_array'][1],
            results['knn']['proba_array'][1]
        ]
    })  # Fixed the bracket issue here
    st.bar_chart(compare_df.set_index('Model'))

# Main app flow
if st.sidebar.button('Predict Solvency'):
    with st.spinner('Making predictions...'):
        # Prepare input
        input_df = prepare_input(input_data)
        
        # Make predictions
        results = predict(input_df)
        
        # Display results
        if results:
            display_results(results)
            
            # Show raw input data
            with st.expander("View Input Data"):
                st.dataframe(input_df.drop(columns=OUTLIER_COLS))
else:
    st.info("Please enter client information in the sidebar and click 'Predict Solvency'")

# Debug information
with st.expander("Debug Information"):
    st.write("**Looking for model files in:**")
    for model_name, filename in MODEL_FILES.items():
        path = os.path.join(MODEL_DIR, filename)
        exists = "✅ Found" if os.path.exists(path) else "❌ Missing"
        st.write(f"- {filename}: {exists}")
    
    if models:
        st.write("**Models loaded successfully**")
        st.write(f"- Logistic Regression: {type(models['logistic'])}")
        st.write(f"- KNN Model: {type(models['knn'])}")
        st.write(f"- Scaler: {type(models['scaler'])}")
