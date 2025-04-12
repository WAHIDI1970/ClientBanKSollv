# app.py
import streamlit as st
import pandas as pd
import joblib
import os
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Set page configuration
st.set_page_config(
    page_title="Credit Scoring Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants
MODEL_DIR = "models"
MODEL_FILES = {
    'logistic': 'logistic_model.pkl',
    'knn': 'KNN (1).pkl',
    'scaler': 'scaler (1).pkl'
}

# Debug function to show file structure
def show_file_structure():
    """Display the current directory structure for debugging"""
    st.sidebar.header("File Structure Debug")
    base_dir = os.getcwd()
    st.sidebar.write(f"Current working directory: {base_dir}")
    
    st.sidebar.write("Looking for model files in:")
    for model_name, filename in MODEL_FILES.items():
        full_path = os.path.join(MODEL_DIR, filename)
        exists = "✅" if os.path.exists(full_path) else "❌"
        st.sidebar.write(f"{exists} {full_path}")

# Load models with error handling
@st.cache_resource
def load_models():
    """Load all required machine learning models"""
    try:
        show_file_structure()
        
        # Verify models directory exists
        if not os.path.exists(MODEL_DIR):
            raise FileNotFoundError(f"Directory '{MODEL_DIR}' not found")
        
        # Load each model
        models = {}
        for model_name, filename in MODEL_FILES.items():
            path = os.path.join(MODEL_DIR, filename)
            if not os.path.exists(path):
                raise FileNotFoundError(f"Model file not found: {path}")
            models[model_name] = joblib.load(path)
        
        return models['logistic'], models['knn'], models['scaler']
    
    except Exception as e:
        st.error(f"Model loading error: {str(e)}")
        show_file_structure()
        return None, None, None

# Load models
log_model, knn_model, scaler = load_models()

if None in [log_model, knn_model, scaler]:
    st.error("Failed to load required models. Please check the error above.")
    st.stop()

# Feature configuration
FEATURES = {
    'Age': {'type': 'number', 'min': 18, 'max': 100, 'default': 30},
    'Marital': {
        'type': 'select',
        'options': ['Single', 'Married', 'Divorced'],
        'mapping': {'Single': 1, 'Married': 2, 'Divorced': 3}
    },
    'Expenses': {'type': 'number', 'min': 0, 'default': 200},
    'Income': {'type': 'number', 'min': 0, 'default': 800},
    'Amount': {'type': 'number', 'min': 0, 'default': 1000},
    'Price': {'type': 'number', 'min': 0, 'default': 1200}
}

OUTLIER_COLUMNS = [
    'outlier_amount', 'outlier_expenses',
    'outlier_income', 'outlier_price'
]

TRAINING_FEATURES = list(FEATURES.keys()) + OUTLIER_COLUMNS

def create_input_form():
    """Create the user input form"""
    with st.form("client_input"):
        st.header("Client Information")
        
        col1, col2 = st.columns(2)
        form_data = {}
        
        with col1:
            form_data['Age'] = st.number_input(
                "Age",
                min_value=FEATURES['Age']['min'],
                max_value=FEATURES['Age']['max'],
                value=FEATURES['Age']['default']
            )
            
            marital_display = st.selectbox(
                "Marital Status",
                options=FEATURES['Marital']['options']
            )
            form_data['Marital'] = FEATURES['Marital']['mapping'][marital_display]
            
            form_data['Expenses'] = st.number_input(
                "Monthly Expenses (€)",
                min_value=FEATURES['Expenses']['min'],
                value=FEATURES['Expenses']['default']
            )
        
        with col2:
            form_data['Income'] = st.number_input(
                "Monthly Income (€)",
                min_value=FEATURES['Income']['min'],
                value=FEATURES['Income']['default']
            )
            
            form_data['Amount'] = st.number_input(
                "Loan Amount (€)",
                min_value=FEATURES['Amount']['min'],
                value=FEATURES['Amount']['default']
            )
            
            form_data['Price'] = st.number_input(
                "Purchase Value (€)",
                min_value=FEATURES['Price']['min'],
                value=FEATURES['Price']['default']
            )
        
        submitted = st.form_submit_button("Predict Solvency")
        return form_data if submitted else None

def prepare_input_data(form_data):
    """Prepare the input data for model prediction"""
    input_df = pd.DataFrame([form_data])
    
    # Add outlier columns initialized to False
    for col in OUTLIER_COLUMNS:
        input_df[col] = False
    
    return input_df[TRAINING_FEATURES]

def make_predictions(input_df):
    """Make predictions using both models"""
    try:
        # Scale features for logistic regression
        scaled_input = scaler.transform(input_df)
        
        # Get predictions
        log_pred = log_model.predict(scaled_input)[0]
        log_proba = log_model.predict_proba(scaled_input)[0][1]
        
        knn_pred = knn_model.predict(input_df)[0]
        knn_proba = knn_model.predict_proba(input_df)[0][1]
        
        return {
            'logistic': {
                'prediction': 'Non-Solvent' if log_pred == 1 else 'Solvent',
                'probability': log_proba,
                'class': log_pred
            },
            'knn': {
                'prediction': 'Non-Solvent' if knn_pred == 1 else 'Solvent',
                'probability': knn_proba,
                'class': knn_pred
            }
        }
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None

def display_results(results):
    """Display the prediction results"""
    st.header("Prediction Results")
    
    # Create columns for side-by-side comparison
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Logistic Regression")
        st.metric(
            "Client Status",
            results['logistic']['prediction'],
            f"{results['logistic']['probability']:.1%} confidence"
        )
        st.progress(results['logistic']['probability'])
    
    with col2:
        st.subheader("KNN Model")
        st.metric(
            "Client Status",
            results['knn']['prediction'],
            f"{results['knn']['probability']:.1%} confidence"
        )
        st.progress(results['knn']['probability'])
    
    # Visual comparison
    st.subheader("Model Confidence Comparison")
    fig, ax = plt.subplots(figsize=(8, 4))
    models = ['Logistic Regression', 'KNN']
    probas = [results['logistic']['probability'], results['knn']['probability']]
    colors = ['#1f77b4', '#ff7f0e']  # Blue and orange
    
    bars = ax.bar(models, probas, color=colors)
    ax.set_ylim(0, 1)
    ax.set_ylabel('Probability of Non-Solvency')
    ax.set_title('Model Confidence Comparison')
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1%}',
                ha='center', va='bottom')
    
    st.pyplot(fig)

def main():
    st.title("📊 Credit Scoring Dashboard")
    
    # Get user input
    form_data = create_input_form()
    
    if form_data:
        with st.spinner("Analyzing client data..."):
            # Prepare input data
            input_df = prepare_input_data(form_data)
            
            # Make predictions
            results = make_predictions(input_df)
        
        if results:
            display_results(results)

if __name__ == "__main__":
    main()
