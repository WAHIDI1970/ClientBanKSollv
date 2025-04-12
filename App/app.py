# app.py
import os
import streamlit as st
import pandas as pd
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

# Set up file paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'Data', 'scoring.sav')
MODEL_PATHS = {
    'logistic': os.path.join(BASE_DIR, 'models', 'logistic_model.pkl'),
    'knn': os.path.join(BASE_DIR, 'models', 'KNN (1).pkl'),
    'scaler': os.path.join(BASE_DIR, 'models', 'scaler (1).pkl')
}

@st.cache_resource
def load_resources():
    """Load all required resources with comprehensive error handling"""
    try:
        # Debug: Show current directory structure
        st.sidebar.write("Current directory:", os.listdir(BASE_DIR))
        if os.path.exists(os.path.join(BASE_DIR, 'data')):
            st.sidebar.write("Data directory:", os.listdir(os.path.join(BASE_DIR, 'data')))
        if os.path.exists(os.path.join(BASE_DIR, 'models')):
            st.sidebar.write("Models directory:", os.listdir(os.path.join(BASE_DIR, 'models')))

        # Check if all files exist
        missing_files = []
        for name, path in MODEL_PATHS.items():
            if not os.path.exists(path):
                missing_files.append(f"{name}: {path}")
        if not os.path.exists(DATA_PATH):
            missing_files.append(f"data: {DATA_PATH}")

        if missing_files:
            raise FileNotFoundError(f"Missing files:\n" + "\n".join(missing_files))

        # Load data and models
        df, meta = pyreadstat.read_sav(DATA_PATH)
        models = {
            'logistic': joblib.load(MODEL_PATHS['logistic']),
            'knn': joblib.load(MODEL_PATHS['knn']),
            'scaler': joblib.load(MODEL_PATHS['scaler'])
        }

        return df, models

    except Exception as e:
        st.error(f"LOADING ERROR: {str(e)}")
        st.error(f"Current working directory: {os.getcwd()}")
        return None, None

# Load resources
df, models = load_resources()

if df is not None and models is not None:
    st.success("✅ All files loaded successfully!")
    TRAINING_FEATURES = [col for col in df.columns if col != 'Statut1']
    st.session_state['features'] = TRAINING_FEATURES
else:
    st.stop()

def prepare_input(form_data):
    """Convert form data to model-ready format"""
    # Create DataFrame with all original features
    input_df = pd.DataFrame(columns=TRAINING_FEATURES)
    
    # Map form inputs to model features
    input_df['Age'] = [int(form_data['Age'])]
    input_df['Marital'] = [1 if form_data['Marital Status'] == 'Single' else 2]
    input_df['Expenses'] = [float(form_data['Monthly Expenses (€)'])]
    input_df['Income'] = [float(form_data['Monthly Income (€)'])]
    input_df['Amount'] = [float(form_data['Loan Amount (€)'])]
    input_df['Price'] = [float(form_data['Purchase Value (€)'])]
    
    # Initialize outlier columns to False
    outlier_cols = [col for col in TRAINING_FEATURES if col.startswith('outlier_')]
    for col in outlier_cols:
        input_df[col] = False
    
    return input_df[TRAINING_FEATURES]

def make_predictions(input_df):
    """Generate predictions from both models"""
    try:
        # Scale features for logistic regression
        scaled_input = models['scaler'].transform(input_df)
        
        # Get predictions
        logistic_pred = models['logistic'].predict(scaled_input)[0]
        logistic_proba = models['logistic'].predict_proba(scaled_input)[0][1]
        
        knn_pred = models['knn'].predict(input_df)[0]
        knn_proba = models['knn'].predict_proba(input_df)[0][1]
        
        return {
            'logistic': {
                'prediction': 'Non-Solvent' if logistic_pred == 1 else 'Solvent',
                'probability': logistic_proba,
                'class': logistic_pred
            },
            'knn': {
                'prediction': 'Non-Solvent' if knn_pred == 1 else 'Solvent',
                'probability': knn_proba,
                'class': knn_pred
            }
        }
    except Exception as e:
        st.error(f"PREDICTION ERROR: {str(e)}")
        return None

def main():
    st.title("📊 Credit Scoring Dashboard")
    
    # Debug information in sidebar
    if st.sidebar.checkbox("Show debug info"):
        st.sidebar.write("**Training Features:**", TRAINING_FEATURES)
        st.sidebar.write("**First row of training data:**")
        st.sidebar.json(df.iloc[0].to_dict())

    # Client information form
    with st.form("client_info"):
        st.header("Client Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.number_input("Age", min_value=18, max_value=100, value=30)
            marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
            expenses = st.number_input("Monthly Expenses (€)", min_value=0.0, value=200.0, step=10.0)
        
        with col2:
            income = st.number_input("Monthly Income (€)", min_value=0.0, value=800.0, step=100.0)
            loan_amount = st.number_input("Loan Amount (€)", min_value=0.0, value=1000.0, step=100.0)
            purchase_value = st.number_input("Purchase Value (€)", min_value=0.0, value=1200.0, step=100.0)
        
        submitted = st.form_submit_button("Predict Solvency")
    
    if submitted:
        # Prepare form data
        form_data = {
            'Age': age,
            'Marital Status': marital_status,
            'Monthly Expenses (€)': expenses,
            'Monthly Income (€)': income,
            'Loan Amount (€)': loan_amount,
            'Purchase Value (€)': purchase_value
        }
        
        with st.spinner("Analyzing client data..."):
            # 1. Prepare input data
            input_df = prepare_input(form_data)
            
            # 2. Make predictions
            results = make_predictions(input_df)
        
        if results:
            st.header("Prediction Results")
            
            # Display results in columns
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
            st.subheader("Model Comparison")
            fig, ax = plt.subplots(figsize=(8, 4))
            models = ['Logistic Regression', 'KNN']
            probas = [results['logistic']['probability'], results['knn']['probability']]
            ax.bar(models, probas, color=['#1f77b4', '#ff7f0e'])
            ax.set_ylim(0, 1)
            ax.set_ylabel('Probability of Non-Solvency')
            ax.set_title('Model Confidence Comparison')
            st.pyplot(fig)

if __name__ == "__main__":
    main()
