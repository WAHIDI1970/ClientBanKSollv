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

# Set up EXACT file paths matching your repository
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PATHS = {
    'data': os.path.join(BASE_DIR, 'Data', 'scoring.sav'),  # Matches your 'Data' folder
    'logistic': os.path.join(BASE_DIR, 'models', 'logistic_model.pkl'),
    'knn': os.path.join(BASE_DIR, 'models', 'KNN (1).pkl'),  # Exact match with space and (1)
    'scaler': os.path.join(BASE_DIR, 'models', 'scaler (1).pkl')  # Exact match
}

@st.cache_resource
def load_resources():
    """Load all required resources with exact path matching"""
    try:
        # Debug: Show what we're looking for
        st.sidebar.markdown("**Searching for files at:**")
        for name, path in PATHS.items():
            st.sidebar.write(f"{name}: {path}")
        
        # Verify all paths exist
        missing = []
        for name, path in PATHS.items():
            if not os.path.exists(path):
                missing.append(f"{name} ({path})")
        
        if missing:
            raise FileNotFoundError(f"Missing files:\n" + "\n".join(missing))
        
        # Load data and models
        df, meta = pyreadstat.read_sav(PATHS['data'])
        models = {
            'logistic': joblib.load(PATHS['logistic']),
            'knn': joblib.load(PATHS['knn']),
            'scaler': joblib.load(PATHS['scaler'])
        }
        
        return df, models

    except Exception as e:
        st.error("CRITICAL LOADING ERROR")
        st.error(str(e))
        st.error("Current directory structure:")
        st.code(os.listdir(BASE_DIR))
        if os.path.exists(os.path.join(BASE_DIR, 'Data')):
            st.error("Contents of Data folder:")
            st.code(os.listdir(os.path.join(BASE_DIR, 'Data')))
        if os.path.exists(os.path.join(BASE_DIR, 'models')):
            st.error("Contents of models folder:")
            st.code(os.listdir(os.path.join(BASE_DIR, 'models')))
        return None, None

# Load resources - will stop if fails
df, models = load_resources()
if df is None or models is None:
    st.stop()

# Get feature names (excluding target)
TRAINING_FEATURES = [col for col in df.columns if col != 'Statut1']

def prepare_input(form_data):
    """Convert form data to model input format"""
    input_df = pd.DataFrame(columns=TRAINING_FEATURES)
    
    # Map form fields to model features
    input_df['Age'] = [int(form_data['Age'])]
    input_df['Marital'] = [1 if form_data['Marital Status'] == 'Single' else 2]
    input_df['Expenses'] = [float(form_data['Monthly Expenses (€)'])]
    input_df['Income'] = [float(form_data['Monthly Income (€)'])]
    input_df['Amount'] = [float(form_data['Loan Amount (€)'])]
    input_df['Price'] = [float(form_data['Purchase Value (€)'])]
    
    # Handle outlier columns if they exist
    for col in [c for c in TRAINING_FEATURES if c.startswith('outlier_')]:
        input_df[col] = False
    
    return input_df[TRAINING_FEATURES]

def make_predictions(input_df):
    """Generate predictions from both models"""
    try:
        # Scale features for logistic regression
        scaled_input = models['scaler'].transform(input_df)
        
        # Get predictions
        return {
            'logistic': {
                'prediction': models['logistic'].predict(scaled_input)[0],
                'probability': models['logistic'].predict_proba(scaled_input)[0][1]
            },
            'knn': {
                'prediction': models['knn'].predict(input_df)[0],
                'probability': models['knn'].predict_proba(input_df)[0][1]
            }
        }
    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")
        return None

def main():
    st.title("📊 Credit Scoring Dashboard")
    
    # Client form
    with st.form("client_info"):
        st.header("Client Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.number_input("Age", min_value=18, max_value=100, value=30)
            marital = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
            expenses = st.number_input("Monthly Expenses (€)", min_value=0.0, value=200.0)
        
        with col2:
            income = st.number_input("Monthly Income (€)", min_value=0.0, value=800.0)
            amount = st.number_input("Loan Amount (€)", min_value=0.0, value=1000.0)
            price = st.number_input("Purchase Value (€)", min_value=0.0, value=1200.0)
        
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
        
        with st.spinner("Analyzing..."):
            input_df = prepare_input(form_data)
            results = make_predictions(input_df)
        
        if results:
            # Display results
            st.header("Prediction Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Logistic Regression")
                pred = "Non-Solvent" if results['logistic']['prediction'] == 1 else "Solvent"
                st.metric("Status", pred, f"{results['logistic']['probability']:.1%} confidence")
            
            with col2:
                st.subheader("KNN Model")
                pred = "Non-Solvent" if results['knn']['prediction'] == 1 else "Solvent"
                st.metric("Status", pred, f"{results['knn']['probability']:.1%} confidence")
            
            # Comparison chart
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.bar(['Logistic', 'KNN'], 
                  [results['logistic']['probability'], results['knn']['probability']],
                  color=['blue', 'orange'])
            ax.set_ylim(0, 1)
            ax.set_ylabel('Non-Solvent Probability')
            st.pyplot(fig)

if __name__ == "__main__":
    main()
