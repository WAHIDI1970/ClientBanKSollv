# app.py
import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

# Set page config
st.set_page_config(
    page_title="Credit Scoring App",
    page_icon="📊",
    layout="wide"
)

# Load models
@st.cache_resource
def load_models():
    try:
        log_model = joblib.load('logistic_model.joblib')
        best_knn = joblib.load('best_knn_model.joblib')
        scaler = joblib.load('scaler.joblib')
        return log_model, best_knn, scaler
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None

log_model, best_knn, scaler = load_models()

# Feature mapping - adjust based on your actual training features
FEATURE_MAPPING = {
    'Age': 'Age',
    'Monthly Income (€)': 'Income',
    'Marital Status': 'Marital',
    'Loan Amount (€)': 'Amount',
    'Monthly Expenses (€)': 'Expenses',
    'Purchase Value (€)': 'Price'
}

# Original features used in training (update this list with your actual features)
TRAINING_FEATURES = [
    'Age', 'Marital', 'Expenses', 'Income', 'Amount', 'Price',
    'outlier_amount', 'outlier_expenses', 'outlier_income', 'outlier_price'
]

def prepare_input_data(form_data):
    """Convert form data to model input format"""
    # Map form fields to model features
    input_data = {
        'Age': float(form_data['Age']),
        'Marital': 1 if form_data['Marital Status'] == 'Single' else 2,  # Adjust based on your encoding
        'Expenses': float(form_data['Monthly Expenses (€)']),
        'Income': float(form_data['Monthly Income (€)']),
        'Amount': float(form_data['Loan Amount (€)']),
        'Price': float(form_data['Purchase Value (€)']),
        # Add outlier columns with default False values
        'outlier_amount': False,
        'outlier_expenses': False,
        'outlier_income': False,
        'outlier_price': False
    }
    
    # Create DataFrame with correct column order
    input_df = pd.DataFrame([input_data])[TRAINING_FEATURES]
    
    return input_df

def predict_credit_risk(input_df):
    """Make predictions using both models"""
    try:
        # Scale features for logistic regression
        scaled_features = scaler.transform(input_df)
        
        # Get predictions
        log_pred = log_model.predict(scaled_features)[0]
        log_proba = log_model.predict_proba(scaled_features)[0]
        
        knn_pred = best_knn.predict(input_df)[0]
        knn_proba = best_knn.predict_proba(input_df)[0]
        
        return {
            'logistic': {
                'prediction': 'Non-Solvent' if log_pred == 1 else 'Solvent',
                'probability': log_proba[1] if log_pred == 1 else log_proba[0],
                'class': log_pred
            },
            'knn': {
                'prediction': 'Non-Solvent' if knn_pred == 1 else 'Solvent',
                'probability': knn_proba[1] if knn_pred == 1 else knn_proba[0],
                'class': knn_pred
            }
        }
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None

# Main app
def main():
    st.title("📊 Credit Scoring Dashboard")
    
    # Client information form
    with st.form("client_info"):
        st.header("Client Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.number_input("Age", min_value=18, max_value=100, value=30)
            marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
            expenses = st.number_input("Monthly Expenses (€)", min_value=0, value=200)
        
        with col2:
            income = st.number_input("Monthly Income (€)", min_value=0, value=800)
            loan_amount = st.number_input("Loan Amount (€)", min_value=0, value=1000)
            purchase_value = st.number_input("Purchase Value (€)", min_value=0, value=1200)
        
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
        
        # Prepare model input
        input_df = prepare_input_data(form_data)
        
        # Make predictions
        with st.spinner("Analyzing client data..."):
            results = predict_credit_risk(input_df)
        
        if results:
            st.header("Dashboard")
            st.subheader("Compare predictions from Logistic Regression and KNN models")
            
            # Display results
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Logistic Regression**")
                st.metric(
                    "Client Solvency Status (Statut)",
                    results['logistic']['prediction'],
                    f"{results['logistic']['probability']:.1%} confidence"
                )
            
            with col2:
                st.markdown("**KNN Model**")
                st.metric(
                    "Client Solvency Status (Statut)",
                    results['knn']['prediction'],
                    f"{results['knn']['probability']:.1%} confidence"
                )
            
            # Visual comparison
            st.subheader("Model Comparison")
            fig, ax = plt.subplots()
            models = ['Logistic Regression', 'KNN']
            probas = [results['logistic']['probability'], results['knn']['probability']]
            ax.bar(models, probas, color=['blue', 'orange'])
            ax.set_ylabel('Probability')
            ax.set_title('Probability of Non-Solvency')
            st.pyplot(fig)

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    main()
