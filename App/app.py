# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Set page config
st.set_page_config(
    page_title="Credit Scoring App",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load models (replace with your actual model paths)
@st.cache_resource
def load_models():
    try:
        log_model = joblib.load('logistic_model.joblib')
        best_knn = joblib.load('KNN (1).joblib')
        scaler = joblib.load('scaler (1).joblib')
        return log_model, best_knn, scaler
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None

log_model, best_knn, scaler = load_models()

# Feature information from your project
FEATURES = {
    'Age': {
        'description': "Age of the client (in years)",
        'dtype': 'int',
        'range': (18, 100)
    },
    'Marital': {
        'description': "Marital status (1=Célibataire, 2=Marié, 3=Divorcé)",
        'dtype': 'int',
        'categories': [1, 2, 3]
    },
    'Expenses': {
        'description': "Monthly expenses of the client",
        'dtype': 'float',
        'range': (0, 1000)
    },
    'Income': {
        'description': "Monthly income of the client",
        'dtype': 'float',
        'range': (0, 5000)
    },
    'Amount': {
        'description': "Loan amount requested",
        'dtype': 'float',
        'range': (0, 10000)
    },
    'Price': {
        'description': "Price of the item being financed",
        'dtype': 'float',
        'range': (0, 15000)
    }
}

# Prediction function
def predict_credit_risk(input_data):
    try:
        # Create DataFrame
        input_df = pd.DataFrame([input_data])
        
        # Scale features for logistic regression
        scaled_features = scaler.transform(input_df)
        
        # Get predictions
        log_pred = log_model.predict(scaled_features)[0]
        log_proba = log_model.predict_proba(scaled_features)[0]
        
        knn_pred = best_knn.predict(input_df)[0]
        knn_proba = best_knn.predict_proba(input_df)[0]
        
        return {
            'logistic': {
                'prediction': 'Non solvable' if log_pred == 1 else 'Solvable',
                'probability': log_proba[1] if log_pred == 1 else log_proba[0],
                'class': log_pred
            },
            'knn': {
                'prediction': 'Non solvable' if knn_pred == 1 else 'Solvable',
                'probability': knn_proba[1] if knn_pred == 1 else knn_proba[0],
                'class': knn_pred
            }
        }
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None

# Main app
def main():
    st.title("📊 Credit Scoring Dashboard")
    st.markdown("""
    This app predicts whether a client is likely to default on a loan (Non solvable) 
    or not (Solvable) using machine learning models.
    """)
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.radio("Choose a page", 
                               ["Single Prediction", "Batch Prediction", "Model Information"])
    
    if app_mode == "Single Prediction":
        st.header("Single Client Prediction")
        st.markdown("Enter the client's information to get a credit risk prediction.")
        
        # Create input form
        with st.form("client_info"):
            cols = st.columns(2)
            input_data = {}
            
            # First column
            with cols[0]:
                input_data['Age'] = st.slider(
                    "Age", 
                    min_value=FEATURES['Age']['range'][0],
                    max_value=FEATURES['Age']['range'][1],
                    value=30,
                    help=FEATURES['Age']['description']
                )
                
                input_data['Marital'] = st.selectbox(
                    "Marital Status",
                    options=FEATURES['Marital']['categories'],
                    format_func=lambda x: {1: "Célibataire", 2: "Marié", 3: "Divorcé"}.get(x),
                    help=FEATURES['Marital']['description']
                )
                
                input_data['Expenses'] = st.number_input(
                    "Monthly Expenses",
                    min_value=FEATURES['Expenses']['range'][0],
                    max_value=FEATURES['Expenses']['range'][1],
                    value=500.0,
                    step=10.0,
                    help=FEATURES['Expenses']['description']
                )
            
            # Second column
            with cols[1]:
                input_data['Income'] = st.number_input(
                    "Monthly Income",
                    min_value=FEATURES['Income']['range'][0],
                    max_value=FEATURES['Income']['range'][1],
                    value=2000.0,
                    step=100.0,
                    help=FEATURES['Income']['description']
                )
                
                input_data['Amount'] = st.number_input(
                    "Loan Amount Requested",
                    min_value=FEATURES['Amount']['range'][0],
                    max_value=FEATURES['Amount']['range'][1],
                    value=5000.0,
                    step=100.0,
                    help=FEATURES['Amount']['description']
                )
                
                input_data['Price'] = st.number_input(
                    "Price of Item Being Financed",
                    min_value=FEATURES['Price']['range'][0],
                    max_value=FEATURES['Price']['range'][1],
                    value=6000.0,
                    step=100.0,
                    help=FEATURES['Price']['description']
                )
            
            submitted = st.form_submit_button("Predict Credit Risk")
        
        if submitted:
            with st.spinner("Making predictions..."):
                results = predict_credit_risk(input_data)
                
                if results:
                    st.success("Prediction completed!")
                    
                    # Display results in columns
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("Logistic Regression")
                        st.metric(
                            "Prediction", 
                            results['logistic']['prediction'],
                            f"{results['logistic']['probability']:.1%} confidence"
                        )
                        
                        # Progress bar for probability
                        st.progress(results['logistic']['probability'])
                        
                    with col2:
                        st.subheader("KNN Classifier")
                        st.metric(
                            "Prediction", 
                            results['knn']['prediction'],
                            f"{results['knn']['probability']:.1%} confidence"
                        )
                        
                        # Progress bar for probability
                        st.progress(results['knn']['probability'])
                    
                    # Visual comparison
                    fig, ax = plt.subplots(figsize=(8, 4))
                    models = ['Logistic Regression', 'KNN']
                    probas = [results['logistic']['probability'], results['knn']['probability']]
                    colors = ['#1f77b4', '#ff7f0e']
                    
                    bars = ax.bar(models, probas, color=colors)
                    ax.set_ylim(0, 1)
                    ax.set_ylabel('Probability of Non-Solvable')
                    ax.set_title('Model Comparison')
                    
                    # Add value labels
                    for bar in bars:
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height,
                                f'{height:.1%}',
                                ha='center', va='bottom')
                    
                    st.pyplot(fig)
    
    elif app_mode == "Batch Prediction":
        st.header("Batch Prediction")
        st.markdown("Upload a CSV file with multiple clients' data to get predictions.")
        
        uploaded_file = st.file_uploader(
            "Upload CSV file", 
            type=["csv"],
            help="File should contain columns: Age, Marital, Expenses, Income, Amount, Price"
        )
        
        if uploaded_file is not None:
            try:
                # Read the file
                batch_df = pd.read_csv(uploaded_file)
                
                # Check required columns
                required_cols = list(FEATURES.keys())
                missing_cols = [col for col in required_cols if col not in batch_df.columns]
                
                if missing_cols:
                    st.error(f"Missing columns in uploaded file: {', '.join(missing_cols)}")
                else:
                    st.success("File uploaded successfully!")
                    st.dataframe(batch_df.head())
                    
                    if st.button("Predict for Batch"):
                        with st.spinner("Processing batch predictions..."):
                            # Scale features
                            scaled_batch = scaler.transform(batch_df[required_cols])
                            
                            # Get predictions
                            batch_df['Logistic_Prediction'] = log_model.predict(scaled_batch)
                            batch_df['Logistic_Probability'] = log_model.predict_proba(scaled_batch)[:, 1]
                            batch_df['KNN_Prediction'] = best_knn.predict(batch_df[required_cols])
                            batch_df['KNN_Probability'] = best_knn.predict_proba(batch_df[required_cols])[:, 1]
                            
                            # Format results
                            batch_df['Logistic_Result'] = batch_df['Logistic_Prediction'].map(
                                {0: 'Solvable', 1: 'Non solvable'})
                            batch_df['KNN_Result'] = batch_df['KNN_Prediction'].map(
                                {0: 'Solvable', 1: 'Non solvable'})
                            
                            st.success("Batch predictions completed!")
                            
                            # Show results
                            st.dataframe(batch_df)
                            
                            # Download button
                            csv = batch_df.to_csv(index=False).encode('utf-8')
                            st.download_button(
                                "Download Predictions",
                                csv,
                                "credit_predictions.csv",
                                "text/csv",
                                key='download-csv'
                            )
                            
                            # Summary statistics
                            st.subheader("Prediction Summary")
                            summary_cols = st.columns(2)
                            
                            with summary_cols[0]:
                                st.markdown("**Logistic Regression**")
                                st.write(batch_df['Logistic_Result'].value_counts())
                                
                            with summary_cols[1]:
                                st.markdown("**KNN Classifier**")
                                st.write(batch_df['KNN_Result'].value_counts())
            
            except Exception as e:
                st.error(f"Error processing file: {e}")
    
    elif app_mode == "Model Information":
        st.header("Model Information")
        st.markdown("Details about the machine learning models used for prediction.")
        
        # Model details
        st.subheader("Models Used")
        
        with st.expander("Logistic Regression"):
            st.markdown("""
            - **Type**: Binary classification
            - **Class weights**: Balanced (to handle imbalanced data)
            - **Features used**: All numerical features standardized
            - **Performance**: AUC score of ~0.85 on test set
            """)
            
            # Confusion matrix (example from your code)
            cm = np.array([[150, 20], [30, 50]])  # Example values
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=["Solvable", "Non solvable"],
                        yticklabels=["Solvable", "Non solvable"],
                        ax=ax)
            ax.set_title("Example Confusion Matrix")
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            st.pyplot(fig)
        
        with st.expander("K-Nearest Neighbors (KNN)"):
            st.markdown("""
            - **Type**: Instance-based classification
            - **Optimal parameters**: Found via grid search
              - n_neighbors: 5
              - weights: distance
              - p: 2 (Euclidean distance)
            - **Performance**: AUC score of ~0.87 on test set
            """)
            
            # Feature importance (example)
            feature_importance = pd.DataFrame({
                'Feature': list(FEATURES.keys()),
                'Importance': [0.25, 0.15, 0.20, 0.10, 0.20, 0.10]  # Example values
            }).sort_values('Importance', ascending=False)
            
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.barplot(x='Importance', y='Feature', data=feature_importance, palette='viridis', ax=ax)
            ax.set_title("Feature Importance (Example)")
            st.pyplot(fig)
        
        # Feature details
        st.subheader("Feature Descriptions")
        feature_table = pd.DataFrame.from_dict(FEATURES, orient='index')
        feature_table.index.name = 'Feature'
        st.table(feature_table[['description', 'dtype']])

if __name__ == "__main__":
    main()
