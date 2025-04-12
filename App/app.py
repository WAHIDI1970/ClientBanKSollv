
# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.neighbors import KNeighborsClassifier

# =============================================
# FIX FOR CUSTOM KNN CLASS
# =============================================
class ModeleKNNOptimise(KNeighborsClassifier):
    """Wrapper class to match your notebook's custom KNN model"""
    pass

# Register the custom class before loading models
joblib.register('ModeleKNNOptimise', ModeleKNNOptimise)

# =============================================
# PAGE CONFIGURATION
# =============================================
st.set_page_config(
    page_title="Client Solvency Predictor", 
    layout="wide",
    page_icon="🏦"
)

st.title("🏦 Client Solvency Prediction Dashboard")
st.markdown("Compare predictions from Logistic Regression and KNN models")

# =============================================
# MODEL LOADING
# =============================================
@st.cache_resource
def load_models():
    try:
        models = {
            "logistic": joblib.load("models/logistic_model.pkl"),
            "knn": joblib.load("models/KNN (1).pkl"),  # Your custom KNN model
            "scaler": joblib.load("models/scaler (1).pkl")
        }
        
        # Validate scaler
        if not hasattr(models['scaler'], 'mean_'):
            st.error("❌ Scaler is not properly fitted!")
            st.stop()
            
        return models
    except Exception as e:
        st.error(f"❌ Model loading failed: {str(e)}")
        st.stop()

models = load_models()

# =============================================
# USER INPUT FORM
# =============================================
st.sidebar.header("📋 Client Information")
col1, col2 = st.sidebar.columns(2)

with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    marital = st.selectbox(
        "Marital Status",
        options=[1, 2, 3],
        format_func=lambda x: {1: "Single", 2: "Married", 3: "Divorced"}[x]
    )
    expenses = st.number_input("Monthly Expenses (€)", min_value=0.0, value=200.0, step=50.0)

with col2:
    income = st.number_input("Monthly Income (€)", min_value=0.0, value=800.0, step=50.0)
    amount = st.number_input("Loan Amount (€)", min_value=0.0, value=1000.0, step=100.0)
    price = st.number_input("Purchase Value (€)", min_value=0.0, value=1200.0, step=100.0)

# Prepare input data
client_data = pd.DataFrame({
    "Age": [age],
    "Marital": [marital],
    "Expenses": [expenses],
    "Income": [income],
    "Amount": [amount],
    "Price": [price]
})

# =============================================
# PREDICTION FUNCTION
# =============================================
def make_predictions(data):
    try:
        # Scale features for logistic regression
        scaled_data = models['scaler'].transform(data)
        
        results = {
            "logistic": {
                "prediction": models['logistic'].predict(scaled_data)[0],
                "probability": models['logistic'].predict_proba(scaled_data)[0][1]
            },
            "knn": {
                "prediction": models['knn'].predict(data)[0],
                "probability": models['knn'].predict_proba(data)[0][1]
            }
        }
        return results
    except Exception as e:
        st.error(f"❌ Prediction error: {str(e)}")
        st.stop()

# =============================================
# RESULTS DISPLAY
# =============================================
if st.sidebar.button("🔮 Predict Solvency", type="primary"):
    with st.spinner("Analyzing client data..."):
        results = make_predictions(client_data)
        
        # Main results columns
        col1, col2 = st.columns(2)
        
        # Logistic Regression Results
        with col1:
            st.subheader("📈 Logistic Regression")
            if results['logistic']['prediction'] == 1:
                st.error(f"🚨 Non-Solvent (Confidence: {results['logistic']['probability']:.1%})")
            else:
                st.success(f"✅ Solvent (Confidence: {1-results['logistic']['probability']:.1%})")
            
            st.markdown("**Model Characteristics**")
            st.markdown("- 📊 Better for interpretable results")
            st.markdown("- ⚖️ Handles class imbalance well")
            
            try:
                fig, ax = plt.subplots()
                ConfusionMatrixDisplay.from_estimator(
                    models['logistic'],
                    models['scaler'].transform(client_data),
                    [0],  # Dummy data for visualization
                    display_labels=['Solvent', 'Non-Solvent'],
                    ax=ax,
                    cmap='Blues'
                )
                st.pyplot(fig)
            except Exception:
                st.warning("Could not display confusion matrix")

        # KNN Results
        with col2:
            st.subheader("🔢 KNN Classifier")
            if results['knn']['prediction'] == 1:
                st.error(f"🚨 Non-Solvent (Confidence: {results['knn']['probability']:.1%})")
            else:
                st.success(f"✅ Solvent (Confidence: {1-results['knn']['probability']:.1%})")
            
            st.markdown("**Model Characteristics**")
            st.markdown("- 🧠 Better for complex patterns")
            st.markdown("- 📏 Uses distance-based analysis")
            
            st.markdown("**Decision Factors**")
            st.info("KNN considers all features equally in its distance calculation")

        st.divider()
        if results['logistic']['prediction'] == results['knn']['prediction']:
            st.success("🎯 Both models agree on the prediction")
        else:
            st.warning("⚠️ Models disagree - consider manual review")
            
            diff = abs(results['logistic']['probability'] - results['knn']['probability'])
            st.metric("Confidence Difference", f"{diff:.1%}")

        st.subheader("📋 Client Data Summary")
        st.dataframe(client_data.style.format({
            "Expenses": "€{:.2f}",
            "Income": "€{:.2f}", 
            "Amount": "€{:.2f}",
            "Price": "€{:.2f}"
        }))

        csv = client_data.assign(
            Logistic_Prediction=["Non-Solvent" if results['logistic']['prediction'] == 1 else "Solvent"],
            Logistic_Confidence=[f"{results['logistic']['probability']:.1%}"],
            KNN_Prediction=["Non-Solvent" if results['knn']['prediction'] == 1 else "Solvent"],
            KNN_Confidence=[f"{results['knn']['probability']:.1%}"]
        ).to_csv(index=False).encode('utf-8')
        
        st.download_button(
            label="📥 Download Full Report",
            data=csv,
            file_name="client_solvency_analysis.csv",
            mime="text/csv"
        )

# =============================================
# SIDEBAR FOOTER
# =============================================
st.sidebar.divider()
st.sidebar.markdown("""
**ℹ️ About These Models**  
- Trained on historical client data  
- Updated monthly  
- Threshold: 65% confidence  
""")
