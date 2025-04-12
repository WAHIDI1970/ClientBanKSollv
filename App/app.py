# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration de la page
st.set_page_config(
    page_title="Credit Scoring App",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Chargement des modèles
@st.cache_resource
def load_models():
    try:
        log_model = joblib.load("models/logistic_model.pkl")
        knn_model = joblib.load("models/knn_model.pkl")
        scaler = joblib.load("models/scaler.pkl")
        return log_model, knn_model, scaler
    except Exception as e:
        st.error(f"Erreur lors du chargement des modèles : {e}")
        return None, None, None

log_model, knn_model, scaler = load_models()

# Caractéristiques des variables
FEATURES = {
    'Age': {'description': "Âge du client", 'dtype': 'int', 'range': (18, 100)},
    'Marital': {'description': "Statut matrimonial (1=Célibataire, 2=Marié, 3=Divorcé)", 'dtype': 'int', 'categories': [1, 2, 3]},
    'Expenses': {'description': "Dépenses mensuelles", 'dtype': 'float', 'range': (0, 1000)},
    'Income': {'description': "Revenu mensuel", 'dtype': 'float', 'range': (0, 5000)},
    'Amount': {'description': "Montant du crédit demandé", 'dtype': 'float', 'range': (0, 10000)},
    'Price': {'description': "Prix du bien financé", 'dtype': 'float', 'range': (0, 15000)}
}

# Fonction de prédiction
def predict_credit_risk(input_data):
    try:
        input_df = pd.DataFrame([input_data])
        scaled_features = scaler.transform(input_df)
        
        log_pred = log_model.predict(scaled_features)[0]
        log_proba = log_model.predict_proba(scaled_features)[0]
        
        knn_pred = knn_model.predict(input_df)[0]
        knn_proba = knn_model.predict_proba(input_df)[0]
        
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
        st.error(f"Erreur de prédiction : {e}")
        return None

# Lancement de l'application
if __name__ == "__main__":
    from credit_app_ui import main
    main(FEATURES, predict_credit_risk)

