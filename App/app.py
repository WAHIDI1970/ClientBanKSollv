# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import joblib

# Configuration de la page
st.set_page_config(
    page_title="Prédiction Solvabilité Client", 
    layout="centered",
    page_icon="🔍"
)

# Titre et description
st.title("🔍 Prédiction du Statut de Solvabilité d'un Client")
st.markdown("""
Cette application prédit si un client est **solvable** ou **non solvable** à partir de ses informations financières.
""")

# Chargement des modèles avec cache pour améliorer les performances
@st.cache_resource
def load_models():
    try:
        model = joblib.load("models/logistic_model.pkl")
        scaler = joblib.load("models/scaler.pkl")
        return model, scaler
    except FileNotFoundError:
        st.error("❌ Modèles non trouvés. Assurez-vous que les fichiers 'logistic_model.pkl' et 'scaler.pkl' existent dans le dossier 'models/'.")
        st.stop()

model, scaler = load_models()

# Sidebar - Entrée utilisateur
st.sidebar.header("🧾 Informations du client")

# Dictionnaire pour les labels du statut marital
marital_labels = {
    1: "Célibataire", 
    2: "Marié(e)", 
    3: "Divorcé(e)"
}

age = st.sidebar.number_input("Âge", min_value=18, max_value=100, value=30)
marital = st.sidebar.selectbox(
    "Statut marital", 
    options=list(marital_labels.keys()), 
    format_func=lambda x: marital_labels[x]
)
expenses = st.sidebar.number_input("Dépenses mensuelles (€)", min_value=0.0, max_value=10000.0, value=200.0, step=50.0)
income = st.sidebar.number_input("Revenu mensuel (€)", min_value=0.0, max_value=50000.0, value=800.0, step=50.0)
amount = st.sidebar.number_input("Montant du crédit demandé (€)", min_value=0.0, max_value=100000.0, value=1000.0, step=100.0)
price = st.sidebar.number_input("Valeur de l'achat (€)", min_value=0.0, max_value=150000.0, value=1200.0, step=100.0)

# Validation des données
if income <= expenses:
    st.sidebar.warning("⚠️ Le revenu doit être supérieur aux dépenses.")

# Préparation des données
user_data = pd.DataFrame({
    "Age": [age],
    "Marital": [marital],
    "Expenses": [expenses],
    "Income": [income],
    "Amount": [amount],
    "Price": [price]
})

# Affichage des données saisies
st.subheader("📋 Données client")
st.dataframe(
    user_data.rename(columns={
        "Marital": "Statut marital",
        "Expenses": "Dépenses (€)",
        "Income": "Revenu (€)",
        "Amount": "Montant crédit (€)",
        "Price": "Valeur achat (€)"
    }).style.format({
        "Dépenses (€)": "{:.2f}",
        "Revenu (€)": "{:.2f}",
        "Montant crédit (€)": "{:.2f}",
        "Valeur achat (€)": "{:.2f}"
    })
)

# Bouton de prédiction
if st.sidebar.button("⚙️ Prédire", type="primary"):
    with st.spinner("Analyse en cours..."):
        try:
            # Transformation et prédiction
            user_scaled = scaler.transform(user_data)
            prediction = model.predict(user_scaled)
            proba = model.predict_proba(user_scaled)[:, 1]
            
            # Affichage des résultats
            if prediction[0] == 1:
                st.error(f"🔴 **Non solvable** (probabilité: {proba[0]:.1%})")
                st.progress(proba[0])
            else:
                st.success(f"🟢 **Solvable** (probabilité: {1-proba[0]:.1%})")
                st.progress(1 - proba[0])
            
            # Téléchargement CSV
            result_df = user_data.copy()
            result_df['Statut Prédit'] = ['Non solvable' if prediction[0] == 1 else 'Solvable']
            result_df['Probabilité (Non Solvable)'] = [f"{proba[0]:.1%}"]
            
            csv = result_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Exporter les résultats (CSV)",
                data=csv,
                file_name="prediction_client.csv",
                mime='text/csv',
                help="Télécharger toutes les informations au format CSV"
            )
            
        except Exception as e:
            st.error(f"❌ Erreur lors de la prédiction: {str(e)}")
            st.exception(e)

# Note informative
st.caption("ℹ️ Les prédictions sont basées sur un modèle statistique et doivent être interprétées avec prudence.")

