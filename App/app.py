# Fichier : App/app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import pyreadstat

# -----------------------
# 📦 Chargement des modèles
# -----------------------
@st.cache_resource
def charger_modeles():
    logistic = joblib.load("models/logistic_model.pkl")
    knn = joblib.load("models/knn_model.pkl")  # Modèle KNN entraîné, pas la classe complète
    scaler = joblib.load("models/scaler.pkl")
    return logistic, knn, scaler

# -----------------------
# 📚 Chargement de scoring.sav
# -----------------------
@st.cache_data
def charger_base():
    df, meta = pyreadstat.read_sav("Data/scoring.sav")
    return df

# -----------------------
# 🚀 Application principale
# -----------------------
def main():
    st.set_page_config(page_title="🧠 Prédiction de Solvabilité", layout="centered")
    st.title("💳 Application de Scoring Bancaire")
    st.markdown("Saisissez les données d'un client pour prédire sa **solvabilité** à l’aide de deux modèles.")

    # Chargement des modèles & données
    logistic_model, knn_model, scaler = charger_modeles()
    df = charger_base()

    # Détecter les modalités de 'Marital'
    marital_options = sorted(df['Marital'].dropna().unique().tolist())
    marital_mapping = {val: idx for idx, val in enumerate(marital_options)}

    # Interface utilisateur
    st.subheader("📝 Saisie des caractéristiques du client")

    age = st.slider("Âge", min_value=18, max_value=100, value=30)
    marital = st.selectbox("Statut marital", marital_options)
    expenses = st.number_input("Dépenses mensuelles (€)", min_value=0, max_value=20000, value=1000)
    income = st.number_input("Revenu mensuel (€)", min_value=0, max_value=50000, value=3000)
    amount = st.number_input("Montant emprunté (€)", min_value=0, max_value=100000, value=10000)
    price = st.number_input("Prix de l'achat (€)", min_value=0, max_value=150000, value=12000)

    if st.button("📊 Prédire la solvabilité"):
        # Préparation des données
        donnees_client = pd.DataFrame([{
            "Age": age,
            "Marital": marital_mapping[marital],
            "Expenses": expenses,
            "Income": income,
            "Amount": amount,
            "Price": price
        }])

        # Mise à l'échelle
        donnees_scaled = scaler.transform(donnees_client)

        # Prédiction Logistique
        pred_log = logistic_model.predict(donnees_scaled)[0]

        # Prédiction KNN
        pred_knn = knn_model.predict(donnees_scaled)[0]

        # Résultats
        st.subheader("🔎 Résultat de la prédiction")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Modèle Logistique")
            if pred_log == 1:
                st.error("❌ Client non solvable")
            else:
                st.success("✅ Client solvable")

        with col2:
            st.markdown("### Modèle KNN")
            if pred_knn == 1:
                st.error("❌ Client non solvable")
            else:
                st.success("✅ Client solvable")

if __name__ == '__main__':
    main()



