import streamlit as st
import joblib
import numpy as np

# ⏳ Chargement des modèles
@st.cache_resource
def charger_modeles():
    modele_logistique = joblib.load("models/logistic_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    return modele_logistique, scaler

# Interface principale
def main():
    st.set_page_config(page_title="Scoring Bancaire", layout="centered")
    st.title("🔍 Prédiction de la Solvabilité Bancaire")
    st.markdown("Entrez les caractéristiques du client pour prédire sa solvabilité.")

    # Champs à remplir (correspondant aux features du modèle)
    age = st.slider("Âge", 18, 90, 30)
    marital = st.selectbox("Statut matrimonial", options=[0, 1])  # Adapter selon codage (0 = célibataire, 1 = marié ?)
    expenses = st.number_input("Dépenses mensuelles", min_value=0.0, value=500.0)
    income = st.number_input("Revenu mensuel", min_value=0.0, value=2500.0)
    amount = st.number_input("Montant du crédit", min_value=0.0, value=5000.0)
    price = st.number_input("Prix total du bien", min_value=0.0, value=15000.0)

    # Charger modèle
    model, scaler = charger_modeles()

    if st.button("🔮 Prédire"):
        # Préparation des données
        input_data = np.array([[age, marital, expenses, income, amount, price]])
        input_scaled = scaler.transform(input_data)

        # Prédiction
        prediction = model.predict(input_scaled)[0]

        # Affichage du résultat
        if prediction == 0:
            st.success("✅ Client **Solvable**")
        else:
            st.error("❌ Client **Non solvable**")

if __name__ == "__main__":
    main()




