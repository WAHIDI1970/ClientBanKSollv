import streamlit as st
import numpy as np
import joblib

# Import de la classe personnalisée
from modele_knn import ModeleKNNOptimise

# -------------------------------
# Chargement du modèle
# -------------------------------
@st.cache_resource
def load_model():
    try:
        model = joblib.load("models/ModeleKNNOptimise.pkl")
        return model
    except Exception as e:
        st.error(f"Erreur lors du chargement du modèle : {e}")
        return None

# -------------------------------
# Interface Utilisateur
# -------------------------------
def main():
    st.set_page_config(page_title="Scoring Client", layout="centered")
    st.title("📊 Application de Scoring Bancaire")
    st.write("Remplissez les informations du client pour prédire sa solvabilité.")

    # Exemple de champs à adapter à tes vraies variables
    age = st.slider("Âge", 18, 70, 30)
    revenu = st.number_input("Revenu mensuel (€)", min_value=0, value=2500)
    montant_credit = st.number_input("Montant du crédit demandé (€)", min_value=0, value=10000)
    duree_credit = st.slider("Durée du crédit (mois)", 6, 60, 24)
    nombre_enfants = st.slider("Nombre d’enfants à charge", 0, 5, 0)

    # Transformer en array numpy pour prédiction
    input_data = np.array([[age, revenu, montant_credit, duree_credit, nombre_enfants]])

    # Bouton pour prédire
    if st.button("Prédire le statut du client"):
        model = load_model()
        if model:
            prediction = model.predict(input_data)[0]
            proba = model.predict_proba(input_data)[0][1]  # proba d'être non solvable

            if prediction == 1:
                st.error(f"❌ Le client est prédit **non solvable**. (Risque: {proba:.2%})")
            else:
                st.success(f"✅ Le client est prédit **solvable**. (Risque: {proba:.2%})")

# -------------------------------
if __name__ == "__main__":
    main()


