# 📦 INSTALLATION DES LIBRAIRIES (si besoin sur Colab)
!pip install pyreadstat imbalanced-learn

# 📚 IMPORTATION DES LIBRAIRIES
import pyreadstat
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from google.colab import files
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    precision_recall_curve, PrecisionRecallDisplay, roc_auc_score
)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# 📂 CHARGEMENT DU FICHIER
uploaded = files.upload()
df, meta = pyreadstat.read_sav("/content/scoring.sav")

# 🔍 SUPPRESSION DES OUTLIERS
def detect_outliers_iqr(data, variable):
    Q1 = data[variable].quantile(0.25)
    Q3 = data[variable].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return ((data[variable] < lower_bound) | (data[variable] > upper_bound))

outlier_vars = ['Amount', 'Expenses', 'Price', 'Income']
for var in outlier_vars:
    df[f'out_{var}'] = detect_outliers_iqr(df, var)

df_clean = df[
    ~(df['out_Amount'] | df['out_Expenses'] | df['out_Price'] | df['out_Income'])
]

# 🧹 PRÉPARATION DES DONNÉES
X = df_clean.drop(columns=['Statut1'] + [f'out_{var}' for var in outlier_vars])
y = df_clean['Statut1']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🌐 NORMALISATION & RÉÉCHANTILLONNAGE
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

sm = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = sm.fit_resample(X_train_scaled, y_train)

# 🔁 VALIDATION CROISÉE & RECHERCHE D’HYPERPARAMÈTRES
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression(random_state=42, class_weight='balanced'))
])

param_grid = {
    'model__C': np.logspace(-3, 2, 6),
    'model__penalty': ['l1', 'l2'],
    'model__solver': ['liblinear'],
    'model__max_iter': [500]
}

grid_search = GridSearchCV(
    pipeline,
    param_grid,
    scoring='f1',
    cv=5,
    verbose=2,
    n_jobs=-1
)

grid_search.fit(X_train_scaled, y_train)
best_model = grid_search.best_estimator_
print("🔍 Meilleurs paramètres :", grid_search.best_params_)
print("🏆 Meilleur score F1 :", grid_search.best_score_)

# 🧪 ÉVALUATION DU MODÈLE SUR LE TEST
y_proba = best_model.predict_proba(X_test_scaled)[:, 1]
precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)
f1_scores = (2 * precisions * recalls) / (precisions + recalls + 1e-8)
optimal_idx = np.argmax(f1_scores)
optimal_threshold = thresholds[optimal_idx]
print(f"🎯 Seuil optimal (max F1): {optimal_threshold:.4f}")

y_pred_optimized = (y_proba >= optimal_threshold).astype(int)
print("\n📊 Rapport de classification optimisé :")
print(classification_report(y_test, y_pred_optimized, target_names=["Solvable", "Non Solvable"]))

# 📉 COURBE & MATRICE DE CONFUSION
plt.figure(figsize=(14, 5))
plt.subplot(1, 2, 1)
cm = confusion_matrix(y_test, y_pred_optimized)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=["Solvable", "Non Solvable"],
            yticklabels=["Solvable", "Non Solvable"])
plt.title("Matrice de confusion")

plt.subplot(1, 2, 2)
PrecisionRecallDisplay.from_estimator(best_model, X_test_scaled, y_test)
plt.title("Courbe Precision-Recall")
plt.tight_layout()
plt.show()

# 💾 SAUVEGARDE DU MODÈLE
model_data = {
    'model': best_model,
    'threshold': optimal_threshold,
    'scaler': scaler,
    'features': list(X.columns)
}
joblib.dump(model_data, "logistic_model.pkl")
print("✅ Modèle sauvegardé : logistic_model.pkl")





